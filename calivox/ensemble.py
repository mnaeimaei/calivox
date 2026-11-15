"""Calibrated ensembles: member calibration -> equal-weight logit mean -> final temperature.

Selection is greedy forward search over a shortlist of runs, minimising cross-fitted
log loss. Because choosing members on the same OOF predictions you report is optimistic,
the selection is also repeated inside outer folds (*nested* validation): members are
chosen and the final temperature fitted on the outer-training subjects only, then scored
on the outer-test subjects. The nested log loss is the honest estimate.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .calibration import Calibrator, cross_fit, evaluate_calibration, fit_calibrator
from .metrics import binary_log_loss, metrics_from_logits, multiclass_log_loss, sigmoid, softmax
from .training import load_oof


def _ll(z: np.ndarray, y: np.ndarray) -> float:
    return binary_log_loss(y, sigmoid(z)) if z.ndim == 1 else multiclass_log_loss(y, softmax(z))


def load_members(run_dirs: list[Path], methods: list[str], folds: int, seed: int,
                  forced: dict[str, str] | None = None) -> dict:
    """Per run: raw OOF logits, cross-fitted calibrated logits, and the deployed calibrator.
    ``forced`` maps a run name to a calibration method to use instead of the best one."""
    members, ids0, y0 = {}, None, None
    forced = forced or {}
    for d in run_dirs:
        z, y, frame = load_oof(d)
        order = np.argsort(frame["id"].astype(str).to_numpy())
        z, y, ids = z[order], y[order], frame["id"].astype(str).to_numpy()[order]
        if ids0 is None:
            ids0, y0 = ids, y
        elif not (np.array_equal(ids, ids0) and np.array_equal(y, y0)):
            raise ValueError(f"{d.name}: different subjects or labels than the other runs")
        cal_file = d / "calibration.json"
        if d.name in forced:
            method = forced[d.name]
            cal = evaluate_calibration(z, y, [method], folds, seed)
        else:
            cal = json.loads(cal_file.read_text()) if cal_file.exists() else evaluate_calibration(z, y, methods, folds, seed)
        members[d.name] = {"dir": d, "raw": z, "cross": cross_fit(cal["selected"], z, y, folds, seed),
                           "method": cal["selected"], "calibrator": cal["calibrator"],
                           "calibrated_log_loss": cal["methods"][cal["selected"]]["log_loss"]}
    return {"members": members, "y": y0, "ids": ids0}


def _score(Z: list[np.ndarray], y: np.ndarray, folds: int, seed: int, final_t: bool) -> float:
    zm = np.mean(Z, axis=0)
    return _ll(cross_fit("temperature", zm, y, folds, seed) if final_t else zm, y)


def greedy_select(cands: dict[str, np.ndarray], y: np.ndarray, max_members: int, folds: int, seed: int,
                  final_t: bool = True) -> tuple[list[str], float]:
    chosen: list[str] = []
    best = np.inf
    while len(chosen) < max_members:
        trial = {n: _score([cands[c] for c in chosen] + [cands[n]], y, folds, seed, final_t)
                 for n in cands if n not in chosen}
        if not trial:
            break
        name, score = min(trial.items(), key=lambda kv: kv[1])
        if score >= best - 1e-6:
            break
        chosen.append(name)
        best = score
    return chosen, best


def build_ensemble(run_dirs: list[str | Path], max_members: int = 5, candidates: int = 12,
                   methods: tuple = ("none", "temperature", "platt", "isotonic"), folds: int = 5,
                   seed: int = 2026, final_temperature: bool = True, nested: bool = True,
                   threshold: float = 0.5, fixed: dict[str, str | None] | None = None) -> dict:
    """Greedy selection over ``run_dirs``; or, with ``fixed`` ({run name: calibration method
    or None for the best}), the given members only (no selection)."""
    run_dirs = [Path(d) for d in run_dirs]
    if fixed:
        missing = set(fixed) - {d.name for d in run_dirs}
        if missing:
            raise ValueError(f"fixed members not among the runs: {sorted(missing)}")
        run_dirs = [d for d in run_dirs if d.name in fixed]
    forced = {k: v for k, v in (fixed or {}).items() if v}
    pool = load_members(run_dirs, list(methods), folds, seed, forced)
    mem, y = pool["members"], pool["y"]
    binary = next(iter(mem.values()))["raw"].ndim == 1
    shortlist = list(fixed) if fixed else sorted(mem, key=lambda n: mem[n]["calibrated_log_loss"])[:candidates]
    shortlist = sorted(shortlist, key=lambda n: mem[n]["calibrated_log_loss"])
    cross = {n: mem[n]["cross"] for n in shortlist}

    if fixed:
        chosen = list(fixed)
    else:
        chosen, _ = greedy_select(cross, y, max_members, folds, seed, final_temperature)
    zm_cross = np.mean([cross[c] for c in chosen], axis=0)
    z_cross_final = cross_fit("temperature", zm_cross, y, folds, seed) if final_temperature else zm_cross
    # deployed: calibrators and temperature fitted on all subjects
    zm_deployed = np.mean([Calibrator.from_dict(mem[c]["calibrator"]).apply(mem[c]["raw"]) for c in chosen], axis=0)
    T = fit_calibrator("temperature", zm_deployed, y).params["T"] if final_temperature else 1.0

    report = {"members": [{"run": c, "dir": str(mem[c]["dir"]), "calibration": mem[c]["method"],
                           "calibrator": mem[c]["calibrator"],
                           "single_log_loss": mem[c]["calibrated_log_loss"]} for c in chosen],
              "final_temperature": T, "candidates": shortlist,
              "cross_fitted": metrics_from_logits(y, z_cross_final, binary, threshold),
              "best_single": {"run": shortlist[0], "log_loss": mem[shortlist[0]]["calibrated_log_loss"]},
              "selection": "fixed" if fixed else "greedy"}

    if nested:
        outer = StratifiedKFold(folds, shuffle=True, random_state=seed + 1)
        z_nested = np.empty_like(zm_cross)
        freq: Counter = Counter()
        for tr, te in outer.split(np.zeros(len(y)), y):
            sub = {n: z[tr] for n, z in cross.items()}
            sel = chosen if fixed else greedy_select(sub, y[tr], max_members, folds, seed, final_temperature)[0]
            freq.update(sel)
            zm_tr = np.mean([cross[c][tr] for c in sel], axis=0)
            zm_te = np.mean([cross[c][te] for c in sel], axis=0)
            Tn = fit_calibrator("temperature", zm_tr, y[tr]).params["T"] if final_temperature else 1.0
            z_nested[te] = zm_te / Tn
        report["nested"] = metrics_from_logits(y, z_nested, binary, threshold)
        report["selection_frequency"] = {k: v / folds for k, v in freq.most_common()}
    return report


def ensemble_markdown(rep: dict) -> str:
    cf, ne = rep["cross_fitted"], rep.get("nested")
    lines = ["# Ensemble", "", "| Member | Calibration | Single-model log loss |", "| --- | --- | --- |"]
    lines += [f"| {m['run']} | {m['calibration']} | {m['single_log_loss']:.4f} |" for m in rep["members"]]
    lines += ["", f"Final temperature: **{rep['final_temperature']:.4f}**", "",
              "| Estimate | Log loss | AUROC / macro-AUROC | Brier | ECE |", "| --- | --- | --- | --- | --- |"]

    def row(name, m):
        auc = m.get("auroc", m.get("macro_auroc"))
        return f"| {name} | {m['log_loss']:.4f} | {auc:.4f} | {m['brier']:.4f} | {m['ece']:.4f} |"

    lines.append(f"| Best single run ({rep['best_single']['run']}) | {rep['best_single']['log_loss']:.4f} | — | — | — |")
    lines.append(row("Ensemble, cross-fitted (selected on the same data)", cf))
    if ne:
        lines.append(row("Ensemble, **nested** (selection inside outer folds)", ne))
        lines += ["", "Selection frequency across outer folds: " +
                  ", ".join(f"{k} {v:.0%}" for k, v in rep["selection_frequency"].items())]
    return "\n".join(lines) + "\n"
