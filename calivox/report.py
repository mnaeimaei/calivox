"""Leaderboards over run folders."""
from __future__ import annotations

import json
from pathlib import Path


def find_runs(paths: list[str | Path]) -> list[Path]:
    """Run folders among the given paths (a run folder, or a folder of run folders)."""
    out = []
    for p in map(Path, paths):
        if (p / "oof.csv").exists():
            out.append(p)
        elif p.is_dir():
            out += sorted(d for d in p.iterdir() if (d / "oof.csv").exists())
    if not out:
        raise FileNotFoundError(f"no run folders (with oof.csv) in {', '.join(map(str, paths))}")
    return out


def collect(runs: list[Path]) -> list[dict]:
    rows = []
    for d in runs:
        m = json.loads((d / "metrics.json").read_text())
        o = m["oof"]
        row = {"run": d.name, "model": m["model"], "pretrained": m.get("pretrained") or "—",
               "inputs": "+".join(m["inputs"]), "augment": "yes" if m["augment"] else "no",
               "log_loss": o["log_loss"], "auroc": o.get("auroc", o.get("macro_auroc")),
               "auprc": o.get("auprc"), "brier": o["brier"], "ece": o["ece"],
               "balanced_accuracy": o["balanced_accuracy"], "seconds": m["seconds"]}
        cal = d / "calibration.json"
        if cal.exists():
            c = json.loads(cal.read_text())
            row["calibration"] = c["selected"]
            row["calibrated_log_loss"] = c["methods"][c["selected"]]["log_loss"]
            row["calibrated_ece"] = c["methods"][c["selected"]]["ece"]
        rows.append(row)
    key = "calibrated_log_loss" if all("calibrated_log_loss" in r for r in rows) else "log_loss"
    return sorted(rows, key=lambda r: r[key])


def leaderboard_markdown(rows: list[dict], title: str = "Leaderboard") -> str:
    cal = all("calibrated_log_loss" in r for r in rows)
    head = ["Run", "Model", "Inputs", "Aug.", "Log loss", "AUROC", "AUPRC", "Brier", "ECE", "Bal. acc."]
    if cal:
        head += ["Calibration", "Calibrated log loss", "Calibrated ECE"]
    head.append("Train (s)")
    f = lambda v, d=4: "—" if v is None else f"{v:.{d}f}"  # noqa: E731
    lines = [f"# {title}\n", "Out-of-fold predictions; calibration is cross-fitted. Sorted by "
             + ("calibrated " if cal else "") + "log loss (lower is better).\n",
             "| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    for r in rows:
        cells = [r["run"], r["model"] + (f" ({r['pretrained']})" if r["pretrained"] != "—" else ""), r["inputs"],
                 r["augment"], f(r["log_loss"]), f(r["auroc"]), f(r["auprc"]), f(r["brier"]), f(r["ece"]),
                 f(r["balanced_accuracy"], 3)]
        if cal:
            cells += [r["calibration"], f(r["calibrated_log_loss"]), f(r["calibrated_ece"])]
        cells.append(f"{r['seconds']:.0f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
