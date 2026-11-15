"""Command-line interface.

    calivox synth      --out data/phantoms --n 400           synthetic DaT-like phantoms
    calivox preprocess CONFIG                                 build and cache model inputs
    calivox cv         CONFIG [--model m --scenario S5]      one cross-validated run (OOF)
    calivox benchmark  CONFIG --models a,b --scenarios S1,S2  every model x scenario
    calivox calibrate  RUNS...                                cross-fitted calibration per run
    calivox ensemble   RUNS... --out ensemble.json            calibrated ensemble + nested estimate
                       [--members spec.yaml]                  ... or a fixed member list
    calivox freeze     ENSEMBLE.json|RUN --out bundle/        frozen runtime bundle
    calivox predict    BUNDLE --manifest new.csv              probabilities for new scans
    calivox verify     BUNDLE --manifest train.csv            bundle reproduces training OOF?
    calivox report     RUNS...                                leaderboard
    calivox models                                            available backbones
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cfg(path, model=None, scenario=None, epochs=None):
    from .config import ExperimentConfig

    cfg = ExperimentConfig.from_yaml(path)
    return cfg.for_run(model, scenario, epochs) if (model or scenario or epochs) else cfg


def cmd_synth(a):
    from .synthetic import make_dataset

    df = make_dataset(a.out, a.n, a.seed)
    print(f"{len(df)} phantoms -> {a.out} ({df['label'].value_counts().to_dict()}, sites {df['site'].nunique()})")


def cmd_preprocess(a):
    from .data import load_manifest, preprocess_inputs

    cfg = _cfg(a.config)
    preprocess_inputs(cfg, load_manifest(cfg), verbose=True)


def cmd_cv(a):
    from .training import cross_validate

    cfg = _cfg(a.config, a.model, a.scenario, a.epochs)
    out = a.out
    if out is None and a.model and a.scenario:      # same folder name as `benchmark`
        out = cfg.resolve(cfg.output_dir) / f"{a.model}__{a.scenario}"
    print(f"{cfg.run_name}: {cfg.cv.folds}-fold CV ...", flush=True)
    res = cross_validate(cfg, out, verbose=a.verbose)
    o = res.metrics["oof"]
    print(f"OOF log loss {o['log_loss']:.4f}  AUROC {o.get('auroc', o.get('macro_auroc'))}  -> {res.run_dir}")


def cmd_benchmark(a):
    from .config import ExperimentConfig
    from .data import load_manifest, preprocess_inputs
    from .report import collect, leaderboard_markdown
    from .training import cross_validate

    base = ExperimentConfig.from_yaml(a.config)
    models = a.models.split(",") if a.models else list(base.models) or [base.model.name]
    scenarios = a.scenarios.split(",") if a.scenarios else list(base.scenarios) or [None]
    man = load_manifest(base)
    paths = preprocess_inputs(base, man, verbose=True)
    runs = []
    for m in models:
        for s in scenarios:
            cfg = base.for_run(m, s, a.epochs)
            run_dir = cfg.resolve(cfg.output_dir) / (f"{m}__{s}" if s else cfg.run_name)
            if a.skip_existing and (run_dir / "metrics.json").exists():
                print(f"skip {run_dir.name} (done)")
            else:
                print(f"== {run_dir.name}", flush=True)
                cross_validate(cfg, run_dir, verbose=a.verbose, manifest=man, paths=paths)
            runs.append(run_dir)
    md = leaderboard_markdown(collect(runs), a.title or f"{base.name} — benchmark")
    print(md)
    if a.write:
        Path(a.write).write_text(md)


def cmd_calibrate(a):
    from .calibration import evaluate_calibration
    from .config import ExperimentConfig
    from .report import collect, find_runs, leaderboard_markdown
    from .training import load_oof

    runs = find_runs(a.runs)
    for d in runs:
        cfg = ExperimentConfig.from_yaml(d / "config.yaml")
        z, y, _ = load_oof(d)
        rep = evaluate_calibration(z, y, list(cfg.calibration.methods), cfg.calibration.folds,
                                   cfg.calibration.seed, cfg.task.threshold)
        (d / "calibration.json").write_text(json.dumps(rep, indent=2))
        print(f"{d.name:45s} " + "  ".join(f"{m} {v['log_loss']:.4f}" for m, v in rep["methods"].items())
              + f"  -> {rep['selected']}")
    md = leaderboard_markdown(collect(runs), a.title or "Calibrated leaderboard")
    if a.write:
        Path(a.write).write_text(md)


def cmd_ensemble(a):
    from .config import ExperimentConfig
    from .ensemble import build_ensemble, ensemble_markdown
    from .report import find_runs

    runs = find_runs(a.runs)
    cfg = ExperimentConfig.from_yaml(runs[0] / "config.yaml")
    fixed = None
    if a.members:
        import yaml

        spec = yaml.safe_load(Path(a.members).read_text())
        fixed = {m["run"]: m.get("calibration") for m in spec["members"]}
    rep = build_ensemble(runs, a.max_members or cfg.ensemble.max_members, a.candidates or cfg.ensemble.candidates,
                         tuple(cfg.calibration.methods), cfg.calibration.folds, cfg.calibration.seed,
                         cfg.ensemble.final_temperature, not a.no_nested, cfg.task.threshold, fixed)
    Path(a.out).write_text(json.dumps(rep, indent=2, default=str))
    md = ensemble_markdown(rep)
    Path(a.out).with_suffix(".md").write_text(md)
    print(md)


def cmd_freeze(a):
    from .bundle import freeze

    out = freeze(a.source, a.out)
    print(f"bundle -> {out}")


def cmd_predict(a):
    from .bundle import BundlePredictor

    bp = BundlePredictor(a.bundle)
    bad = bp.check_integrity()
    if bad:
        sys.exit(f"error: bundle files changed since freezing: {bad[:5]}")
    df = bp.predict(a.manifest, a.cache)
    df.to_csv(a.out, index=False)
    print(f"{len(df)} predictions -> {a.out}")


def cmd_verify(a):
    from .bundle import BundlePredictor

    rep = BundlePredictor(a.bundle).verify(a.manifest, a.tol, a.cache)
    print(json.dumps(rep, indent=2))
    if not rep["all_equivalent"]:
        sys.exit(1)


def cmd_report(a):
    from .report import collect, find_runs, leaderboard_markdown

    md = leaderboard_markdown(collect(find_runs(a.runs)), a.title or "Leaderboard")
    print(md)
    if a.write:
        Path(a.write).write_text(md)


def cmd_models(_a):
    from .models import BACKBONES

    for name in sorted(BACKBONES):
        doc = (BACKBONES[name].__doc__ or "").strip().splitlines()
        print(f"  {name:12s} {doc[0] if doc else ''}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="calivox", description="Calibrated 3D image classification")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("synth", help="write synthetic DaT-like phantoms")
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=400)
    s.add_argument("--seed", type=int, default=42)
    s.set_defaults(func=cmd_synth)

    s = sub.add_parser("preprocess", help="build and cache model inputs")
    s.add_argument("config")
    s.set_defaults(func=cmd_preprocess)

    for name, fn, hlp in [("cv", cmd_cv, "one cross-validated run"), ("benchmark", cmd_benchmark, "model x scenario grid")]:
        s = sub.add_parser(name, help=hlp)
        s.add_argument("config")
        s.add_argument("--epochs", type=int)
        s.add_argument("-v", "--verbose", action="store_true")
        if name == "cv":
            s.add_argument("--model")
            s.add_argument("--scenario")
            s.add_argument("--out")
        else:
            s.add_argument("--models")
            s.add_argument("--scenarios")
            s.add_argument("--skip-existing", action="store_true")
            s.add_argument("--write")
            s.add_argument("--title")
        s.set_defaults(func=fn)

    s = sub.add_parser("calibrate", help="cross-fitted calibration of run OOF predictions")
    s.add_argument("runs", nargs="+")
    s.add_argument("--write")
    s.add_argument("--title")
    s.set_defaults(func=cmd_calibrate)

    s = sub.add_parser("ensemble", help="select a calibrated ensemble")
    s.add_argument("runs", nargs="+")
    s.add_argument("--out", default="ensemble.json")
    s.add_argument("--max-members", type=int)
    s.add_argument("--candidates", type=int)
    s.add_argument("--no-nested", action="store_true")
    s.add_argument("--members", help="YAML with a fixed member list: members: [{run: NAME, calibration: temperature}]")
    s.set_defaults(func=cmd_ensemble)

    s = sub.add_parser("freeze", help="freeze an ensemble or a run into a runtime bundle")
    s.add_argument("source", help="ensemble.json or a run folder")
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_freeze)

    for name, fn in [("predict", cmd_predict), ("verify", cmd_verify)]:
        s = sub.add_parser(name)
        s.add_argument("bundle")
        s.add_argument("--manifest", required=True)
        s.add_argument("--cache")
        if name == "predict":
            s.add_argument("--out", default="predictions.csv")
        else:
            s.add_argument("--tol", type=float, default=1e-4)
        s.set_defaults(func=fn)

    s = sub.add_parser("report", help="leaderboard of run folders")
    s.add_argument("runs", nargs="+")
    s.add_argument("--write")
    s.add_argument("--title")
    s.set_defaults(func=cmd_report)

    s = sub.add_parser("models", help="list backbones")
    s.set_defaults(func=cmd_models)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
