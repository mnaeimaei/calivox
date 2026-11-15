import json

import numpy as np
import pandas as pd
import pytest

from calivox.bundle import BundlePredictor, freeze
from calivox.cli import main
from calivox.ensemble import build_ensemble
from calivox.training import cross_validate

from .conftest import write_config


@pytest.fixture(scope="module")
def trained(phantoms, tmp_path_factory):
    """Two tiny cross-validated runs on the phantoms."""
    folder = tmp_path_factory.mktemp("work")
    cfg = write_config(folder, phantoms)
    runs = [cross_validate(cfg.for_run("simple3d", "S2")),
            cross_validate(cfg.for_run("simple3d", "S3"))]
    return cfg, runs, folder


def test_oof_covers_every_subject_once(trained):
    cfg, runs, _ = trained
    oof = runs[0].oof
    assert len(oof) == 40 and oof["id"].is_unique and oof["logit"].notna().all()
    assert set(oof["fold"]) == {0, 1, 2}
    m = json.loads((runs[0].run_dir / "metrics.json").read_text())
    assert m["inputs"] == ["roi"] and 0 < m["oof"]["log_loss"] < 5 and m["validation"] == "inner"
    assert all((runs[0].run_dir / f"fold{k}" / "model.pt").exists() for k in range(3))
    assert runs[1].metrics["inputs"] == ["whole", "roi"]


def test_heldout_validation_and_augmentation(phantoms, tmp_path):
    cfg = write_config(tmp_path, phantoms, **{"cv.validation": "heldout", "training.epochs": 1})
    res = cross_validate(cfg.for_run("simple3d", "S5"))
    assert res.metrics["augment"] and res.metrics["validation"] == "heldout" and len(res.oof) == 40


def test_multiclass_task(phantoms, tmp_path):
    df = pd.read_csv(phantoms / "manifest.csv")
    df["grade"] = pd.cut(df["severity"], [-1, 0.0, 0.45, 1.0], labels=["none", "mild", "marked"])
    df.to_csv(tmp_path / "graded.csv", index=False)
    cfg = write_config(tmp_path, phantoms, **{"task": {"type": "multiclass"}, "data.label": "grade",
                                              "data.manifest": str(tmp_path / "graded.csv"),
                                              "data.root": str(phantoms), "training.epochs": 1})
    res = cross_validate(cfg.for_run("simple3d", "S2"))
    assert {"prob_marked", "prob_mild", "prob_none"} <= set(res.oof.columns)
    assert "macro_auroc" in res.metrics["oof"]


def test_ensemble_freeze_verify_predict(trained, phantoms, tmp_path):
    cfg, runs, folder = trained
    rep = build_ensemble([r.run_dir for r in runs], max_members=2, candidates=2, folds=3, seed=0)
    assert rep["members"] and "nested" in rep and rep["final_temperature"] > 0
    bundle = freeze(rep, tmp_path / "bundle")
    assert (bundle / "assets" / "occipital_mask.nii.gz").exists()
    bp = BundlePredictor(bundle)
    assert bp.check_integrity() == []
    v = bp.verify(phantoms / "manifest.csv")
    assert v["all_equivalent"], v
    new = pd.read_csv(phantoms / "manifest.csv").drop(columns=["label"]).head(6)
    new["scan_path"] = [str(phantoms / p) for p in new["scan_path"]]
    new.to_csv(tmp_path / "new.csv", index=False)
    pred = bp.predict(tmp_path / "new.csv")
    assert len(pred) == 6 and pred["probability"].between(0, 1).all()
    assert set(pred["predicted"]) <= {"normal", "pathologic"}
    (bundle / "bundle.json").read_text()
    with open(bundle / "members" / rep["members"][0]["run"] / "fold0" / "model.pt", "ab") as f:
        f.write(b"tamper")
    assert bp.check_integrity()                       # modified files are detected


def test_ensemble_rejects_mismatched_runs(trained, tmp_path):
    _, runs, _ = trained
    bad = tmp_path / "bad"
    bad.mkdir()
    oof = pd.read_csv(runs[0].run_dir / "oof.csv").head(10)
    oof.to_csv(bad / "oof.csv", index=False)
    (bad / "metrics.json").write_text((runs[0].run_dir / "metrics.json").read_text())
    with pytest.raises(ValueError, match="different subjects"):
        build_ensemble([runs[0].run_dir, bad], folds=3)


def test_cli(trained, phantoms, tmp_path, capsys):
    cfg, runs, folder = trained
    assert main(["models"]) == 0
    assert "slice2p5d" in capsys.readouterr().out
    assert main(["synth", "--out", str(tmp_path / "ph"), "--n", "8"]) == 0
    assert main(["calibrate", str(folder / "runs"), "--write", str(tmp_path / "cal.md")]) == 0
    assert "Calibrated log loss" in (tmp_path / "cal.md").read_text()
    assert main(["report", str(folder / "runs")]) == 0
    assert main(["ensemble", str(folder / "runs"), "--out", str(tmp_path / "ens.json"), "--max-members", "2"]) == 0
    assert (tmp_path / "ens.md").exists()
    assert main(["freeze", str(runs[0].run_dir), "--out", str(tmp_path / "b1")]) == 0
    assert main(["verify", str(tmp_path / "b1"), "--manifest", str(phantoms / "manifest.csv")]) == 0
    assert main(["report", str(tmp_path / "nothing")]) == 1


def test_fixed_ensemble_members(trained, tmp_path):
    _, runs, folder = trained
    names = [r.run_dir.name for r in runs]
    rep = build_ensemble([r.run_dir for r in runs], folds=3, fixed={names[0]: "none", names[1]: "temperature"})
    assert [m["run"] for m in rep["members"]] == names and rep["selection"] == "fixed"
    assert [m["calibration"] for m in rep["members"]] == ["none", "temperature"]
    spec = tmp_path / "ens.yaml"
    spec.write_text(f"members:\n  - {{run: {names[1]}, calibration: platt}}\n")
    assert main(["ensemble", str(folder / "runs"), "--members", str(spec), "--out", str(tmp_path / "e.json")]) == 0
    with pytest.raises(ValueError, match="not among"):
        build_ensemble([r.run_dir for r in runs], folds=3, fixed={"missing_run": None})


def test_medmnist_example_converter_and_config(tmp_path):
    import importlib.util

    from calivox.config import ExperimentConfig
    from calivox.data import load_manifest

    from .conftest import ROOT

    rng = np.random.default_rng(0)
    fake = {f"{s}_images": rng.integers(0, 255, (n, 28, 28, 28)).astype(np.uint8) for s, n in
            [("train", 12), ("val", 4), ("test", 5)]}
    fake.update({f"{s}_labels": rng.integers(0, 3, (n, 1)) for s, n in [("train", 12), ("val", 4), ("test", 5)]})
    np.savez(tmp_path / "fake.npz", **fake)
    spec = importlib.util.spec_from_file_location("prep", ROOT / "examples" / "medmnist3d" / "prepare.py")
    prep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prep)
    dev, test = prep.convert(tmp_path / "fake.npz", tmp_path / "data")
    assert len(dev) == 16 and len(test) == 5 and (tmp_path / "data" / "manifest_test.csv").exists()
    cfg = ExperimentConfig.from_yaml(ROOT / "examples" / "medmnist3d" / "config.yaml")
    cfg = cfg.override(**{"data.manifest": str(tmp_path / "data" / "manifest.csv")})
    man = load_manifest(cfg)
    assert cfg.task.type == "multiclass" and len(man.classes) == dev["label"].nunique()
