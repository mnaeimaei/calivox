"""Frozen runtime bundle: everything needed to score new scans, and a check that it
reproduces the training-time predictions.

    bundle/
      bundle.json          task, classes, members, calibrators, final temperature, file hashes
      members/<run>/       config.yaml, folds.csv, oof.csv, fold<k>/model.pt
      assets/              files referenced by preprocessing (e.g. reference masks)

Inference per member: preprocess exactly as in training -> every fold model -> mean of
fold logits -> member calibrator; then the equal-weight logit mean of the members and the
final temperature.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .calibration import Calibrator
from .config import ExperimentConfig
from .data import VolumeDataset, collate, load_manifest, preprocess_inputs
from .metrics import sigmoid, softmax


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _bundle_config(cfg_path: Path, assets: Path) -> str:
    """Copy files referenced by preprocessing steps into assets/ and point the config at them."""
    cfg = ExperimentConfig.from_yaml(cfg_path)
    data = cfg.model_dump(mode="json")
    for steps in data["preprocess"].values():
        for step in steps:
            if step.get("mask"):
                src = cfg.resolve(step["mask"])
                dst = assets / src.name
                assets.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                step["mask"] = str(Path("..") / ".." / "assets" / src.name)
    plugins = []
    for plugin in data.get("plugins", []):
        if plugin.endswith(".py"):
            src = cfg.resolve(plugin)
            assets.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, assets / src.name)
            plugin = str(Path("..") / ".." / "assets" / src.name)
        plugins.append(plugin)
    data["plugins"] = plugins
    data.pop("base_dir", None)
    data["data"]["root"] = None          # new scans come with their own manifest
    import yaml

    return yaml.safe_dump(data, sort_keys=False)


def freeze(ensemble: dict | str | Path, out_dir: str | Path) -> Path:
    """Freeze an ensemble report (from ``build_ensemble``) or a single run folder."""
    out = Path(out_dir)
    if isinstance(ensemble, (str, Path)) and (Path(ensemble) / "oof.csv").exists():
        run = Path(ensemble)
        cal = json.loads((run / "calibration.json").read_text()) if (run / "calibration.json").exists() else None
        ensemble = {"members": [{"run": run.name, "dir": str(run),
                                 "calibrator": cal["calibrator"] if cal else {"method": "none", "params": {}}}],
                    "final_temperature": 1.0}
    elif isinstance(ensemble, (str, Path)):
        ensemble = json.loads(Path(ensemble).read_text())
    if out.exists():
        shutil.rmtree(out)
    members, hashes = [], {}
    for m in ensemble["members"]:
        src, dst = Path(m["dir"]), out / "members" / m["run"]
        dst.mkdir(parents=True)
        (dst / "config.yaml").write_text(_bundle_config(src / "config.yaml", out / "assets"))
        for f in ("folds.csv", "oof.csv"):
            shutil.copy2(src / f, dst / f)
        for fold in sorted(src.glob("fold*/model.pt")):
            (dst / fold.parent.name).mkdir()
            shutil.copy2(fold, dst / fold.parent.name / "model.pt")
        members.append({"run": m["run"], "calibrator": m["calibrator"]})
    for f in sorted(out.rglob("*")):
        if f.is_file():
            hashes[str(f.relative_to(out))] = _sha256(f)
    first = ExperimentConfig.from_yaml(out / "members" / members[0]["run"] / "config.yaml")
    meta = {"calivox_version": __version__, "task": first.task.model_dump(),
            "classes": json.loads((Path(ensemble["members"][0]["dir"]) / "metrics.json").read_text())["classes"],
            "members": members, "final_temperature": float(ensemble["final_temperature"]), "files": hashes}
    (out / "bundle.json").write_text(json.dumps(meta, indent=2))
    return out


class BundlePredictor:
    def __init__(self, bundle_dir: str | Path, device: str = "cpu") -> None:
        import torch

        self.dir = Path(bundle_dir)
        self.meta = json.loads((self.dir / "bundle.json").read_text())
        self.binary = self.meta["task"]["type"] == "binary"
        self.device = torch.device(device)
        self.members = []
        for m in self.meta["members"]:
            mdir = self.dir / "members" / m["run"]
            cfg = ExperimentConfig.from_yaml(mdir / "config.yaml")
            self.members.append({"run": m["run"], "dir": mdir, "cfg": cfg,
                                 "calibrator": Calibrator.from_dict(m["calibrator"]),
                                 "folds": sorted(mdir.glob("fold*/model.pt"))})

    def check_integrity(self) -> list[str]:
        """Files whose SHA-256 differs from bundle.json (empty list = intact)."""
        return [f for f, h in self.meta["files"].items() if _sha256(self.dir / f) != h]

    def _member_fold_logits(self, member: dict, manifest_path: Path, cache_dir: Path) -> tuple[np.ndarray, np.ndarray]:
        """(n_folds, N[, K]) raw logits of every fold model, and subject ids."""
        import torch
        from torch.utils.data import DataLoader

        from .training import build_for, predict_logits

        cfg = member["cfg"]
        man = load_manifest(cfg, manifest_path, require_labels=False)
        man.classes = self.meta["classes"]
        paths = preprocess_inputs(cfg, man, cfg.scenario.inputs, cache_dir)
        sel = {i: paths[i] for i in cfg.scenario.inputs}
        dl = DataLoader(VolumeDataset(sel, None, np.arange(len(man.ids)), cfg.data.depth_axis),
                        batch_size=8, shuffle=False, collate_fn=collate)
        out = []
        for ckpt in member["folds"]:
            model = build_for(cfg.override(**{"model.pretrained": None}), man).to(self.device)
            model.load_state_dict(torch.load(ckpt, map_location=self.device, weights_only=True))
            out.append(predict_logits(model, dl, self.device))
        return np.stack(out), man.ids

    def predict(self, manifest_path: str | Path, cache_dir: str | Path | None = None) -> pd.DataFrame:
        cache = Path(cache_dir) if cache_dir else self.dir / ".cache"
        member_z, ids = [], None
        for m in self.members:
            fold_z, ids = self._member_fold_logits(m, Path(manifest_path), cache)
            member_z.append(m["calibrator"].apply(fold_z.mean(axis=0)))
        z = np.mean(member_z, axis=0) / self.meta["final_temperature"]
        frame = pd.DataFrame({"id": ids})
        classes = self.meta["classes"]
        if self.binary:
            pos = classes[-1] if self.meta["task"].get("positive") is None else self.meta["task"]["positive"]
            frame["probability"] = sigmoid(z)
            frame["predicted"] = np.where(frame["probability"] >= self.meta["task"]["threshold"], pos,
                                          [c for c in classes if str(c) != str(pos)][0])
            for m, zm in zip(self.members, member_z):
                frame[f"p_{m['run']}"] = sigmoid(zm)
        else:
            P = softmax(z)
            for j, c in enumerate(classes):
                frame[f"prob_{c}"] = P[:, j]
            frame["predicted"] = [classes[i] for i in P.argmax(axis=1)]
        return frame

    def verify(self, manifest_path: str | Path, tol: float = 1e-4, cache_dir: str | Path | None = None) -> dict:
        """Runtime equivalence: each fold model, run from the bundle, must reproduce the
        training-time OOF logits of its own held-out subjects."""
        cache = Path(cache_dir) if cache_dir else self.dir / ".cache"
        report = {}
        for m in self.members:
            fold_z, ids = self._member_fold_logits(m, Path(manifest_path), cache)
            folds = pd.read_csv(m["dir"] / "folds.csv").astype({"id": str}).set_index("id")["fold"]
            oof = pd.read_csv(m["dir"] / "oof.csv").astype({"id": str}).set_index("id")
            cols = ["logit"] if "logit" in oof else [c for c in oof.columns if c.startswith("logit_")]
            diffs = []
            for i, sid in enumerate(ids):
                if sid not in folds.index:
                    continue
                k = int(folds[sid])
                got = np.atleast_1d(fold_z[k][i])
                diffs.append(np.max(np.abs(got - oof.loc[sid, cols].to_numpy(float))))
            if not diffs:
                raise ValueError(f"{m['run']}: no training subjects found in {manifest_path}")
            report[m["run"]] = {"subjects": len(diffs), "max_abs_logit_diff": float(np.max(diffs)),
                                "equivalent": bool(np.max(diffs) <= tol)}
        report["all_equivalent"] = all(v["equivalent"] for v in report.values() if isinstance(v, dict))
        return report
