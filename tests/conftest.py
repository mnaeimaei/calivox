from pathlib import Path

import pytest
import yaml

from calivox.config import ExperimentConfig
from calivox.synthetic import make_dataset

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def phantoms(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("phantoms")
    make_dataset(d, n=40, seed=3)
    return d


def write_config(folder: Path, data_dir: Path, **over) -> ExperimentConfig:
    raw = {
        "name": "tiny",
        "task": {"type": "binary", "positive": "pathologic"},
        "data": {"manifest": str(data_dir / "manifest.csv"), "id": "subject_id", "label": "label",
                 "group": "site", "inputs": {"whole": "scan_path", "roi": "scan_path"}},
        "preprocess": {
            "whole": [{"op": "normalize", "method": "reference_mask", "mask": str(data_dir / "occipital_mask.nii.gz")},
                      {"op": "resample", "spacing": [8.0, 8.0, 8.0]},
                      {"op": "crop", "mode": "center", "shape": [24, 24, 20]}],
            "roi": [{"op": "normalize", "method": "reference_mask", "mask": str(data_dir / "occipital_mask.nii.gz")},
                    {"op": "crop", "mode": "peak", "shape": [24, 16, 12],
                     "search": [[8, 40], [14, 40], [10, 30]], "top_fraction": 0.01}]},
        "scenarios": {"S2": {"inputs": ["roi"]}, "S3": {"inputs": ["whole", "roi"]},
                      "S5": {"inputs": ["roi"], "augment": True}},
        "model": {"name": "simple3d", "params": {"width": 4}},
        "models": {"simple3d": {"name": "simple3d", "params": {"width": 4}},
                   "slice2p5d": {"name": "slice2p5d", "params": {"slices": 4}}},
        "cv": {"folds": 3, "seed": 1, "validation": "inner", "inner_fraction": 0.2},
        "training": {"epochs": 2, "batch_size": 8, "learning_rate": 1e-3, "num_threads": 1,
                     "cache_in_memory": True, "early_stopping_patience": 2, "warmup_epochs": 1},
        "calibration": {"folds": 3},
        "output_dir": str(folder / "runs"),
        "cache_dir": str(folder / "cache"),
    }
    for key, value in over.items():
        node = raw
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    path = folder / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return ExperimentConfig.from_yaml(path)
