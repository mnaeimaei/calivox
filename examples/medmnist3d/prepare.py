"""Convert a MedMNIST3D .npz (e.g. organmnist3d.npz, 28x28x28 or 64x64x64) into a CaliVox
manifest + one .npy volume per sample.

    pip install medmnist && python -c "import medmnist; medmnist.OrganMNIST3D(split='train', download=True, root='.')"
    python prepare.py organmnist3d.npz --out data

The official train/val/test splits are kept in a `split` column; train + val are merged for
cross-validation and the test split goes to a separate manifest (manifest_test.csv).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def convert(npz_path: str | Path, out: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = Path(out)
    (out / "volumes").mkdir(parents=True, exist_ok=True)
    data = np.load(npz_path)
    rows = []
    for split in ("train", "val", "test"):
        images, labels = data[f"{split}_images"], data[f"{split}_labels"].reshape(-1)
        for i, (img, lab) in enumerate(zip(images, labels)):
            sid = f"{split}-{i:05d}"
            path = Path("volumes") / f"{sid}.npy"
            np.save(out / path, img.astype(np.float32))
            rows.append({"id": sid, "label": int(lab), "split": split, "path": str(path)})
    df = pd.DataFrame(rows)
    dev, test = df[df.split != "test"], df[df.split == "test"]
    dev.to_csv(out / "manifest.csv", index=False)
    test.drop(columns=["label"]).to_csv(out / "manifest_test.csv", index=False)
    test[["id", "label"]].to_csv(out / "labels_test.csv", index=False)
    return dev, test


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("npz")
    p.add_argument("--out", default="data")
    a = p.parse_args()
    dev, test = convert(a.npz, a.out)
    print(f"{len(dev)} train+val, {len(test)} test, {dev['label'].nunique()} classes -> {a.out}")
