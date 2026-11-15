"""Synthetic DaT-SPECT-like phantoms, for offline experiments, tests and CI.

Each subject is a 3D volume of a head with low background uptake and two bilateral
striatal structures per hemisphere (a rounder caudate, an elongated putamen) with high
uptake. Pathologic subjects lose putaminal uptake — strongest posteriorly, more on one
side — and a little caudate uptake, with a severity that ranges from subtle to marked.

Realistic nuisance factors:
  * acquisition sites with different resolution (blur), global count level and noise
  * small head-position shifts between subjects
  * a Poisson-like noise model

Files written:
  scans/<id>.nii.gz           the volumes (4 mm voxels)
  occipital_mask.nii.gz       a fixed reference region (posterior background)
  manifest.csv                subject_id, label (normal | pathologic), site, severity, scan_path

These phantoms only imitate the *structure* of the problem; they are not a model of
real DaT-SPECT physics or of any patient data.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

SHAPE = (48, 48, 40)          # x (right), y (anterior), z (superior)
VOXEL_MM = 4.0
SITES = {"site_A": (1.0, 1.00, 0.10), "site_B": (1.5, 0.75, 0.14), "site_C": (2.0, 1.30, 0.12)}
#          name     (blur sigma in voxels, count scale, relative noise)


def _ellipsoid(grid, center, radii) -> np.ndarray:
    x, y, z = grid
    return (((x - center[0]) / radii[0]) ** 2 + ((y - center[1]) / radii[1]) ** 2
            + ((z - center[2]) / radii[2]) ** 2) <= 1.0


def occipital_mask() -> np.ndarray:
    grid = np.indices(SHAPE).astype(float)
    head = _ellipsoid(grid, (23.5, 23.5, 19.5), (19, 21, 16))
    return head & (grid[1] < 12) & (grid[2] > 12) & (grid[2] < 28)


def phantom(rng: np.random.Generator, pathologic: bool, site: str) -> tuple[np.ndarray, float]:
    blur, scale, noise = SITES[site]
    grid = np.indices(SHAPE).astype(float)
    shift = rng.normal(0, 1.2, 3)
    c = np.array([23.5, 23.5, 19.5]) + shift
    vol = np.zeros(SHAPE, dtype=float)
    head = _ellipsoid(grid, c, (19, 21, 16) * rng.uniform(0.95, 1.05, 3))
    vol[head] = 1.0

    severity = float(rng.uniform(0.15, 0.75)) if pathologic else 0.0
    worse = rng.integers(0, 2)               # which hemisphere is more affected
    for side in (0, 1):
        sx = -1 if side == 0 else 1
        caud_c = c + np.array([sx * 4.5, 6.0, 1.0])
        put_c = c + np.array([sx * 8.5, 1.0, -0.5])
        caud = _ellipsoid(grid, caud_c, (2.6, 3.2, 3.2))
        put = _ellipsoid(grid, put_c, (2.4, 5.5, 3.0))
        base = rng.uniform(3.4, 4.2)
        loss = severity * (1.0 if side == worse else rng.uniform(0.4, 0.8))
        # posterior-to-anterior gradient of loss along the putamen (y axis)
        post = np.clip((put_c[1] + 5.5 - grid[1]) / 11.0, 0, 1)
        vol[put] = 1.0 + (base - 1.0) * (1 - loss * (0.5 + 0.5 * post[put]))
        vol[caud] = 1.0 + (base - 1.0) * (1 - 0.35 * loss)
    vol = gaussian_filter(vol, blur)
    counts = vol * scale * 100.0
    noisy = counts + rng.normal(0, 1, SHAPE) * np.sqrt(np.maximum(counts, 1.0)) * noise * 10
    return np.clip(noisy, 0, None).astype(np.float32), severity


def make_dataset(out_dir: str | Path, n: int = 400, seed: int = 42, pathologic_fraction: float = 0.55) -> pd.DataFrame:
    import nibabel as nib

    out = Path(out_dir)
    (out / "scans").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    affine = np.diag([VOXEL_MM, VOXEL_MM, VOXEL_MM, 1.0])
    affine[:3, 3] = -np.array(SHAPE) * VOXEL_MM / 2
    nib.save(nib.Nifti1Image(occipital_mask().astype(np.uint8), affine), str(out / "occipital_mask.nii.gz"))
    rows = []
    sites = list(SITES)
    for i in range(n):
        sid = f"sub-{i + 1:04d}"
        patho = bool(rng.random() < pathologic_fraction)
        site = sites[int(rng.integers(0, len(sites)))]
        vol, sev = phantom(rng, patho, site)
        path = Path("scans") / f"{sid}.nii.gz"
        nib.save(nib.Nifti1Image(vol, affine), str(out / path))
        rows.append({"subject_id": sid, "label": "pathologic" if patho else "normal", "site": site,
                     "severity": round(sev, 3), "scan_path": str(path)})
    df = pd.DataFrame(rows)
    df.to_csv(out / "manifest.csv", index=False)
    return df
