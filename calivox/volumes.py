"""Reading and writing volumes (NIfTI and NumPy), and the preprocessing steps.

Volumes are handled as ``Volume(array, affine)`` with the array in (x, y, z) order and a
4x4 voxel-to-world affine, as nibabel provides them.

Preprocessing is a list of steps per input, e.g.

    - {op: reorient}                                   # to RAS (closest canonical)
    - {op: resample, spacing: [2.0, 2.0, 2.0]}         # isotropic voxels, trilinear
    - {op: normalize, method: reference_mask, mask: templates/occipital.nii.gz, stat: mean}
    - {op: crop, mode: center, shape: [96, 96, 80]}    # center crop / zero pad
    - {op: crop, mode: peak, shape: [24, 24, 20], search: [[10, 38], [16, 40], [12, 30]]}

Register your own step with ``@register_step("name")``; it receives the Volume, the
step parameters and the subject's manifest row, and returns a Volume.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

STEPS: dict[str, Callable] = {}


@dataclass
class Volume:
    array: np.ndarray            # (x, y, z) float32
    affine: np.ndarray           # 4 x 4

    @property
    def spacing(self) -> np.ndarray:
        return np.sqrt((self.affine[:3, :3] ** 2).sum(axis=0))


# ---- io ----------------------------------------------------------------------------

def load_volume(path: str | Path) -> Volume:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"image not found: {path}")
    name = path.name.lower()
    if name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        img = nib.load(str(path))
        arr = np.asanyarray(img.dataobj, dtype=np.float32)
        if arr.ndim == 4 and arr.shape[-1] == 1:
            arr = arr[..., 0]
        return Volume(arr, img.affine.astype(np.float64))
    if name.endswith(".npy"):
        return Volume(np.load(path).astype(np.float32), np.eye(4))
    if name.endswith(".npz"):
        z = np.load(path)
        return Volume(z["array"].astype(np.float32), z["affine"] if "affine" in z else np.eye(4))
    raise ValueError(f"unsupported image format: {path.name} (use .nii, .nii.gz, .npy or .npz)")


def save_volume(vol: Volume, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.name.endswith((".nii", ".nii.gz")):
        import nibabel as nib

        nib.save(nib.Nifti1Image(vol.array.astype(np.float32), vol.affine), str(path))
    else:
        np.savez_compressed(path.with_suffix(".npz") if path.suffix != ".npz" else path,
                            array=vol.array.astype(np.float32), affine=vol.affine)
    return path


# ---- step registry -----------------------------------------------------------------

def register_step(name: str):
    def deco(fn):
        STEPS[name] = fn
        return fn
    return deco


def apply_steps(vol: Volume, steps: list[dict], row: dict | None = None,
                resolve: Callable[[str], Path] | None = None) -> Volume:
    row = row or {}
    resolve = resolve or (lambda p: Path(p))
    for spec in steps:
        spec = dict(spec)
        op = spec.pop("op", None)
        if op not in STEPS:
            raise ValueError(f"unknown preprocessing op {op!r}; available: {sorted(STEPS)}")
        vol = STEPS[op](vol, spec, row, resolve)
    return vol


def _mask_for(params: dict, row: dict, resolve) -> np.ndarray:
    """A mask either from a fixed file (``mask``) or a per-subject manifest column (``mask_column``)."""
    if params.get("mask_column"):
        return load_volume(resolve(str(row[params["mask_column"]]))).array > 0.5
    if params.get("mask"):
        return load_volume(resolve(params["mask"])).array > 0.5
    raise ValueError("this step needs 'mask' (a file) or 'mask_column' (a manifest column)")


# ---- built-in steps ----------------------------------------------------------------

@register_step("reorient")
def _reorient(vol: Volume, p: dict, row: dict, resolve) -> Volume:
    """Reorient to the closest canonical orientation (RAS)."""
    import nibabel as nib

    img = nib.as_closest_canonical(nib.Nifti1Image(vol.array, vol.affine))
    return Volume(np.asanyarray(img.dataobj, dtype=np.float32), img.affine)


@register_step("resample")
def _resample(vol: Volume, p: dict, row: dict, resolve) -> Volume:
    """Resample to ``spacing`` (mm) with trilinear interpolation (``order``: 0-3)."""
    from scipy.ndimage import zoom

    target = np.asarray(p["spacing"], dtype=float)
    factors = vol.spacing / target
    arr = zoom(vol.array, factors, order=int(p.get("order", 1)), mode="nearest").astype(np.float32)
    aff = vol.affine.copy()
    aff[:3, :3] = vol.affine[:3, :3] / factors[None, :]
    return Volume(arr, aff)


@register_step("normalize")
def _normalize(vol: Volume, p: dict, row: dict, resolve) -> Volume:
    """Intensity normalisation.

    method: zscore            (x - mean) / std over foreground (> ``threshold``) or all voxels
            minmax            to [0, 1]
            percentile        divide by the ``q``-th percentile (default 99)
            reference_mask    divide by the ``stat`` (mean | median) inside a reference mask —
                              e.g. an occipital region for DaT-SPECT specific-binding ratios
    """
    a = vol.array
    method = p.get("method", "zscore")
    if method == "zscore":
        fg = a[a > p["threshold"]] if "threshold" in p else a
        out = (a - fg.mean()) / (fg.std() or 1.0)
    elif method == "minmax":
        lo, hi = float(a.min()), float(a.max())
        out = (a - lo) / ((hi - lo) or 1.0)
    elif method == "percentile":
        out = a / (np.percentile(a, p.get("q", 99.0)) or 1.0)
    elif method == "reference_mask":
        mask = _mask_for(p, row, resolve)
        if mask.shape != a.shape:
            raise ValueError(f"reference mask shape {mask.shape} != image shape {a.shape}")
        vals = a[mask]
        if vals.size == 0:
            raise ValueError("reference mask is empty for this image")
        ref = float(np.median(vals) if p.get("stat", "mean") == "median" else vals.mean())
        if ref <= 0:
            raise ValueError(f"non-positive reference value {ref}")
        out = a / ref
    else:
        raise ValueError(f"unknown normalize method {method!r}")
    return Volume(out.astype(np.float32), vol.affine)


def _crop_or_pad(a: np.ndarray, center: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros(shape, dtype=np.float32)
    src_lo = np.round(center - np.asarray(shape) / 2).astype(int)
    for_slices_src, for_slices_dst = [], []
    for ax in range(3):
        lo, hi = src_lo[ax], src_lo[ax] + shape[ax]
        s_lo, s_hi = max(lo, 0), min(hi, a.shape[ax])
        d_lo = s_lo - lo
        for_slices_src.append(slice(s_lo, max(s_hi, s_lo)))
        for_slices_dst.append(slice(d_lo, d_lo + max(s_hi - s_lo, 0)))
    out[tuple(for_slices_dst)] = a[tuple(for_slices_src)]
    return out


@register_step("crop")
def _crop(vol: Volume, p: dict, row: dict, resolve) -> Volume:
    """Crop (zero-padding where needed) to a fixed ``shape``.

    mode: center       around the array center
          box_center   around a fixed voxel ``center: [x, y, z]``
          mask_center  around the center of mass of a mask (``mask`` file or ``mask_column``)
          peak         around the intensity-weighted centroid of the brightest ``top_fraction``
                       of voxels inside a ``search`` box [[x0, x1], [y0, y1], [z0, z1]]
    """
    shape = tuple(int(s) for s in p["shape"])
    a = vol.array
    mode = p.get("mode", "center")
    if mode == "center":
        c = (np.asarray(a.shape) - 1) / 2
    elif mode == "box_center":
        c = np.asarray(p["center"], dtype=float)
    elif mode == "mask_center":
        m = _mask_for(p, row, resolve)
        idx = np.argwhere(m)
        if not len(idx):
            raise ValueError("mask_center: empty mask")
        c = idx.mean(axis=0)
    elif mode == "peak":
        box = p.get("search") or [[0, s] for s in a.shape]
        sl = tuple(slice(int(lo), int(hi)) for lo, hi in box)
        sub = a[sl]
        thr = np.quantile(sub, 1 - float(p.get("top_fraction", 0.02)))
        w = np.where(sub >= thr, sub, 0.0)
        idx = np.indices(sub.shape).reshape(3, -1)
        c = (idx * w.reshape(-1)).sum(axis=1) / (w.sum() or 1.0) + np.array([s.start for s in sl])
    else:
        raise ValueError(f"unknown crop mode {mode!r}")
    lo = np.round(c - np.asarray(shape) / 2)
    aff = vol.affine.copy()
    aff[:3, 3] = vol.affine[:3, :3] @ lo + vol.affine[:3, 3]
    return Volume(_crop_or_pad(a, c, shape), aff)


@register_step("clip")
def _clip(vol: Volume, p: dict, row: dict, resolve) -> Volume:
    """Clip intensities to [``min``, ``max``] or to percentiles ``[q_low, q_high]``."""
    a = vol.array
    lo = p["min"] if "min" in p else (np.percentile(a, p["q_low"]) if "q_low" in p else None)
    hi = p["max"] if "max" in p else (np.percentile(a, p["q_high"]) if "q_high" in p else None)
    return Volume(np.clip(a, lo, hi).astype(np.float32), vol.affine)
