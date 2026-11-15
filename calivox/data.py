"""Manifest, cached preprocessing, the PyTorch dataset, and cross-validation folds.

The manifest is a CSV with one row per subject:

    subject_id, label, [site], whole_path, roi_path, [mask columns ...]

Image paths are relative to ``data.root`` (default: the manifest's folder).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .volumes import apply_steps, load_volume

FOLD = "__fold__"


# ---- manifest ----------------------------------------------------------------------

@dataclass
class Manifest:
    df: pd.DataFrame
    root: Path
    classes: list                      # sorted class labels
    positive_index: int                # binary: column of the positive class

    @property
    def y(self) -> np.ndarray:
        lookup = {c: i for i, c in enumerate(self.classes)}
        return np.array([lookup[v] for v in self.df["__label__"]], dtype=np.int64)

    @property
    def ids(self) -> np.ndarray:
        return self.df["__id__"].astype(str).to_numpy()


def load_manifest(cfg: ExperimentConfig, path: str | Path | None = None, require_labels: bool = True) -> Manifest:
    d = cfg.data
    mpath = Path(path) if path else cfg.resolve(d.manifest)
    if not mpath.exists():
        raise FileNotFoundError(f"manifest not found: {mpath}")
    df = pd.read_csv(mpath)
    need = [d.id, *d.inputs.values()] + ([d.label] if require_labels else [])
    missing = [c for c in need if c not in df]
    if missing:
        raise ValueError(f"manifest {mpath.name} is missing columns: {missing}")
    if df[d.id].duplicated().any():
        raise ValueError(f"duplicate subject ids in {mpath.name}: "
                         f"{df.loc[df[d.id].duplicated(), d.id].head().tolist()}")
    df = df.copy()
    df["__id__"] = df[d.id].astype(str)
    df["__label__"] = df[d.label] if d.label in df else np.nan
    root = cfg.resolve(d.root) if d.root else mpath.parent
    if require_labels:
        classes = sorted(pd.unique(df["__label__"]).tolist())
        if cfg.task.type == "binary" and len(classes) != 2:
            raise ValueError(f"binary task needs exactly 2 classes, found {classes}")
        if len(classes) < 2:
            raise ValueError("need at least 2 classes")
    else:
        classes = []
    pos = len(classes) - 1
    if cfg.task.positive is not None and classes:
        matches = [i for i, c in enumerate(classes) if str(c) == str(cfg.task.positive)]
        if not matches:
            raise ValueError(f"positive class {cfg.task.positive!r} not in {classes}")
        pos = matches[0]
    return Manifest(df.reset_index(drop=True), root, classes, pos)


# ---- preprocessing with a cache ----------------------------------------------------

def _steps_key(steps: list[dict]) -> str:
    return hashlib.sha1(json.dumps(steps, sort_keys=True, default=str).encode()).hexdigest()[:10]


def resolve_image(man: Manifest, value: str) -> Path:
    p = Path(str(value))
    return p if p.is_absolute() else man.root / p


def preprocess_inputs(cfg: ExperimentConfig, man: Manifest, inputs: list[str] | None = None,
                      cache_dir: str | Path | None = None, verbose: bool = False) -> dict[str, list[Path]]:
    """Return, per input, the path of the model-ready array for every subject.

    Inputs without preprocessing steps are read as they are. Inputs with steps are
    processed once and cached as ``<cache>/<input>-<hash>/<id>.npy``; the hash covers
    the step list, so changing a step rebuilds only that input.
    """
    inputs = inputs or list(cfg.data.inputs)
    cache = Path(cache_dir) if cache_dir else cfg.resolve(cfg.cache_dir)
    out: dict[str, list[Path]] = {}
    for name in inputs:
        col, steps = cfg.data.inputs[name], cfg.preprocess.get(name, [])
        paths = []
        folder = cache / f"{name}-{_steps_key(steps)}" if steps else None
        for i, row in man.df.iterrows():
            src = resolve_image(man, row[col])
            if not steps:
                paths.append(src)
                continue
            dst = folder / f"{row['__id__']}.npy"
            if not dst.exists():
                vol = apply_steps(load_volume(src), steps, row.to_dict(), _resolver(cfg, man))
                dst.parent.mkdir(parents=True, exist_ok=True)
                np.save(dst, vol.array.astype(np.float32))
            paths.append(dst)
        out[name] = paths
        if verbose:
            print(f"  {name}: {len(paths)} images ({'cached' if steps else 'as is'})")
    return out


def _resolver(cfg: ExperimentConfig, man: Manifest):
    """Paths in steps (mask files, mask columns): absolute, else relative to the config
    folder if the file exists there, else relative to the manifest root."""
    def resolve(p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        in_cfg = cfg.resolve(p)
        return in_cfg if in_cfg.exists() else man.root / path
    return resolve


def to_dhw(arr: np.ndarray, depth_axis: str) -> np.ndarray:
    """(x, y, z) array -> (D, H, W) with the chosen axis as depth."""
    order = {"z": (2, 1, 0), "y": (1, 2, 0), "x": (0, 1, 2)}[depth_axis]
    return np.ascontiguousarray(np.transpose(arr, order))


def load_array(path: Path, depth_axis: str) -> np.ndarray:
    arr = np.load(path) if path.suffix == ".npy" else load_volume(path).array
    if arr.ndim != 3:
        raise ValueError(f"{path.name}: expected a 3D volume, got shape {arr.shape}")
    return to_dhw(arr.astype(np.float32), depth_axis)


# ---- torch dataset -----------------------------------------------------------------

class VolumeDataset:
    """Returns ({input: (1, D, H, W) tensor}, label index) per subject."""

    def __init__(self, paths: dict[str, list[Path]], y: np.ndarray | None, indices: np.ndarray,
                 depth_axis: str = "z", augment=None, cache_in_memory: bool = False) -> None:
        self.paths, self.y, self.indices = paths, y, np.asarray(indices)
        self.depth_axis, self.augment = depth_axis, augment
        self._cache: dict = {} if cache_in_memory else None

    def __len__(self) -> int:
        return len(self.indices)

    def _get(self, name: str, i: int) -> np.ndarray:
        if self._cache is not None and (name, i) in self._cache:
            return self._cache[(name, i)]
        arr = load_array(self.paths[name][i], self.depth_axis)
        if self._cache is not None:
            self._cache[(name, i)] = arr
        return arr

    def __getitem__(self, k: int):
        import torch

        i = int(self.indices[k])
        x = {}
        for name in self.paths:
            t = torch.from_numpy(self._get(name, i)).unsqueeze(0)
            x[name] = self.augment(t) if self.augment is not None else t
        y = int(self.y[i]) if self.y is not None else -1
        return x, y


def collate(batch):
    import torch

    xs, ys = zip(*batch)
    shapes = {name: {tuple(x[name].shape) for x in xs} for name in xs[0]}
    bad = {n: s for n, s in shapes.items() if len(s) > 1}
    if bad:
        raise ValueError(f"inputs have different shapes within a batch: {bad}. "
                         "Add a crop step (fixed shape) to preprocessing.")
    return ({name: torch.stack([x[name] for x in xs]) for name in xs[0]},
            torch.as_tensor(ys, dtype=torch.long))


# ---- folds -------------------------------------------------------------------------

def assign_folds(cfg: ExperimentConfig, man: Manifest) -> np.ndarray:
    """Fold id per subject: frozen (column / file), group-aware, or stratified."""
    cv, df = cfg.cv, man.df
    if cv.fold_column:
        if cv.fold_column not in df:
            raise ValueError(f"fold_column {cv.fold_column!r} not in manifest")
        folds = df[cv.fold_column].astype(int).to_numpy()
    elif cv.folds_file:
        ff = pd.read_csv(cfg.resolve(cv.folds_file))
        id_col = cfg.data.id if cfg.data.id in ff else ff.columns[0]
        fold_col = next((c for c in ("fold", "fold_id", "cv_fold", "fold_index") if c in ff), None)
        if fold_col is None:
            raise ValueError(f"{cv.folds_file}: no fold column (fold, fold_id, cv_fold, fold_index)")
        as_int = ff[fold_col].astype(str).str.lower().str.replace("fold_", "").str.replace("fold", "").astype(int)
        lookup = dict(zip(ff[id_col].astype(str), as_int))
        missing = [i for i in man.ids if i not in lookup]
        if missing:
            raise ValueError(f"{len(missing)} subjects have no fold in {cv.folds_file}: {missing[:5]}")
        folds = np.array([lookup[i] for i in man.ids])
    else:
        from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, StratifiedKFold

        y = man.y
        if cv.by_group:
            if not cfg.data.group or cfg.data.group not in df:
                raise ValueError("cv.by_group needs data.group (a column in the manifest)")
            groups = df[cfg.data.group].astype(str).to_numpy()
            n_groups = len(np.unique(groups))
            if n_groups < cv.folds:
                raise ValueError(f"cv.by_group: {n_groups} groups < {cv.folds} folds")
            splitter = (StratifiedGroupKFold(cv.folds, shuffle=True, random_state=cv.seed)
                        if cv.stratify else GroupKFold(cv.folds))
            split = splitter.split(np.zeros(len(y)), y, groups)
        else:
            from sklearn.model_selection import KFold

            splitter = (StratifiedKFold(cv.folds, shuffle=True, random_state=cv.seed) if cv.stratify
                        else KFold(cv.folds, shuffle=True, random_state=cv.seed))
            split = splitter.split(np.zeros(len(y)), y)
        folds = np.empty(len(y), dtype=int)
        for k, (_, test) in enumerate(split):
            folds[test] = k
    uniq = np.unique(folds)
    if len(uniq) < 2:
        raise ValueError("need at least 2 folds")
    return folds
