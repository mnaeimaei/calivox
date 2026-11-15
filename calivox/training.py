"""Cross-validated training with out-of-fold (OOF) predictions.

For every fold k: train a fresh model on the other folds, pick the best epoch by
validation loss (inner split by default), reload that checkpoint and predict fold k.
Concatenating the K held-out predictions gives one OOF prediction per subject, each from
a model that never trained on that subject.

A run folder contains

    config.yaml           the exact configuration
    folds.csv             subject -> fold
    fold<k>/model.pt      best checkpoint of fold k
    fold<k>/history.json  per-epoch losses
    oof.csv               subject, fold, label, logits, probabilities
    metrics.json          OOF metrics (uncalibrated)
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import Manifest, VolumeDataset, assign_folds, collate, load_manifest, preprocess_inputs
from .metrics import metrics_from_logits, sigmoid, softmax


@dataclass
class RunResult:
    run_dir: Path
    oof: pd.DataFrame
    metrics: dict
    seconds: float


def _device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _seed_all(seed: int) -> None:
    import random

    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _inner_split(y: np.ndarray, idx: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    tr, va = train_test_split(idx, test_size=fraction, stratify=y[idx], random_state=seed)
    return np.sort(tr), np.sort(va)


def n_outputs(cfg: ExperimentConfig, man: Manifest) -> int:
    return 1 if cfg.task.type == "binary" else len(man.classes)


def build_for(cfg: ExperimentConfig, man: Manifest):
    from .models import build_model

    m = cfg.model
    return build_model(m.name, cfg.scenario.inputs, n_outputs(cfg, man), m.pretrained, m.params, m.fusion_dropout)


def _loss_fn(binary: bool):
    from torch import nn

    if binary:
        return lambda out, y: nn.functional.binary_cross_entropy_with_logits(out, y.float())
    return lambda out, y: nn.functional.cross_entropy(out, y)


def predict_logits(model, loader, device) -> np.ndarray:
    import torch

    model.eval()
    outs = []
    with torch.no_grad():
        for x, _ in loader:
            outs.append(model({k: v.to(device) for k, v in x.items()}).float().cpu().numpy())
    return np.concatenate(outs)


def train_fold(cfg: ExperimentConfig, man: Manifest, paths: dict, y: np.ndarray, train_idx: np.ndarray,
               val_idx: np.ndarray, out_dir: Path, verbose: bool = False) -> list[dict]:
    """Train one model; ``y`` holds task labels (0/1 for binary, class index otherwise)."""
    import torch
    from torch.utils.data import DataLoader

    from .augment import Augmenter

    tr = cfg.training
    _seed_all(tr.seed)
    if tr.num_threads:
        torch.set_num_threads(tr.num_threads)
    device = _device(tr.device)
    binary = cfg.task.type == "binary"
    y = np.asarray(y, dtype=np.int64)
    sel = {k: paths[k] for k in cfg.scenario.inputs}
    aug = Augmenter(cfg.augment, seed=tr.seed) if cfg.scenario.augment else None
    ds_tr = VolumeDataset(sel, y, train_idx, cfg.data.depth_axis, aug, tr.cache_in_memory)
    ds_va = VolumeDataset(sel, y, val_idx, cfg.data.depth_axis, None, tr.cache_in_memory)
    g = torch.Generator().manual_seed(tr.seed)
    dl_tr = DataLoader(ds_tr, batch_size=tr.batch_size, shuffle=True, collate_fn=collate,
                       num_workers=tr.num_workers, generator=g)
    dl_va = DataLoader(ds_va, batch_size=max(tr.batch_size, 4), shuffle=False, collate_fn=collate,
                       num_workers=tr.num_workers)

    model = build_for(cfg, man).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tr.learning_rate, weight_decay=tr.weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=tr.plateau_factor,
                                                       patience=tr.plateau_patience, min_lr=tr.min_learning_rate)
    loss_fn = _loss_fn(binary)
    use_amp = tr.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_state, stale, history = None, None, 0, []
    for epoch in range(1, tr.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        run, n = 0.0, 0
        for x, yb in dl_tr:
            x = {k: v.to(device) for k, v in x.items()}
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=use_amp):
                loss = loss_fn(model(x), yb)
            scaler.scale(loss).backward()
            if tr.grad_clip:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tr.grad_clip)
            scaler.step(opt)
            scaler.update()
            run += loss.item() * len(yb)
            n += len(yb)
        z = predict_logits(model, dl_va, device)
        with torch.no_grad():
            val_loss = float(loss_fn(torch.from_numpy(z), torch.from_numpy(y[val_idx])).item())
        sched.step(val_loss)
        rec = {"epoch": epoch, "train_loss": round(run / max(n, 1), 5), "val_loss": round(val_loss, 5),
               "lr": opt.param_groups[0]["lr"], "seconds": round(time.perf_counter() - t0, 2)}
        history.append(rec)
        if verbose:
            print(f"    epoch {epoch:3d}  train {rec['train_loss']:.4f}  val {rec['val_loss']:.4f}  ({rec['seconds']}s)", flush=True)
        if best is None or val_loss < best:
            best, best_state, stale = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if epoch >= tr.warmup_epochs and stale >= tr.early_stopping_patience:
                break
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    return history


def cross_validate(cfg: ExperimentConfig, run_dir: str | Path | None = None, verbose: bool = False,
                   manifest: Manifest | None = None, paths: dict | None = None) -> RunResult:
    import torch
    from torch.utils.data import DataLoader

    t_start = time.perf_counter()
    man = manifest or load_manifest(cfg)
    paths = paths or preprocess_inputs(cfg, man, cfg.scenario.inputs)
    folds = assign_folds(cfg, man)
    run_dir = Path(run_dir) if run_dir else cfg.resolve(cfg.output_dir) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(cfg.to_yaml(), encoding="utf-8")
    pd.DataFrame({"id": man.ids, "fold": folds}).to_csv(run_dir / "folds.csv", index=False)

    binary = cfg.task.type == "binary"
    y_task = (man.y == man.positive_index).astype(int) if binary else man.y
    K = 1 if binary else len(man.classes)
    oof = np.full((len(man.ids), K), np.nan)
    device = _device(cfg.training.device)
    for k in sorted(np.unique(folds)):
        test_idx = np.where(folds == k)[0]
        train_all = np.where(folds != k)[0]
        if cfg.cv.validation == "inner":
            train_idx, val_idx = _inner_split(y_task, train_all, cfg.cv.inner_fraction, cfg.cv.seed + int(k))
        else:
            train_idx, val_idx = train_all, test_idx
        if verbose:
            print(f"  fold {k}: train {len(train_idx)}  val {len(val_idx)}  test {len(test_idx)}", flush=True)
        fold_dir = run_dir / f"fold{k}"
        train_fold(cfg, man, paths, y_task, train_idx, val_idx, fold_dir, verbose)
        model = build_for(cfg, man).to(device)
        model.load_state_dict(torch.load(fold_dir / "model.pt", map_location=device, weights_only=True))
        sel = {i: paths[i] for i in cfg.scenario.inputs}
        dl = DataLoader(VolumeDataset(sel, y_task, test_idx, cfg.data.depth_axis), batch_size=8,
                        shuffle=False, collate_fn=collate)
        oof[test_idx] = predict_logits(model, dl, device).reshape(len(test_idx), K)

    frame = pd.DataFrame({"id": man.ids, "fold": folds, "label": man.df["__label__"].to_numpy(),
                          "y": y_task})
    if binary:
        frame["logit"] = oof[:, 0]
        frame["prob"] = sigmoid(oof[:, 0])
    else:
        P = softmax(oof)
        for j, c in enumerate(man.classes):
            frame[f"logit_{c}"] = oof[:, j]
        for j, c in enumerate(man.classes):
            frame[f"prob_{c}"] = P[:, j]
    frame.to_csv(run_dir / "oof.csv", index=False)
    m = metrics_from_logits(y_task, oof[:, 0] if binary else oof, binary, cfg.task.threshold)
    seconds = time.perf_counter() - t_start
    summary = {"run": cfg.run_name, "model": cfg.model.name, "pretrained": cfg.model.pretrained,
               "inputs": cfg.scenario.inputs, "augment": cfg.scenario.augment, "folds": int(len(np.unique(folds))),
               "validation": cfg.cv.validation, "classes": [str(c) for c in man.classes],
               "seconds": round(seconds, 1), "oof": m}
    (run_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    return RunResult(run_dir, frame, summary, seconds)


def load_oof(run_dir: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """(logits, y, frame) from a run folder; logits are (N,) for binary, (N, K) otherwise."""
    df = pd.read_csv(Path(run_dir) / "oof.csv")
    if "logit" in df:
        return df["logit"].to_numpy(float), df["y"].to_numpy(int), df
    cols = [c for c in df.columns if c.startswith("logit_")]
    return df[cols].to_numpy(float), df["y"].to_numpy(int), df
