"""Probability-quality and classification metrics.

Binary (from the positive-class probability):
    log_loss, auroc, auprc, brier, ece, and at the decision threshold:
    balanced_accuracy, sensitivity, specificity, precision, f1, confusion_matrix
Multi-class (from the probability matrix):
    log_loss, accuracy, balanced_accuracy, macro_f1, macro_auroc (one-vs-rest), brier, ece (top label)

Log loss is the primary metric: it rewards correct *and* well-calibrated probabilities and
penalises confident mistakes heavily.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, confusion_matrix,
                             f1_score, precision_score, roc_auc_score)

EPS = 1e-7


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def binary_log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def multiclass_log_loss(y: np.ndarray, P: np.ndarray) -> float:
    P = np.clip(np.asarray(P, dtype=float), EPS, 1.0)
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def ece(correct_or_y: np.ndarray, conf: np.ndarray, n_bins: int = 15) -> float:
    """Expected calibration error with equal-width bins on [0, 1]."""
    conf = np.asarray(conf, dtype=float)
    target = np.asarray(correct_or_y, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1]), 0, n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(target[m].mean() - conf[m].mean())
    return float(total)


def binary_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    pred = (p >= threshold).astype(int)
    both = len(np.unique(y)) > 1
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "log_loss": binary_log_loss(y, p),
        "auroc": float(roc_auc_score(y, p)) if both else None,
        "auprc": float(average_precision_score(y, p)) if both else None,
        "brier": float(np.mean((p - y) ** 2)),
        "ece": ece(y, p),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "sensitivity": float(tp / (tp + fn)) if tp + fn else None,
        "specificity": float(tn / (tn + fp)) if tn + fp else None,
        "precision": float(precision_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "threshold": float(threshold),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y)), "positive_rate": float(y.mean()),
    }


def multiclass_metrics(y: np.ndarray, P: np.ndarray) -> dict:
    y = np.asarray(y).astype(int)
    P = np.asarray(P, dtype=float)
    pred = P.argmax(axis=1)
    K = P.shape[1]
    onehot = np.eye(K)[y]
    try:
        auroc = float(roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(K))))
    except ValueError:
        auroc = None
    return {
        "log_loss": multiclass_log_loss(y, P),
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=list(range(K)), zero_division=0)),
        "macro_auroc": auroc,
        "brier": float(np.mean(((P - onehot) ** 2).sum(axis=1))),
        "ece": ece(pred == y, P.max(axis=1)),
        "confusion_matrix": confusion_matrix(y, pred, labels=list(range(K))).tolist(),
        "n": int(len(y)),
    }


def metrics_from_logits(y: np.ndarray, z: np.ndarray, binary: bool, threshold: float = 0.5) -> dict:
    return binary_metrics(y, sigmoid(z), threshold) if binary else multiclass_metrics(y, softmax(z))
