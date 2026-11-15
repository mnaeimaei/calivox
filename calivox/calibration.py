"""Post-hoc probability calibration, evaluated with cross-fitting.

Methods (binary):  none · temperature (z / T) · platt (a·z + b) · isotonic (on probabilities)
Methods (multi-class): none · temperature

Cross-fitting: the out-of-fold predictions are split into K folds; each fold is calibrated
by a calibrator fitted on the other K-1 folds. A calibrator is therefore never evaluated
on the subjects it was fitted on. The deployed calibrator is then fitted on all of them.

Every calibrator maps logits to calibrated logits, so calibrated members can be averaged
in logit space by the ensemble.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold

from .metrics import binary_log_loss, logit, metrics_from_logits, multiclass_log_loss, sigmoid, softmax

BINARY_METHODS = ("none", "temperature", "platt", "isotonic")
MULTICLASS_METHODS = ("none", "temperature")


@dataclass
class Calibrator:
    method: str
    params: dict

    def apply(self, z: np.ndarray) -> np.ndarray:
        """Logits in -> calibrated logits out."""
        m, p = self.method, self.params
        if m == "none":
            return np.asarray(z, dtype=float)
        if m == "temperature":
            return np.asarray(z, dtype=float) / p["T"]
        if m == "platt":
            return p["a"] * np.asarray(z, dtype=float) + p["b"]
        if m == "isotonic":
            prob = np.interp(sigmoid(z), p["x"], p["y"])
            return logit(prob)
        raise ValueError(f"unknown calibration method {m!r}")

    def to_dict(self) -> dict:
        return {"method": self.method, "params": self.params}

    @classmethod
    def from_dict(cls, d: dict) -> "Calibrator":
        return cls(d["method"], d.get("params", {}))


def _nll(z: np.ndarray, y: np.ndarray) -> float:
    return binary_log_loss(y, sigmoid(z)) if z.ndim == 1 else multiclass_log_loss(y, softmax(z))


def fit_calibrator(method: str, z: np.ndarray, y: np.ndarray) -> Calibrator:
    z = np.asarray(z, dtype=float)
    y = np.asarray(y).astype(int)
    if method == "none":
        return Calibrator("none", {})
    if method == "temperature":
        res = minimize_scalar(lambda logT: _nll(z / np.exp(logT), y), bounds=(-3.0, 3.0), method="bounded")
        return Calibrator("temperature", {"T": float(np.exp(res.x))})
    if z.ndim != 1:
        raise ValueError(f"{method} calibration is binary only; use temperature for multi-class")
    if method == "platt":
        res = minimize(lambda ab: _nll(ab[0] * z + ab[1], y), x0=np.array([1.0, 0.0]), method="L-BFGS-B")
        return Calibrator("platt", {"a": float(res.x[0]), "b": float(res.x[1])})
    if method == "isotonic":
        iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(sigmoid(z), y)
        return Calibrator("isotonic", {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()})
    raise ValueError(f"unknown calibration method {method!r}")


def cross_fit(method: str, z: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 2026) -> np.ndarray:
    """Cross-fitted calibrated logits: every row calibrated by a calibrator not fitted on it."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    if method == "none":
        return z.copy()
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for tr, te in skf.split(np.zeros(len(y)), y):
        out[te] = fit_calibrator(method, z[tr], y[tr]).apply(z[te])
    return out


def evaluate_calibration(z: np.ndarray, y: np.ndarray, methods: list[str], folds: int = 5,
                         seed: int = 2026, threshold: float = 0.5) -> dict:
    """Cross-fitted metrics per method, the selected method (lowest log loss) and its
    calibrator fitted on all rows."""
    binary = np.asarray(z).ndim == 1
    allowed = BINARY_METHODS if binary else MULTICLASS_METHODS
    methods = [m for m in methods if m in allowed]
    results = {}
    for m in methods:
        zc = cross_fit(m, z, y, folds, seed)
        results[m] = metrics_from_logits(y, zc, binary, threshold)
    best = min(results, key=lambda m: results[m]["log_loss"])
    return {"methods": results, "selected": best, "calibrator": fit_calibrator(best, z, y).to_dict()}
