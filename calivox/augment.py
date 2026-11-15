"""Mild 3D augmentation for training batches.

Each call draws a random affine (small rotation about the three axes, translation and
isotropic scaling) with probability ``spatial_probability``, then optional intensity
scaling, Gaussian noise relative to the foreground standard deviation, and optional
smoothing. The transformations are kept mild so they do not distort small anatomy.
Validation, test and inference data are never augmented.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .config import AugmentConfig


def _rotation(ax: float, ay: float, az: float) -> torch.Tensor:
    cx, sx, cy, sy, cz, sz = math.cos(ax), math.sin(ax), math.cos(ay), math.sin(ay), math.cos(az), math.sin(az)
    rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
    ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
    return rz @ ry @ rx


class Augmenter:
    def __init__(self, cfg: AugmentConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.gen = torch.Generator()
        if seed is not None:
            self.gen.manual_seed(seed)

    def _u(self, lo: float, hi: float) -> float:
        return float(lo + (hi - lo) * torch.rand(1, generator=self.gen))

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """x: (C, D, H, W) -> augmented copy."""
        c = self.cfg
        out = x
        if torch.rand(1, generator=self.gen) < c.spatial_probability:
            r = math.radians(c.rotation_degrees)
            scale = self._u(1 - c.scale_fraction, 1 + c.scale_fraction)
            t = torch.tensor([self._u(-c.translation_fraction, c.translation_fraction) * 2 for _ in range(3)])
            if c.spatial_mode == "inplane_2d":
                # one 2D transform for every depth slice: no interpolation across depth
                # (grid coordinates are (W, H, D); the D row stays the identity)
                rot = _rotation(0.0, 0.0, self._u(-r, r))
                lin = rot.clone()
                lin[:2, :2] /= scale
                t[2] = 0.0
            else:
                rot = _rotation(self._u(-r, r), self._u(-r, r), self._u(-r, r))
                lin = rot / scale
            theta = torch.cat([lin, t[:, None]], dim=1).unsqueeze(0)
            grid = F.affine_grid(theta, (1, *out.shape), align_corners=False)
            out = F.grid_sample(out.unsqueeze(0), grid, mode="bilinear", padding_mode="zeros",
                                align_corners=False)[0]
        if torch.rand(1, generator=self.gen) < c.intensity_probability:
            out = out * self._u(1 - c.intensity_scale_fraction, 1 + c.intensity_scale_fraction)
        if torch.rand(1, generator=self.gen) < c.noise_probability:
            fg = out[out > 0]
            sd = float(fg.std()) if fg.numel() > 1 else float(out.std())
            out = out + torch.randn(out.shape, generator=self.gen) * sd * c.noise_std_fraction
        if c.smoothing_probability and torch.rand(1, generator=self.gen) < c.smoothing_probability:
            out = gaussian_blur3d(out, c.smoothing_sigma)
        return out


def gaussian_blur3d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of a (C, D, H, W) tensor."""
    radius = max(1, int(round(3 * sigma)))
    k = torch.exp(-(torch.arange(-radius, radius + 1, dtype=torch.float32) ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    y = x.unsqueeze(0)
    for dim in range(3):
        shape = [1, 1, 1, 1, 1]
        shape[2 + dim] = len(k)
        w = k.view(shape).repeat(x.shape[0], 1, 1, 1, 1)
        pad = [0, 0, 0, 0, 0, 0]
        pad[2 * (2 - dim)] = pad[2 * (2 - dim) + 1] = radius
        y = F.conv3d(F.pad(y, pad, mode="replicate"), w, groups=x.shape[0])
    return y[0]
