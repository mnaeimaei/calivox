"""Backbones: each maps a (B, 1, D, H, W) volume to a (B, n_features) vector.

    simple3d     4-stage residual 3D CNN (~3.6 M parameters)
    resnet3d     MONAI 3D ResNet (18/34/50); optional MedicalNet weights file
    densenet3d   MONAI 3D DenseNet-121
    r3d18        torchvision R3D-18 (depth as the temporal axis); optional Kinetics-400 weights
    swin3d_t     torchvision Swin3D-Tiny; optional Kinetics-400 weights
    slice2p5d    2D ResNet-18 on selected depth slices + learned slice attention; optional ImageNet weights

Pretrained RGB stems are converted to one channel by summing the colour kernels
(W_gray = W_R + W_G + W_B), so the pretrained response to a uniform input is kept.
Register a new backbone with ``@register_backbone("name")``.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import torch
from torch import Tensor, nn

BACKBONES: dict[str, Callable[..., nn.Module]] = {}
DEFAULT_DROPOUT: dict[str, float] = {}


def register_backbone(name: str, dropout: float = 0.0):
    def deco(fn):
        BACKBONES[name] = fn
        DEFAULT_DROPOUT[name] = dropout
        return fn
    return deco


def build_backbone(name: str, pretrained: str | None = None, **params) -> nn.Module:
    if name not in BACKBONES:
        raise ValueError(f"unknown model {name!r}; available: {sorted(BACKBONES)}")
    bb = BACKBONES[name](pretrained=pretrained, **params)
    if not hasattr(bb, "n_features"):
        raise TypeError(f"backbone {name!r} must set n_features")
    return bb


def _gray_stem(conv: nn.Module, cls) -> nn.Module:
    """Replace a 3-channel conv by a 1-channel conv with summed RGB kernels."""
    new = cls(1, conv.out_channels, kernel_size=conv.kernel_size, stride=conv.stride,
              padding=conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    return new


# ---- simple3d ----------------------------------------------------------------------

class _ResBlock3D(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm3d(cout)
        self.conv2 = nn.Conv3d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm3d(cout)
        self.relu = nn.ReLU(inplace=True)
        self.down = (nn.Sequential(nn.Conv3d(cin, cout, 1, stride, bias=False), nn.BatchNorm3d(cout))
                     if stride != 1 or cin != cout else nn.Identity())

    def forward(self, x: Tensor) -> Tensor:
        out = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(out)) + self.down(x))


class Simple3D(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv3d(1, w, 3, 1, 1, bias=False), nn.BatchNorm3d(w),
                                  nn.ReLU(inplace=True), nn.MaxPool3d(2, 2))
        self.stages = nn.Sequential(_ResBlock3D(w, w), _ResBlock3D(w, 2 * w, 2),
                                    _ResBlock3D(2 * w, 4 * w, 2), _ResBlock3D(4 * w, 8 * w, 2))
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.n_features = 8 * w
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.pool(self.stages(self.stem(x))).flatten(1)


@register_backbone("simple3d", dropout=0.3)
def simple3d(pretrained=None, width: int = 32, **_):
    """4-stage residual 3D CNN (~3.6 M parameters), trained from scratch."""
    if pretrained:
        raise ValueError("simple3d has no pretrained weights")
    return Simple3D(width)


# ---- MONAI ResNet / DenseNet -------------------------------------------------------

class _Wrap(nn.Module):
    def __init__(self, net: nn.Module, n_features: int, fn=None) -> None:
        super().__init__()
        self.net, self.n_features, self.fn = net, n_features, fn

    def forward(self, x: Tensor) -> Tensor:
        return self.fn(self.net, x) if self.fn else self.net(x)


def _load_medicalnet(net: nn.Module, path: Path) -> None:
    """Load a MedicalNet (Med3D) checkpoint into a MONAI ResNet encoder (head not loaded)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items() if not k.startswith(("module.conv_seg", "conv_seg"))}
    own = net.state_dict()
    usable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    net.load_state_dict(usable, strict=False)
    skipped = len(own) - len(usable)
    if not usable:
        raise ValueError(f"no compatible tensors in {path.name}; check depth / shortcut_type")
    if skipped:
        warnings.warn(f"MedicalNet: loaded {len(usable)} tensors, {skipped} encoder tensors left at init")


@register_backbone("resnet3d")
def resnet3d(pretrained=None, depth: int = 18, shortcut_type: str = "B", **kw):
    """MONAI 3D ResNet-18/34/50; pretrained = path to MedicalNet weights (optional)."""
    from monai.networks.nets import resnet

    fn = {18: resnet.resnet18, 34: resnet.resnet34, 50: resnet.resnet50}[int(depth)]
    net = fn(spatial_dims=3, n_input_channels=1, feed_forward=False, shortcut_type=shortcut_type, **kw)
    if pretrained:
        p = Path(pretrained)
        if not p.exists():
            raise FileNotFoundError(f"resnet3d: pretrained weights file not found: {p} "
                                    "(e.g. MedicalNet resnet_18_23dataset.pth)")
        _load_medicalnet(net, p)
    return _Wrap(net, 512 if int(depth) < 50 else 2048)


@register_backbone("densenet3d")
def densenet3d(pretrained=None, **kw):
    """MONAI 3D DenseNet-121 (needs >= 32 voxels per axis)."""
    from monai.networks.nets import DenseNet121

    if pretrained:
        raise ValueError("densenet3d has no built-in pretrained weights")
    net = DenseNet121(spatial_dims=3, in_channels=1, out_channels=1, **kw)
    n = net.class_layers.out.in_features

    def features(m, x):
        if min(x.shape[2:]) < 32:
            raise ValueError(f"densenet3d needs at least 32 voxels per axis, got {tuple(x.shape[2:])}")
        return torch.flatten(nn.functional.adaptive_avg_pool3d(torch.relu(m.features(x)), 1), 1)

    return _Wrap(net, n, features)


# ---- torchvision video models ------------------------------------------------------

def _video_weights(name: str, pretrained: str | None):
    if not pretrained:
        return None
    if pretrained.lower() not in {"kinetics400", "kinetics"}:
        raise ValueError(f"{name}: pretrained must be 'kinetics400' or empty")
    import torchvision.models.video as v

    return {"r3d18": v.R3D_18_Weights.KINETICS400_V1, "swin3d_t": v.Swin3D_T_Weights.KINETICS400_V1}[name]


@register_backbone("r3d18")
def r3d18(pretrained=None, **_):
    """torchvision R3D-18, depth as the temporal axis; pretrained: kinetics400 (optional).

    Do not apply Kinetics RGB normalisation to medical volumes."""
    import torchvision.models.video as v

    net = v.r3d_18(weights=_video_weights("r3d18", pretrained))
    net.stem[0] = _gray_stem(net.stem[0], nn.Conv3d)
    net.fc = nn.Identity()
    return _Wrap(net, 512)


@register_backbone("swin3d_t")
def swin3d_t(pretrained=None, **_):
    """torchvision Swin3D-Tiny transformer; pretrained: kinetics400 (optional)."""
    import torchvision.models.video as v

    net = v.swin3d_t(weights=_video_weights("swin3d_t", pretrained))
    net.patch_embed.proj = _gray_stem(net.patch_embed.proj, nn.Conv3d)
    net.head = nn.Identity()
    return _Wrap(net, 768)


# ---- 2.5D: 2D ResNet-18 on slices + attention --------------------------------------

class SliceAttention2p5D(nn.Module):
    """Encode selected depth slices with one shared 2D ResNet-18, pool them with learned
    attention (a scalar score per slice, softmax over slices)."""

    def __init__(self, pretrained: str | None = None, slices: int | str = "all") -> None:
        super().__init__()
        import torchvision.models as tvm

        if pretrained and pretrained.lower() != "imagenet":
            raise ValueError("slice2p5d: pretrained must be 'imagenet' or empty")
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.conv1 = _gray_stem(net.conv1, nn.Conv2d)
        net.fc = nn.Identity()
        self.encoder, self.slices, self.n_features = net, slices, 512
        self.score = nn.Linear(512, 1)
        self.last_attention: Tensor | None = None

    def _select(self, D: int) -> Tensor:
        if self.slices == "all" or int(self.slices) >= D:
            return torch.arange(D)
        return torch.linspace(0, D - 1, int(self.slices)).round().long()

    def forward(self, x: Tensor) -> Tensor:
        B, C, D, H, W = x.shape
        idx = self._select(D).to(x.device)
        s = x[:, :, idx].permute(0, 2, 1, 3, 4).reshape(B * len(idx), C, H, W)
        f = self.encoder(s).view(B, len(idx), -1)                    # (B, S, 512)
        a = torch.softmax(self.score(f).squeeze(-1), dim=1)          # (B, S)
        self.last_attention = a.detach()
        return (a.unsqueeze(-1) * f).sum(dim=1)


@register_backbone("slice2p5d")
def slice2p5d(pretrained=None, slices: int | str = "all", **_):
    """2.5D: 2D ResNet-18 per depth slice + learned slice attention; pretrained: imagenet."""
    return SliceAttention2p5D(pretrained, slices)
