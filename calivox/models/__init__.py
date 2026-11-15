"""Model = one backbone per input + a classification head (late fusion for 2+ inputs)."""
from __future__ import annotations

from torch import Tensor, nn

from .backbones import BACKBONES, DEFAULT_DROPOUT, build_backbone, register_backbone

__all__ = ["BACKBONES", "Classifier", "build_backbone", "build_model", "register_backbone"]


class Classifier(nn.Module):
    """Independent backbones per input -> concatenated features -> dropout -> linear head.

    Binary tasks have one output logit; multi-class tasks have one logit per class.
    """

    def __init__(self, backbones: dict[str, nn.Module], n_outputs: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.backbones = nn.ModuleDict(backbones)
        n = sum(b.n_features for b in backbones.values())
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(n, n_outputs)
        nn.init.zeros_(self.head.bias)

    def features(self, x: dict[str, Tensor]) -> Tensor:
        import torch

        return torch.cat([self.backbones[k](x[k]) for k in self.backbones], dim=1)

    def forward(self, x: dict[str, Tensor]) -> Tensor:
        out = self.head(self.dropout(self.features(x)))
        return out.squeeze(-1) if out.shape[-1] == 1 else out


def build_model(name: str, inputs: list[str], n_outputs: int, pretrained: str | None = None,
                params: dict | None = None, fusion_dropout: float = 0.0) -> Classifier:
    params = dict(params or {})
    dropout = params.pop("dropout", DEFAULT_DROPOUT.get(name, 0.0))
    backbones = {i: build_backbone(name, pretrained, **params) for i in inputs}
    # single input: the backbone's head dropout; several inputs: the fusion head's own dropout
    return Classifier(backbones, n_outputs, fusion_dropout if len(inputs) > 1 else dropout)
