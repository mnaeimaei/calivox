"""Experiment configuration: one YAML file per use case.

    name: datscan
    task: {type: binary, positive: 1}           # binary | multiclass
    data:
      manifest: data/manifest.csv               # one row per subject
      id: subject_id
      label: label
      group: site                               # optional: site / scanner, for group-aware splits
      inputs: {whole: whole_path, roi: roi_path}  # named image inputs -> manifest columns
    preprocess:                                 # optional, per input, cached as .npy
      whole: [{op: reorient}, {op: resample, spacing: [2.46, 2.46, 2.46]},
              {op: normalize, method: reference_mask, mask: templates/occipital.nii.gz},
              {op: crop, mode: center, shape: [192, 192, 160]}]
    scenario: {inputs: [roi], augment: true}    # which inputs the model sees (2 = late fusion)
    model: {name: r3d18, pretrained: kinetics400}
    cv: {folds: 5, seed: 2026}
    training: {epochs: 100, batch_size: 2, learning_rate: 1.0e-4}
    calibration: {methods: [none, temperature, platt, isotonic]}

``scenarios`` and ``models`` blocks name several of each for ``calivox benchmark``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, model_validator


def load_plugin(spec: str, base_dir: str = ".") -> None:
    """Import a plugin once: a module name (``my_pkg.steps``) or a ``.py`` file path."""
    import importlib
    import importlib.util
    import sys

    if not spec.endswith(".py"):
        importlib.import_module(spec)
        return
    path = Path(spec) if Path(spec).is_absolute() else Path(base_dir) / spec
    path = path.resolve()
    key = f"calivox_plugin_{abs(hash(str(path)))}"
    if key in sys.modules:
        return
    if not path.exists():
        raise ValueError(f"plugin file not found: {path}")
    mod_spec = importlib.util.spec_from_file_location(key, path)
    module = importlib.util.module_from_spec(mod_spec)
    sys.modules[key] = module
    mod_spec.loader.exec_module(module)


class TaskConfig(BaseModel):
    type: Literal["binary", "multiclass"] = "binary"
    positive: str | int | None = None   # binary: the positive label (default: the largest)
    threshold: float = 0.5              # binary decision threshold for threshold metrics


class DataConfig(BaseModel):
    manifest: str
    id: str = "subject_id"
    label: str = "label"
    group: str | None = None            # e.g. site / scanner / acquisition protocol
    inputs: dict[str, str]              # input name -> manifest column with the image path
    root: str | None = None             # image paths are relative to this (default: manifest folder)
    depth_axis: Literal["x", "y", "z"] = "z"  # array axis used as depth D in (D, H, W)

    @model_validator(mode="after")
    def _check(self) -> "DataConfig":
        if not self.inputs:
            raise ValueError("data.inputs must name at least one image column")
        return self


class AugmentConfig(BaseModel):
    """Mild 3D augmentation, applied to training batches only."""
    spatial_probability: float = 0.80
    rotation_degrees: float = 5.0
    translation_fraction: float = 0.03
    scale_fraction: float = 0.05
    intensity_probability: float = 0.50
    intensity_scale_fraction: float = 0.05
    noise_probability: float = 0.30
    noise_std_fraction: float = 0.01      # x foreground std
    smoothing_probability: float = 0.0    # optional mild Gaussian smoothing
    smoothing_sigma: float = 0.5
    # 3d: rotate/scale/translate in 3D.  inplane_2d: the same 2D transform on every depth
    # slice, never mixing slices (for 2.5D models).
    spatial_mode: Literal["3d", "inplane_2d"] = "3d"


class ScenarioConfig(BaseModel):
    inputs: list[str]                     # one input, or two for late fusion
    augment: bool = False

    @model_validator(mode="after")
    def _check(self) -> "ScenarioConfig":
        if not 1 <= len(self.inputs) <= 4:
            raise ValueError("a scenario uses 1 to 4 inputs")
        return self


class ModelConfig(BaseModel):
    name: str = "simple3d"
    pretrained: str | None = None         # e.g. kinetics400, imagenet, or a weights file path
    params: dict = Field(default_factory=dict)
    fusion_dropout: float = 0.0
    # Per-model training overrides, e.g. batch_size: {roi: 8, whole: 2, whole+roi: 1}
    # (keys are the scenario inputs joined by "+"; a plain int applies to all scenarios).
    batch_size: int | dict[str, int] | None = None
    augment: dict = Field(default_factory=dict)   # overrides of the augment section


class CVConfig(BaseModel):
    folds: int = 5
    seed: int = 2026
    stratify: bool = True
    by_group: bool = False                # keep each group (site) within one fold
    fold_column: str | None = None        # use frozen fold ids from the manifest
    folds_file: str | None = None         # or from a CSV: id,fold
    # Which subjects pick the best epoch (early stopping / checkpoint):
    #   inner    a stratified split of the TRAINING folds -> the held-out fold stays unseen
    #   heldout  the held-out fold itself (slightly optimistic OOF; matches some protocols)
    validation: Literal["inner", "heldout"] = "inner"
    inner_fraction: float = 0.15

    @model_validator(mode="after")
    def _check(self) -> "CVConfig":
        if self.folds < 2:
            raise ValueError("cv.folds must be >= 2")
        return self


class TrainingConfig(BaseModel):
    epochs: int = 100
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    min_learning_rate: float = 1e-7
    early_stopping_patience: int = 15
    warmup_epochs: int = 5                # no early stopping before this
    grad_clip: float | None = 1.0
    amp: bool = True                      # used on CUDA only
    seed: int = 2026
    device: str = "auto"
    num_workers: int = 0
    num_threads: int | None = None
    cache_in_memory: bool = False


class CalibrationConfig(BaseModel):
    methods: list[Literal["none", "temperature", "platt", "isotonic"]] = Field(
        default_factory=lambda: ["none", "temperature", "platt", "isotonic"])
    folds: int = 5                        # cross-fitting folds
    seed: int = 2026


class EnsembleConfig(BaseModel):
    max_members: int = 5
    candidates: int = 12                  # shortlist size (best by calibrated OOF log loss)
    final_temperature: bool = True


class ExperimentConfig(BaseModel):
    name: str = "experiment"
    description: str = ""
    task: TaskConfig = Field(default_factory=TaskConfig)
    data: DataConfig
    preprocess: dict[str, list[dict]] = Field(default_factory=dict)
    scenario: ScenarioConfig | None = None
    scenarios: dict[str, ScenarioConfig] = Field(default_factory=dict)  # for benchmark
    model: ModelConfig = Field(default_factory=ModelConfig)
    models: dict[str, ModelConfig] = Field(default_factory=dict)        # for benchmark
    augment: AugmentConfig = Field(default_factory=AugmentConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    calibration: CalibrationConfig = Field(default_factory=CalibrationConfig)
    ensemble: EnsembleConfig = Field(default_factory=EnsembleConfig)
    output_dir: str = "runs"
    cache_dir: str = "cache"
    base_dir: str = "."
    # Python modules or .py files (relative to the config) imported before the run, to
    # register custom preprocessing steps (@register_step) or backbones (@register_backbone).
    plugins: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> "ExperimentConfig":
        for plugin in self.plugins:
            load_plugin(plugin, self.base_dir)
        if self.scenario is None:
            self.scenario = (next(iter(self.scenarios.values())) if self.scenarios
                             else ScenarioConfig(inputs=[next(iter(self.data.inputs))]))
        for s in [self.scenario, *self.scenarios.values()]:
            missing = [i for i in s.inputs if i not in self.data.inputs]
            if missing:
                raise ValueError(f"scenario uses unknown inputs {missing}; defined: {list(self.data.inputs)}")
        unknown = set(self.preprocess) - set(self.data.inputs)
        if unknown:
            raise ValueError(f"preprocess defined for unknown inputs {sorted(unknown)}")
        return self

    # -- io -------------------------------------------------------------------------
    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        data.setdefault("base_dir", str(path.parent.resolve()))
        return cls(**data)

    def to_yaml(self, keep_base_dir: bool = True) -> str:
        data = self.model_dump(exclude=None if keep_base_dir else {"base_dir"}, mode="json")
        return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)

    def resolve(self, p: str | None) -> Path | None:
        if p is None:
            return None
        p = Path(p)
        return p if p.is_absolute() else Path(self.base_dir) / p

    def override(self, **kw) -> "ExperimentConfig":
        data = self.model_dump()
        for key, value in kw.items():
            if value is None:
                continue
            node = data
            *parents, leaf = key.split(".")
            for p in parents:
                node = node.setdefault(p, {})
            node[leaf] = value
        return ExperimentConfig(**data)

    def for_run(self, model: str | None = None, scenario: str | None = None,
                epochs: int | None = None) -> "ExperimentConfig":
        """Config for one benchmark cell: model x scenario."""
        over: dict = {"training.epochs": epochs}
        if model is not None:
            m = self.models.get(model)
            over["model"] = (m.model_dump() if m else {"name": model})
        if scenario is not None:
            if scenario not in self.scenarios:
                raise ValueError(f"unknown scenario {scenario!r}; defined: {list(self.scenarios)}")
            over["scenario"] = self.scenarios[scenario].model_dump()
        cfg = self.override(**over)
        bs, key = cfg.model.batch_size, "+".join(cfg.scenario.inputs)
        if isinstance(bs, dict) and key in bs:
            cfg = cfg.override(**{"training.batch_size": bs[key]})
        elif isinstance(bs, int):
            cfg = cfg.override(**{"training.batch_size": bs})
        if cfg.model.augment:
            cfg = cfg.override(**{f"augment.{k}": v for k, v in cfg.model.augment.items()})
        return cfg

    @property
    def run_name(self) -> str:
        s = self.scenario
        return f"{self.model.name}__{'+'.join(s.inputs)}{'__aug' if s.augment else ''}"
