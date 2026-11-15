# CaliVox — Calibrated 3D Image Classification

[![CI](https://github.com/mnaeimaei/calivox/actions/workflows/ci.yml/badge.svg)](https://github.com/mnaeimaei/calivox/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.x-teal.svg)](https://monai.io/)
[![License MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**CaliVox** is a framework for classifying 3D images (MRI, CT, PET, SPECT or any volumetric scan) with **probabilities you can trust**. You give it a CSV manifest of NIfTI or NumPy volumes and one YAML file. It then:
- benchmarks 3D CNNs, a 2.5D slice-attention model and a 3D transformer;
- evaluates every run with out-of-fold cross-validation;
- calibrates the probabilities with cross-fitting;
- builds a calibrated ensemble and reports a *nested* estimate of how good that ensemble really is;
- freezes the result into a verified runtime bundle for new scans.

```text
manifest.csv + config.yaml
    ─▶ preprocess (reorient · resample · reference-region normalisation · ROI crop)   cached per step list
    ─▶ benchmark  models × input scenarios, K-fold CV, out-of-fold logits            calivox benchmark
    ─▶ calibrate  none / temperature / Platt / isotonic, cross-fitted                calivox calibrate
    ─▶ ensemble   calibrated logit mean + final temperature, nested estimate          calivox ensemble
    ─▶ bundle     frozen models + calibrators + SHA-256, equivalence-checked          calivox freeze / verify
    ─▶ predict    probabilities for new scans                                         calivox predict
```

The primary metric throughout is **log loss**: it rewards correct *and* well-calibrated probabilities, and punishes confident mistakes.

Two use cases are included:
- **DaT-SPECT phantoms.** 400 synthetic, DaT-SPECT-like brain volumes from three acquisition sites (normal vs pathologic striatal uptake), generated with `calivox synth` and benchmarked below.
- **MedMNIST3D.** A multi-class config with a data converter; bring your own download.

The framework was built from a real study: [**DaT-SPECT classification for Parkinsonian disorder assessment**](https://github.com/mnaeimaei/parkinson-datscan-classifier), 1,362 subjects and 60 experiments. That repository now runs its experiment layer on CaliVox configs.

**Phantom benchmark** (400 synthetic subjects, 5-fold CV, 14 model × scenario runs, all on 2 CPU cores):

| | Run | OOF log loss | AUROC |
| --- | --- | --- | --- |
| Best single run, calibrated | 3D ResNet-18 · ROI + augmentation · temperature | **0.2244** | 0.9675 |
| Calibrated ensemble, **nested** estimate | ResNet-18 ROI + Simple3D whole+ROI, final T = 0.825 | **0.2397** | 0.9699 |
| Same ensemble, selected and scored on the same data | | 0.2005 | 0.9752 |

Calibration improved 11 of the 14 runs, by up to 0.052 in log loss. The ensemble shows why CaliVox reports a nested estimate. Scored on the data used to choose its members, the ensemble looks clearly better than the best single run (0.2005 vs 0.2244). With the choice repeated inside outer folds, it is not (0.2397). [Full results below.](#use-cases-and-results)

---

## Main features

- **Any volumetric format the manifest points to:** `.nii` / `.nii.gz` (nibabel), `.npy`, `.npz`. Several inputs per subject (for example a whole volume and an ROI crop, or two modalities) can be combined by late fusion.
- **Preprocessing as a list of steps per input**, cached by a hash of the steps:
  - `reorient` (to RAS) and `resample` (target spacing);
  - `normalize`: z-score, min-max, percentile, or **reference-region** (divide by the mean/median inside a mask file or a per-subject mask);
  - `crop`: center, fixed box, mask center, or **peak**, which locates the brightest region inside a search box;
  - `clip`;
  - your own steps via `@register_step`.
- **Six backbones**, all single-channel and fusable:

  | Name | Model | Pretrained weights |
  | --- | --- | --- |
  | `simple3d` | 4-stage residual 3D CNN (~3.6 M parameters) | — |
  | `resnet3d` | MONAI 3D ResNet-18/34/50 | MedicalNet (file) |
  | `densenet3d` | MONAI 3D DenseNet-121 | — |
  | `r3d18` | torchvision R3D-18 | Kinetics-400 |
  | `swin3d_t` | torchvision Swin3D-Tiny | Kinetics-400 |
  | `slice2p5d` | 2D ResNet-18 per depth slice + learned slice attention | ImageNet |

  RGB-pretrained stems are converted to one channel by summing their kernels, which keeps the pretrained response to grey images. Your own backbones plug in via `@register_backbone`.
- **Scenarios:** named input combinations, each with or without augmentation (`S1: {inputs: [whole]}`, `S5: {inputs: [roi], augment: true}`, …), benchmarked against every model.
- **Augmentation**, applied to training batches only:
  - mild 3D affine transforms (rotation, scale, translation), intensity scaling, foreground-scaled noise and optional smoothing;
  - an `inplane_2d` mode for 2.5D models, which never interpolates across slices.
- **Cross-validation that respects your design:**
  - stratified, group-aware (site, scanner, patient) or frozen folds from a column or a file;
  - early stopping on an **inner split** of the training folds (default), so out-of-fold predictions stay unbiased; or on the held-out fold, to match existing protocols.
- **Cross-fitted calibration:** every subject is calibrated by a calibrator that never saw it. The best method per run is chosen by log loss.
- **Ensembles:**
  - greedy forward selection over calibrated members, then an equal-weight logit mean and a final temperature;
  - or a fixed member list (`--members spec.yaml`);
  - always with a **nested** estimate that repeats the selection inside outer folds.
- **Frozen runtime bundle.** It holds configs, fold checkpoints, calibrators, referenced masks and plugins, plus a SHA-256 for every file. `verify` re-runs each fold model from the bundle and checks that it reproduces its training-time out-of-fold logits.
- **Binary or multi-class**, CPU or GPU (mixed precision on CUDA), and fully seeded.

## Tech stack

| Concern | Tools |
| --- | --- |
| Deep learning | PyTorch, torchvision (R3D-18, Swin3D, ResNet-18), MONAI (3D ResNet, DenseNet) |
| Imaging I/O and geometry | nibabel, SciPy (resampling) |
| Validation, calibration | scikit-learn (folds, isotonic regression, metrics), SciPy (temperature / Platt fitting) |
| Configuration | YAML + Pydantic |
| Delivery | CLI (`calivox …`), Docker, GitHub Actions |

---

## Getting started

### 1. Install

```bash
git clone https://github.com/mnaeimaei/calivox && cd calivox
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # for CPU-only machines, install torch from the CPU index first
calivox models                 # list the backbones
```

### 2. Run the phantom use case

```bash
make phantoms     # 400 synthetic DaT-SPECT-like volumes -> examples/datscan-phantoms/data
make benchmark    # models x scenarios, 5-fold CV
make calibrate    # cross-fitted calibration of every run
make bundle       # ensemble -> frozen bundle -> equivalence check
```

### 3. Score new scans

```bash
calivox predict examples/datscan-phantoms/bundle --manifest new_scans.csv --out predictions.csv
```

`new_scans.csv` has the same id and path columns as the training manifest, without labels. The output holds, per subject, the ensemble probability, the predicted class and each member's calibrated probability.

---

## How to use the framework

### Describe the data

```yaml
name: my-study
task: {type: binary, positive: pathologic}          # or {type: multiclass}

data:
  manifest: data/manifest.csv     # one row per subject
  id: subject_id
  label: label
  group: site                     # optional: keep groups together in CV
  inputs:                         # model input name -> manifest column with a file path
    whole: scan_path
    roi: scan_path                # the same file can feed several inputs
```

### Preprocess each input

```yaml
preprocess:
  whole:
    - {op: reorient}
    - {op: resample, spacing: [2.0, 2.0, 2.0]}
    - {op: normalize, method: reference_mask, mask: data/reference_region.nii.gz}
    - {op: crop, mode: center, shape: [96, 96, 80]}
  roi:
    - {op: normalize, method: reference_mask, mask: data/reference_region.nii.gz}
    - {op: crop, mode: peak, shape: [44, 44, 36], search: [[30, 70], [40, 80], [20, 60]]}
```

Inputs without steps are read as they are, which suits data that a pipeline of your own has already preprocessed.

### Define scenarios and models

```yaml
scenarios:
  S1: {inputs: [whole]}
  S2: {inputs: [roi]}
  S3: {inputs: [whole, roi]}              # late fusion: one backbone per input
  S5: {inputs: [roi], augment: true}

models:
  simple3d: {name: simple3d}
  r3d18_k400: {name: r3d18, pretrained: kinetics400, batch_size: {roi: 8, whole: 2, whole+roi: 1}}
  attn2p5d: {name: slice2p5d, pretrained: imagenet, augment: {spatial_mode: inplane_2d}}
```

`batch_size` and `augment` can be set per model, and per scenario input through the `+`-joined keys.

### Train, calibrate, ensemble, freeze

```bash
calivox cv        config.yaml --model r3d18_k400 --scenario S5 -v   # one run
calivox benchmark config.yaml --skip-existing --write results.md    # the whole grid
calivox calibrate runs --write calibrated.md
calivox ensemble  runs --out ensemble.json                          # greedy + nested estimate
calivox freeze    ensemble.json --out bundle
calivox verify    bundle --manifest data/manifest.csv
```

Each run folder holds `config.yaml`, `folds.csv`, `fold<k>/model.pt`, `fold<k>/history.json`, `oof.csv`, `metrics.json` and, after calibration, `calibration.json`.

### Extend it

```python
# my_plugin.py  — listed in the config as   plugins: [my_plugin.py]
from calivox.volumes import Volume, register_step
from calivox.models import register_backbone

@register_step("hemisphere_flip")
def hemisphere_flip(vol: Volume, params: dict, row: dict, resolve) -> Volume:
    """A step gets the volume, its YAML parameters, the subject's manifest row and a path resolver."""
    return Volume(vol.array[::-1].copy(), vol.affine)

@register_backbone("tiny", dropout=0.2)
def tiny(pretrained=None, **params):
    """Any nn.Module mapping (B, 1, D, H, W) -> (B, n_features), with an n_features attribute."""
    ...
```

Plugins listed in a config are copied into frozen bundles, so a bundle stays self-contained. The DaT-SPECT repository uses this for its localised 2.5D slice window.

### Use it from Python

```python
from calivox import ExperimentConfig
from calivox.training import cross_validate
from calivox.calibration import evaluate_calibration

cfg = ExperimentConfig.from_yaml("config.yaml").for_run("simple3d", "S2")
run = cross_validate(cfg)
cal = evaluate_calibration(run.oof["logit"].to_numpy(), run.oof["y"].to_numpy(), ["none", "temperature", "platt"])
print(run.metrics["oof"]["log_loss"], cal["selected"])
```

### Tips

- **Report the nested ensemble estimate.** Choosing members on the same out-of-fold predictions you report is optimistic.
- **Prefer `cv.validation: inner`.** Selecting checkpoints on the held-out fold leaks a little information into the out-of-fold estimate.
- **Mind group structure.** If subjects share a scanner, site or patient, set `data.group` and `cv.by_group: true`.
- **Check input sizes.** DenseNet-121 needs at least 32 voxels per axis, and models with BatchNorm need training batches with more than one value per channel.

---

## Use cases and results

### DaT-SPECT-like phantoms: binary, multi-site

`examples/datscan-phantoms/` imitates the structure of the DaT-SPECT study at CPU scale:
- 400 volumes (48 × 48 × 40, 4 mm) from 3 sites, each with its own resolution, count level and noise; 219 pathologic and 181 normal.
- Pathologic subjects lose putaminal uptake, posterior first and asymmetrically, with severity from subtle to marked.
- Two inputs come from the same scan, both normalised to an occipital reference mask: the whole brain resampled to 8 mm (24 × 24 × 20), and a 24 × 16 × 12 bilateral striatal ROI found by `crop: peak`.
- Six scenarios (S1–S6: whole / ROI / whole+ROI, without and with augmentation).
- Settings: 5-fold stratified CV, early stopping on an inner 15 % split, up to 30 epochs.
- Backbones are CPU-sized: Simple3D at width 16, and no pretrained weights.

The grid is not run in full: 14 cells, with Simple3D on all six scenarios and the larger backbones on the ROI scenarios.

```bash
make phantoms benchmark calibrate bundle
```

**Leaderboard** (out-of-fold; calibration cross-fitted, best method per run; `results/calibrated.md` has every column):

| Run | Inputs | Aug. | Log loss | Calibration | Calibrated log loss | Calibrated ECE | AUROC | Train (s) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| resnet3d (3D ResNet-18) | roi | yes | 0.2553 | temperature | **0.2244** | 0.035 | 0.968 | 1653 |
| simple3d | whole+roi | yes | 0.2802 | temperature | 0.2718 | 0.034 | 0.954 | 183 |
| simple3d | whole+roi | no | 0.3057 | temperature | 0.2864 | 0.051 | 0.950 | 127 |
| simple3d | whole | yes | 0.2962 | temperature | 0.2918 | 0.035 | 0.958 | 208 |
| resnet3d | roi | no | 0.3203 | platt | 0.2996 | 0.063 | 0.948 | 996 |
| simple3d | whole | no | 0.3418 | temperature | 0.3171 | 0.052 | 0.941 | 103 |
| r3d18 | roi | yes | 0.3951 | temperature | 0.3522 | 0.065 | 0.936 | 1067 |
| slice2p5d | whole | yes | 0.3567 | none | 0.3567 | 0.050 | 0.933 | 799 |
| simple3d | roi | no | 0.4371 | temperature | 0.3951 | 0.060 | 0.909 | 49 |
| simple3d | roi | yes | 0.4324 | temperature | 0.3991 | 0.067 | 0.909 | 70 |
| slice2p5d | roi | no | 0.4824 | temperature | 0.4531 | 0.074 | 0.858 | 354 |
| slice2p5d | roi | yes | 0.4738 | none | 0.4738 | 0.062 | 0.864 | 277 |
| r3d18 | roi | no | 0.5867 | temperature | 0.5344 | 0.107 | 0.835 | 812 |
| swin3d_t | roi | no | 0.6455 | none | 0.6455 | 0.076 | 0.653 | 508 |

**Ensemble** (greedy selection over the 12 best calibrated runs, equal-weight logit mean, final temperature):

| Estimate | Log loss | AUROC | Brier | ECE |
| --- | --- | --- | --- | --- |
| Best single run (resnet3d, ROI + aug., temperature) | 0.2244 | 0.968 | — | 0.035 |
| Ensemble, selected and scored on the same OOF predictions | 0.2005 | 0.975 | 0.056 | 0.023 |
| Ensemble, **nested** (selection inside 5 outer folds) | **0.2397** | 0.970 | 0.059 | 0.028 |

The ResNet ROI run was chosen in every outer fold. The second member varied: Simple3D whole+ROI in 60 % of folds, Simple3D whole and others in 20–40 %. The frozen bundle (2 members × 5 fold checkpoints, plus calibrators, the occipital mask and SHA-256 hashes) reproduced every training-time OOF logit exactly: maximum difference 0.0 over 400 subjects (`results/verify.json`).

**What this run shows**, on these phantoms:
- **Calibration is not optional.** Uncalibrated, the best run had log loss 0.255. Temperature scaling, cross-fitted, brought it to 0.224 and cut ECE from 0.046 to 0.035. Isotonic regression was never selected; with 320 subjects per fit, it overfits.
- **Augmentation helped most 3D runs.** Calibrated log loss: ResNet ROI 0.300 → 0.224, R3D-18 ROI 0.534 → 0.352, Simple3D fused 0.286 → 0.272. It did not help the 2.5D model or Simple3D on the ROI.
- **The best input depends on the model.** Simple3D did better on the whole volume and the fused inputs than on the ROI alone. ResNet-18 did best on the ROI.
- **Swin3D-Tiny from scratch is close to chance at this size** (AUROC 0.65). This is expected for a transformer trained on 400 small volumes.
- **Ensemble gains can be selection artefacts.** The naive ensemble estimate (0.2005) beats the best single run. The nested one (0.2397) does not, so here the honest conclusion is "no reliable ensemble gain". The best single run's 0.2244 is also the best of 14 and therefore optimistic. Only a held-out cohort would settle it.

### MedMNIST3D (bring your own data)

`examples/medmnist3d/` converts a MedMNIST3D `.npz` (for example OrganMNIST3D, 11 classes) into a manifest of `.npy` volumes and runs a multi-class benchmark with temperature calibration:

```bash
python examples/medmnist3d/prepare.py organmnist3d.npz --out examples/medmnist3d/data
calivox benchmark examples/medmnist3d/config.yaml
```

The converter and config are covered by the tests on a fake `.npz`. No MedMNIST results are reported here.

### DaT-SPECT (real study)

[parkinson-datscan-classifier](https://github.com/mnaeimaei/parkinson-datscan-classifier) expresses all 60 experiments, the 240 calibration configurations and its final ensemble as CaliVox configs. The original research pipeline for preprocessing and registration is kept alongside them. Its reported results are in that repository.

---

## Project structure

```text
calivox/
├── calivox/
│   ├── config.py          YAML schema (Pydantic), scenarios, per-model overrides, plugins
│   ├── volumes.py         NIfTI/NumPy I/O, preprocessing step registry and built-in steps
│   ├── data.py            manifest, cached preprocessing, dataset, fold assignment
│   ├── augment.py         3D / in-plane affine, intensity, noise, smoothing
│   ├── models/            backbones + multi-input late-fusion classifier
│   ├── training.py        cross-validation, early stopping, OOF predictions
│   ├── metrics.py         log loss, AUROC, AUPRC, Brier, ECE, ...
│   ├── calibration.py     temperature, Platt, isotonic; cross-fitting
│   ├── ensemble.py        greedy / fixed ensembles, nested estimate
│   ├── bundle.py          freeze, predict, verify, integrity check
│   ├── report.py          leaderboards
│   ├── synthetic.py       DaT-SPECT-like phantom generator
│   └── cli.py             `calivox` command
├── examples/
│   ├── datscan-phantoms/  config + results/ (leaderboards, ensemble, verify report)
│   └── medmnist3d/        converter + config
├── tests/                 38 tests, offline, about 20 s on CPU
├── Dockerfile  Makefile  pyproject.toml  .github/workflows/ci.yml
└── README.md
```

## Tests

```bash
make test
```

The tests run offline on tiny phantoms. They cover:
- every preprocessing step, the cache keying, and fold assignment (stratified, grouped, frozen);
- in-plane augmentation, all backbones (single, fused, multi-class), the grey-stem conversion and the MedicalNet loader;
- calibration recovering a known over-confidence, and cross-fitting being out-of-sample;
- CV with OOF coverage, multi-class runs, greedy and fixed ensembles;
- the bundle: freeze, verify, predict and tamper detection;
- plugins and the CLI.

CI also runs an end-to-end smoke pipeline.

## Docker

```bash
docker build -t calivox .
docker run --rm -v "$PWD/examples:/app/examples" calivox benchmark examples/datscan-phantoms/config.yaml
```

## Limitations

- **The phantoms are not physics.** They reproduce the structure of the DaT-SPECT problem (striatal uptake loss, site effects, positioning noise), not real SPECT acquisition or reconstruction. Results on them show the framework working; they are not evidence about real scans.
- **CPU-sized phantom benchmark.** It uses short training on small volumes and some backbones in reduced form. Rankings between architectures there should not be read as general conclusions.
- **Preprocessing is generic.** Registration to a template or an atlas is not included. Bring registered volumes, or add a step plugin.
- **Calibration is post-hoc only.** Temperature, Platt and isotonic scaling (binary), and temperature scaling (multi-class). No calibration-aware training losses yet.
- **Evaluation is cross-validated.** External validation on an independent cohort is still needed before any clinical claim.

## License

MIT; see [LICENSE](LICENSE). Pretrained weights (Kinetics-400, ImageNet, MedicalNet) are subject to their own licenses.
