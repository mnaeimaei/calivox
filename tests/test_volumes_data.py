import numpy as np
import pandas as pd
import pytest
import torch

from calivox.augment import Augmenter, gaussian_blur3d
from calivox.config import AugmentConfig, ExperimentConfig
from calivox.data import assign_folds, load_manifest, preprocess_inputs, to_dhw
from calivox.volumes import Volume, apply_steps, load_volume, save_volume

from .conftest import ROOT, write_config


def vol(shape=(20, 20, 16), spacing=2.0) -> Volume:
    a = np.random.default_rng(0).random(shape).astype(np.float32)
    return Volume(a, np.diag([spacing, spacing, spacing, 1.0]))


# ---- io and steps -----------------------------------------------------------------

def test_nifti_and_npz_roundtrip(tmp_path):
    v = vol()
    for name in ("a.nii.gz", "b.npz"):
        out = load_volume(save_volume(v, tmp_path / name))
        assert np.allclose(out.array, v.array) and np.allclose(out.spacing, 2.0 if "nii" in name else out.spacing)


def test_resample_changes_shape_and_spacing():
    out = apply_steps(vol(), [{"op": "resample", "spacing": [4, 4, 4]}])
    assert out.array.shape == (10, 10, 8) and np.allclose(out.spacing, 4.0)


def test_reorient_flipped_axis_to_ras():
    v = vol()
    v.affine[0, 0] = -2.0  # L instead of R
    out = apply_steps(v, [{"op": "reorient"}])
    assert out.affine[0, 0] > 0 and np.allclose(out.array, v.array[::-1])


def test_normalize_methods(tmp_path):
    v = vol()
    z = apply_steps(v, [{"op": "normalize", "method": "zscore"}]).array
    assert abs(z.mean()) < 1e-5 and abs(z.std() - 1) < 1e-4
    mm = apply_steps(v, [{"op": "normalize", "method": "minmax"}]).array
    assert mm.min() == 0 and abs(mm.max() - 1) < 1e-6
    mask = np.zeros(v.array.shape, np.uint8)
    mask[:5] = 1
    save_volume(Volume(mask.astype(np.float32), v.affine), tmp_path / "m.nii.gz")
    ref = apply_steps(v, [{"op": "normalize", "method": "reference_mask", "mask": str(tmp_path / "m.nii.gz")}]).array
    assert abs(ref[:5].mean() - 1.0) < 1e-5  # the reference region averages to 1


def test_reference_mask_errors(tmp_path):
    v = vol()
    save_volume(Volume(np.zeros((3, 3, 3), np.float32), v.affine), tmp_path / "bad.nii.gz")
    with pytest.raises(ValueError, match="shape"):
        apply_steps(v, [{"op": "normalize", "method": "reference_mask", "mask": str(tmp_path / "bad.nii.gz")}])
    with pytest.raises(ValueError, match="mask"):
        apply_steps(v, [{"op": "normalize", "method": "reference_mask"}])


def test_crop_modes():
    a = np.zeros((30, 30, 30), np.float32)
    a[20:24, 5:9, 10:14] = 10.0                     # a bright blob off-center
    v = Volume(a, np.eye(4))
    c = apply_steps(v, [{"op": "crop", "mode": "center", "shape": [10, 10, 10]}]).array
    assert c.shape == (10, 10, 10) and c.sum() == 0
    p = apply_steps(v, [{"op": "crop", "mode": "peak", "shape": [8, 8, 8], "top_fraction": 0.005}]).array
    assert p.sum() == a.sum()                        # the peak crop captures the whole blob
    big = apply_steps(v, [{"op": "crop", "mode": "box_center", "center": [0, 0, 0], "shape": [6, 6, 6]}]).array
    assert big.shape == (6, 6, 6)                    # zero-padded outside the image


def test_unknown_op():
    with pytest.raises(ValueError, match="available"):
        apply_steps(vol(), [{"op": "nope"}])


def test_to_dhw_axes():
    a = np.zeros((2, 3, 4))
    assert to_dhw(a, "z").shape == (4, 3, 2) and to_dhw(a, "x").shape == (2, 3, 4)


# ---- augmentation -----------------------------------------------------------------

def test_augmenter_shape_seed_and_identity():
    x = torch.rand(1, 12, 16, 16)
    a1, a2 = Augmenter(AugmentConfig(), seed=1), Augmenter(AugmentConfig(), seed=1)
    out = a1(x)
    assert out.shape == x.shape and torch.allclose(out, a2(x))
    off = AugmentConfig(spatial_probability=0, intensity_probability=0, noise_probability=0)
    assert torch.equal(Augmenter(off, seed=0)(x), x)
    assert gaussian_blur3d(x, 1.0).std() < x.std()


# ---- manifest, preprocessing, folds -----------------------------------------------

def test_manifest_and_positive_class(phantoms, tmp_path):
    cfg = write_config(tmp_path, phantoms)
    man = load_manifest(cfg)
    assert man.classes == ["normal", "pathologic"] and man.positive_index == 1
    bad = pd.read_csv(phantoms / "manifest.csv")
    bad = pd.concat([bad, bad.head(1)])
    bad.to_csv(tmp_path / "dup.csv", index=False)
    with pytest.raises(ValueError, match="duplicate"):
        load_manifest(cfg, tmp_path / "dup.csv")


def test_preprocess_cache_is_keyed_by_steps(phantoms, tmp_path):
    cfg = write_config(tmp_path, phantoms)
    man = load_manifest(cfg)
    p1 = preprocess_inputs(cfg, man)
    assert len(p1["roi"]) == 40 and all(p.exists() for p in p1["roi"])
    assert np.load(p1["roi"][0]).shape == (24, 16, 12) and np.load(p1["whole"][0]).shape == (24, 24, 20)
    cfg2 = write_config(tmp_path / "b" if (tmp_path / "b").mkdir() is None else tmp_path, phantoms,
                        **{"cache_dir": str(tmp_path / "cache")})
    cfg2.preprocess["roi"][1]["shape"] = [20, 16, 12]
    p2 = preprocess_inputs(cfg2, man)
    assert p2["roi"][0].parent != p1["roi"][0].parent and p2["whole"][0] == p1["whole"][0]


def test_folds_stratified_group_and_frozen(phantoms, tmp_path):
    cfg = write_config(tmp_path, phantoms)
    man = load_manifest(cfg)
    f = assign_folds(cfg, man)
    assert sorted(np.unique(f)) == [0, 1, 2]
    g = assign_folds(cfg.override(**{"cv.by_group": True}), man)
    sites = man.df["site"].to_numpy()
    assert all(len(np.unique(g[sites == s])) == 1 for s in np.unique(sites))   # a site stays in one fold
    pd.DataFrame({"subject_id": man.ids, "fold": np.arange(40) % 4}).to_csv(tmp_path / "f.csv", index=False)
    fr = assign_folds(cfg.override(**{"cv.folds_file": str(tmp_path / "f.csv")}), man)
    assert (fr == np.arange(40) % 4).all()


def test_config_validation(phantoms, tmp_path):
    with pytest.raises(ValueError, match="unknown inputs"):
        write_config(tmp_path, phantoms, scenarios={"X": {"inputs": ["pet"]}})
    cfg = ExperimentConfig.from_yaml(ROOT / "examples" / "datscan-phantoms" / "config.yaml")
    assert set(cfg.scenarios) == {"S1", "S2", "S3", "S4", "S5", "S6"}
    assert cfg.for_run("r3d18", "S5").run_name == "r3d18__roi__aug"


def test_inplane_augmentation_never_mixes_slices():
    x = torch.zeros(1, 6, 16, 16)
    x[:, 3] = 1.0                                   # one bright depth slice
    cfg = AugmentConfig(spatial_probability=1, intensity_probability=0, noise_probability=0,
                        rotation_degrees=10, spatial_mode="inplane_2d")
    out = Augmenter(cfg, seed=0)(x)
    assert out[:, [0, 1, 2, 4, 5]].abs().max() == 0 and out[:, 3].sum() > 0


def test_per_model_batch_size_and_augment(phantoms, tmp_path):
    cfg = write_config(tmp_path, phantoms, models={
        "s": {"name": "simple3d", "batch_size": {"roi": 8, "whole+roi": 1},
              "augment": {"spatial_mode": "inplane_2d"}}})
    assert cfg.for_run("s", "S2").training.batch_size == 8
    assert cfg.for_run("s", "S3").training.batch_size == 1
    assert cfg.for_run("s", "S2").augment.spatial_mode == "inplane_2d"


def test_plugin_step(phantoms, tmp_path):
    (tmp_path / "myplug.py").write_text(
        "from calivox.volumes import register_step\n"
        "@register_step('double_test')\n"
        "def double(vol, params, row, resolve):\n"
        "    vol.array = vol.array * 2\n"
        "    return vol\n")
    cfg = write_config(tmp_path, phantoms, plugins=["myplug.py"])
    assert cfg.plugins == ["myplug.py"]
    from calivox.volumes import STEPS
    try:
        assert apply_steps(vol(), [{"op": "double_test"}]).array.max() > 1.0
    finally:
        STEPS.pop("double_test")
    with pytest.raises(ValueError, match="not found"):
        write_config(tmp_path, phantoms, plugins=["nope.py"])


def test_folds_file_with_fold_names(phantoms, tmp_path):
    cfg = write_config(tmp_path, phantoms)
    man = load_manifest(cfg)
    pd.DataFrame({"subject_id": man.ids, "cv_fold": [f"fold_{i % 5}" for i in range(40)]}).to_csv(tmp_path / "f.csv", index=False)
    f = assign_folds(cfg.override(**{"cv.folds_file": str(tmp_path / "f.csv")}), man)
    assert (f == np.arange(40) % 5).all()
