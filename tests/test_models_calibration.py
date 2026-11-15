import numpy as np
import pytest
import torch

from calivox.calibration import cross_fit, evaluate_calibration, fit_calibrator
from calivox.metrics import binary_log_loss, binary_metrics, ece, multiclass_metrics, sigmoid
from calivox.models import BACKBONES, build_model, register_backbone

SMALL = {"simple3d": {"width": 4}, "slice2p5d": {"slices": 3}}


@pytest.mark.parametrize("name", ["simple3d", "resnet3d", "r3d18", "swin3d_t", "slice2p5d"])
def test_backbones_single_fused_multiclass(name):
    torch.set_num_threads(1)
    x = {"roi": torch.randn(2, 1, 12, 16, 24), "whole": torch.randn(2, 1, 20, 24, 24)}
    assert build_model(name, ["roi"], 1, params=SMALL.get(name))(x).shape == (2,)
    assert build_model(name, ["whole", "roi"], 3, params=SMALL.get(name))(x).shape == (2, 3)


def test_densenet_needs_32_voxels():
    m = build_model("densenet3d", ["roi"], 1)
    assert m({"roi": torch.randn(2, 1, 32, 32, 32)}).shape == (2,)
    with pytest.raises(ValueError, match="32"):
        m({"roi": torch.randn(2, 1, 12, 16, 24)})


def test_gray_stem_keeps_pretrained_response():
    """Summing RGB kernels gives the same response to a grey image as the RGB stem."""
    from torch import nn

    from calivox.models.backbones import _gray_stem

    rgb = nn.Conv3d(3, 4, 3, padding=1)
    gray = _gray_stem(rgb, nn.Conv3d)
    x = torch.rand(1, 1, 5, 5, 5)
    assert torch.allclose(gray(x), rgb(x.repeat(1, 3, 1, 1, 1)), atol=1e-5)


def test_medicalnet_loader(tmp_path):
    from monai.networks.nets import resnet18

    ref = resnet18(spatial_dims=3, n_input_channels=1, feed_forward=False)
    state = {f"module.{k}": v for k, v in ref.state_dict().items()}
    state["module.conv_seg.weight"] = torch.zeros(2)              # segmentation head, not loaded
    torch.save({"state_dict": state}, tmp_path / "resnet_18_23dataset.pth")
    m = build_model("resnet3d", ["roi"], 1, pretrained=str(tmp_path / "resnet_18_23dataset.pth"))
    loaded = m.backbones["roi"].net.state_dict()
    assert torch.equal(loaded["conv1.weight"], ref.state_dict()["conv1.weight"])
    with pytest.raises(FileNotFoundError):
        build_model("resnet3d", ["roi"], 1, pretrained=str(tmp_path / "missing.pth"))


def test_custom_backbone():
    from torch import nn

    @register_backbone("meanpool_test")
    def meanpool(pretrained=None, **_):
        """mean intensity"""
        class M(nn.Module):
            n_features = 1

            def forward(self, x):
                return x.mean(dim=(1, 2, 3, 4)).unsqueeze(1)
        return M()

    try:
        assert build_model("meanpool_test", ["roi"], 1)({"roi": torch.ones(2, 1, 4, 4, 4)}).shape == (2,)
    finally:
        BACKBONES.pop("meanpool_test")


# ---- metrics and calibration ------------------------------------------------------

def _overconfident(n=2000, seed=0, T=2.5):
    rng = np.random.default_rng(seed)
    z_true = rng.normal(0, 1.5, n)
    y = (rng.random(n) < sigmoid(z_true)).astype(int)
    return z_true * T, y          # logits T times too large


def test_metrics_basics():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.6, 0.9])
    m = binary_metrics(y, p)
    assert m["auroc"] == 1.0 and m["confusion_matrix"] == [[2, 0], [0, 2]]
    assert abs(m["log_loss"] - binary_log_loss(y, p)) < 1e-12
    assert ece(np.array([1, 0]), np.array([1.0, 0.0])) == 0.0
    mc = multiclass_metrics(np.array([0, 1, 2]), np.eye(3))
    assert mc["accuracy"] == 1.0 and mc["log_loss"] < 1e-6


def test_temperature_recovers_overconfidence():
    z, y = _overconfident(T=2.5)
    T = fit_calibrator("temperature", z, y).params["T"]
    assert 2.0 < T < 3.0
    rep = evaluate_calibration(z, y, ["none", "temperature", "platt", "isotonic"], folds=5, seed=0)
    assert rep["methods"]["temperature"]["log_loss"] < rep["methods"]["none"]["log_loss"]
    assert rep["selected"] != "none"


def test_cross_fit_is_out_of_sample():
    z, y = _overconfident(n=300, seed=1)
    full = fit_calibrator("isotonic", z, y).apply(z)
    crossed = cross_fit("isotonic", z, y, folds=5, seed=0)
    assert binary_log_loss(y, sigmoid(full)) < binary_log_loss(y, sigmoid(crossed))  # in-sample looks better


def test_multiclass_temperature_only():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(200, 3)) * 3
    y = rng.integers(0, 3, 200)
    assert fit_calibrator("temperature", Z, y).apply(Z).shape == Z.shape
    with pytest.raises(ValueError, match="binary"):
        fit_calibrator("platt", Z, y)
    assert set(evaluate_calibration(Z, y, ["none", "temperature", "platt"], 3)["methods"]) == {"none", "temperature"}
