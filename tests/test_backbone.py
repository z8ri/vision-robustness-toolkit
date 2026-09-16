import pytest
import torch

from models.backbone import STAGE3_CHANNELS, ConvNeXtTinySWFA


def test_forward_shape_at_reduced_resolution():
    # img_size=64 keeps this test fast; test_forward_at_real_production_resolution
    # below separately confirms the actual 224 config works.
    torch.manual_seed(0)
    model = ConvNeXtTinySWFA(num_classes=5)
    x = torch.randn(2, 3, 64, 64)
    out = model(x)
    assert out.shape == (2, 5)


def test_wfa_modules_are_inserted_at_the_documented_positions():
    model = ConvNeXtTinySWFA(num_classes=3, wfa_positions=(4, 8))
    assert set(model.wfa_modules.keys()) == {"4", "8"}
    for key in model.wfa_modules:
        assert model.wfa_modules[key].gamma.shape == (STAGE3_CHANNELS,)


def test_custom_wfa_positions_are_respected():
    model = ConvNeXtTinySWFA(num_classes=3, wfa_positions=(0,))
    assert set(model.wfa_modules.keys()) == {"0"}


def test_out_of_range_wfa_position_raises():
    with pytest.raises(ValueError):
        ConvNeXtTinySWFA(num_classes=3, wfa_positions=(9,))  # Stage 3 only has blocks 0..8


def test_gradient_reaches_the_swfa_gate_not_just_the_head():
    torch.manual_seed(0)
    model = ConvNeXtTinySWFA(num_classes=3)
    x = torch.randn(1, 3, 64, 64)
    model(x).sum().backward()
    assert model.wfa_modules["4"].gate.mlp[0].weight.grad is not None
    assert model.wfa_modules["8"].gate.mlp[0].weight.grad is not None


def test_forward_at_real_production_resolution():
    """The actual configuration (224x224, positions [4, 8], reduction=32,
    hf_kernel=3) — slower on CPU, so kept to one test."""
    torch.manual_seed(0)
    model = ConvNeXtTinySWFA(num_classes=5, wfa_positions=(4, 8), wfa_reduction=32, wfa_hf_kernel=3)
    x = torch.randn(1, 3, 224, 224)
    out = model(x)
    assert out.shape == (1, 5)
    assert torch.isfinite(out).all()
