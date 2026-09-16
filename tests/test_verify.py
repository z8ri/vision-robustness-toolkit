import pytest
import torch

from deploy.export import export_to_onnx
from deploy.toy_model import ToyDefectClassifier
from deploy.verify import check_no_fallback, compare_pytorch_onnx


def _export_toy(tmp_path, num_classes=5, seed=0):
    torch.manual_seed(seed)
    model = ToyDefectClassifier(num_classes=num_classes).eval()
    dummy = torch.randn(2, 3, 32, 32)
    path = str(tmp_path / "model.onnx")
    export_to_onnx(model, dummy, path)
    return model, path


# ---- compare_pytorch_onnx ----

def test_pytorch_and_onnx_outputs_match_closely(tmp_path):
    model, path = _export_toy(tmp_path)
    inputs = torch.randn(6, 3, 32, 32)
    result = compare_pytorch_onnx(model, path, inputs)
    assert result.n_samples == 6
    assert result.max_abs_diff < 1e-4  # float32 export/inference precision
    assert result.argmax_agreement_rate == pytest.approx(1.0)


def test_shape_mismatch_between_model_and_exported_graph_raises(tmp_path):
    _model_a, path = _export_toy(tmp_path, num_classes=5)
    model_b = ToyDefectClassifier(num_classes=4).eval()  # different output width
    inputs = torch.randn(2, 3, 32, 32)
    with pytest.raises(ValueError):
        compare_pytorch_onnx(model_b, path, inputs)


# ---- check_no_fallback ----

def test_cpu_provider_reports_no_fallback(tmp_path):
    """The only scenario testable on this machine (no GPU): requesting
    CPUExecutionProvider and confirming every executed kernel actually ran on
    it. The interesting real-world case — CUDA silently falling back to CPU for
    an unsupported op — needs a CUDA-capable machine to exercise; this test
    proves the reporting mechanism is accurate, not that a fallback was caught."""
    _model, path = _export_toy(tmp_path)
    inputs = torch.randn(2, 3, 32, 32)
    report = check_no_fallback(path, inputs, providers=["CPUExecutionProvider"])
    assert report.requested_providers == ("CPUExecutionProvider",)
    assert report.no_fallback is True
    assert report.used_providers.get("CPUExecutionProvider", 0) > 0
    assert set(report.used_providers.keys()) == {"CPUExecutionProvider"}


def test_check_no_fallback_rejects_empty_providers(tmp_path):
    _model, path = _export_toy(tmp_path)
    inputs = torch.randn(2, 3, 32, 32)
    with pytest.raises(ValueError):
        check_no_fallback(path, inputs, providers=[])
