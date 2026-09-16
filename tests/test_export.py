import os

import onnxruntime as ort
import torch

from deploy.export import export_to_onnx
from deploy.toy_model import ToyDefectClassifier


def test_export_produces_a_loadable_onnx_file(tmp_path):
    torch.manual_seed(0)
    model = ToyDefectClassifier(num_classes=5)
    dummy = torch.randn(2, 3, 32, 32)
    path = str(tmp_path / "model.onnx")

    export_to_onnx(model, dummy, path)

    assert os.path.exists(path)
    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    assert session.get_inputs()[0].name == "input"
    assert session.get_outputs()[0].name == "output"


def test_dynamic_batch_axis_accepts_a_different_batch_size_at_inference(tmp_path):
    """This is the whole point of dynamic_batch=True — export once at one batch
    size, run at another."""
    torch.manual_seed(0)
    model = ToyDefectClassifier(num_classes=4)
    dummy = torch.randn(3, 3, 32, 32)
    path = str(tmp_path / "model.onnx")
    export_to_onnx(model, dummy, path, dynamic_batch=True)

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    for batch in (1, 5):
        x = torch.randn(batch, 3, 32, 32).numpy()
        out = session.run(None, {"input": x})[0]
        assert out.shape == (batch, 4)


def test_static_batch_export_still_produces_a_working_session(tmp_path):
    torch.manual_seed(0)
    model = ToyDefectClassifier(num_classes=4)
    dummy = torch.randn(2, 3, 32, 32)
    path = str(tmp_path / "model_static.onnx")
    export_to_onnx(model, dummy, path, dynamic_batch=False)

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    out = session.run(None, {"input": dummy.numpy()})[0]
    assert out.shape == (2, 4)
