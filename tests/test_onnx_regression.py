"""tests/test_onnx_regression.py — the "clean/ID/OOD regression" check that
test_verify.py doesn't cover.

test_verify.py's compare_pytorch_onnx tests confirm PyTorch and ONNX Runtime
agree on *random input tensors*. They don't confirm the exported model still
produces the same *robustness-evaluation numbers* once it's driven through the
actual degradation protocol (Component 1) and Robustness Cube (Component 4) —
the source design doc's ONNX minimal-closed-loop checklist explicitly asks for
this as a separate check, since a numerically-close-on-random-input model could
still diverge on a specific corruption's input distribution. This file runs the
*same* ID and OOD robustness protocol through both backends and compares the
resulting per-condition macro-F1 numbers directly.
"""
import onnxruntime as ort
import pytest
import torch
from torchvision import transforms as T

from degradation.params import ID_TYPES, N_SEVERITIES, OOD_TYPES
from deploy.export import export_to_onnx
from deploy.toy_model import ToyDefectClassifier
from eval.cube_protocol import run_cube_protocol
from tests._synthetic import make_synthetic_samples

IMG_SIZE = (32, 32)
CLASS_NAMES = ["low", "high"]
TRANSFORM = T.Compose([T.Resize(IMG_SIZE), T.PILToTensor(), lambda t: t.float() / 255.0])


def _pytorch_predict_fn(model):
    model.eval()

    def _fn(loader):
        preds, confs, labels = [], [], []
        with torch.no_grad():
            for imgs, lbls in loader:
                probs = torch.softmax(model(imgs), dim=1)
                conf, pred = probs.max(dim=1)
                preds.extend(int(p) for p in pred.tolist())
                confs.extend(float(c) for c in conf.tolist())
                labels.extend(int(l) for l in lbls.tolist())
        return preds, confs, labels

    return _fn


def _onnx_predict_fn(onnx_path):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    def _fn(loader):
        preds, confs, labels = [], [], []
        for imgs, lbls in loader:
            logits = torch.from_numpy(session.run(None, {input_name: imgs.numpy()})[0])
            probs = torch.softmax(logits, dim=1)
            conf, pred = probs.max(dim=1)
            preds.extend(int(p) for p in pred.tolist())
            confs.extend(float(c) for c in conf.tolist())
            labels.extend(int(l) for l in lbls.tolist())
        return preds, confs, labels

    return _fn


def _assert_cubes_agree(pytorch_cube, onnx_cube, deg_types):
    assert pytorch_cube.overall_mPC() == pytest.approx(onnx_cube.overall_mPC(), abs=1e-4)
    assert pytorch_cube.condition_macro_f1(None, None) == pytest.approx(
        onnx_cube.condition_macro_f1(None, None), abs=1e-4)
    for deg_type in deg_types:
        for severity in range(N_SEVERITIES):
            pt_f1 = pytorch_cube.condition_macro_f1(deg_type, severity)
            onnx_f1 = onnx_cube.condition_macro_f1(deg_type, severity)
            assert pt_f1 == pytest.approx(onnx_f1, abs=1e-4), f"{deg_type}/s{severity + 1} diverged after export"


def test_onnx_and_pytorch_agree_across_the_full_id_robustness_protocol(tmp_path):
    torch.manual_seed(0)
    model = ToyDefectClassifier(num_classes=2, in_channels=1, stem_channels=8, wfa_channels=16).eval()
    samples = make_synthetic_samples(n_per_class=5, n_classes=2, size=IMG_SIZE)

    onnx_path = str(tmp_path / "model.onnx")
    export_to_onnx(model, torch.randn(2, 1, *IMG_SIZE), onnx_path)

    pytorch_cube = run_cube_protocol(samples, TRANSFORM, _pytorch_predict_fn(model), regime="id",
                                      class_names=CLASS_NAMES, batch_size=4, min_samples_per_class=1)
    onnx_cube = run_cube_protocol(samples, TRANSFORM, _onnx_predict_fn(onnx_path), regime="id",
                                   class_names=CLASS_NAMES, batch_size=4, min_samples_per_class=1)

    _assert_cubes_agree(pytorch_cube, onnx_cube, ID_TYPES)


def test_onnx_and_pytorch_agree_across_the_full_ood_robustness_protocol(tmp_path):
    torch.manual_seed(1)
    model = ToyDefectClassifier(num_classes=2, in_channels=1, stem_channels=8, wfa_channels=16).eval()
    samples = make_synthetic_samples(n_per_class=4, n_classes=2, size=IMG_SIZE)

    onnx_path = str(tmp_path / "model.onnx")
    export_to_onnx(model, torch.randn(2, 1, *IMG_SIZE), onnx_path)

    pytorch_cube = run_cube_protocol(samples, TRANSFORM, _pytorch_predict_fn(model), regime="ood",
                                      class_names=CLASS_NAMES, batch_size=4, min_samples_per_class=1)
    onnx_cube = run_cube_protocol(samples, TRANSFORM, _onnx_predict_fn(onnx_path), regime="ood",
                                   class_names=CLASS_NAMES, batch_size=4, min_samples_per_class=1)

    _assert_cubes_agree(pytorch_cube, onnx_cube, OOD_TYPES)
