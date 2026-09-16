"""tests/test_integration_e2e.py — one smoke test stringing all six components
into a single (synthetic) pipeline: A-PhysDeg sampling -> a few real gradient
steps through a model containing S-WFA -> Robustness Cube evaluation on the
ID protocol -> temperature scaling + selective prediction -> ONNX export,
verification, and latency benchmarking.

Every other test file in this repo exercises one component against a stub
(a synthetic sample list, a hand-built predict_fn, hand-crafted logits). This
file exists to catch the failure mode none of those can: two components whose
own unit tests both pass, but whose interfaces don't actually fit together.
Still entirely synthetic data, still CPU, still no trained checkpoint — this
proves the six components compose, not that the resulting model is good.
"""
import random

import pytest
import torch
import torch.nn.functional as F
from torchvision import transforms as T

from augmentation.a_physdeg import AdaptivePhysDegController, APhysDegTransform
from calibration.selective import SelectivePredictor
from degradation.params import ID_TYPES, N_SEVERITIES
from deploy.benchmark import benchmark_latency, model_size_mb
from deploy.export import export_to_onnx
from deploy.toy_model import ToyDefectClassifier
from deploy.verify import compare_pytorch_onnx
from eval.bootstrap import group_bootstrap_ci
from eval.cube_protocol import run_cube_protocol
from eval.metrics import macro_f1
from tests._synthetic import make_synthetic_samples

IMG_SIZE = (32, 32)
CLASS_NAMES = ["low", "high"]
BASE_TRANSFORM = T.Compose([T.Resize(IMG_SIZE), T.PILToTensor(), lambda t: t.float() / 255.0])


def _jsd(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Symmetric KL between two batches of class distributions (the
    consistency term PhysDeg's design pairs with CE(clean))."""
    m = 0.5 * (p + q)
    kl = lambda a, b: (a * (torch.log(a + eps) - torch.log(b + eps))).sum(dim=1)
    return 0.5 * (kl(p, m) + kl(q, m)).mean()


def _predict_fn(model):
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


def test_full_six_component_pipeline_runs_end_to_end(tmp_path):
    torch.manual_seed(0)

    samples = make_synthetic_samples(n_per_class=10, n_classes=2, size=IMG_SIZE)
    class0, class1 = samples[:10], samples[10:]
    train_samples = class0[:6] + class1[:6]
    calib_samples = class0[6:8] + class1[6:8]
    eval_samples = class0[8:] + class1[8:]

    model = ToyDefectClassifier(num_classes=2, in_channels=1, stem_channels=8, wfa_channels=16)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    # ---- 1 + 3: A-PhysDeg-driven training ----
    controller = AdaptivePhysDegController()
    a_phys_transform = APhysDegTransform(BASE_TRANSFORM, controller, rng=random.Random(0))
    assert controller.curriculum.unlocked_severities(0.0) == [0]
    assert set(controller.curriculum.unlocked_severities(1.0)) == {0, 1, 2}

    def difficulty_fn(images, labels):
        model.eval()
        with torch.no_grad():
            loss = F.cross_entropy(model(images), labels)
        return float(torch.tanh(loss / 2.0))

    n_epochs = 3
    for epoch in range(n_epochs):
        controller.set_progress(epoch / (n_epochs - 1))
        controller.run_diagnostics(train_samples, BASE_TRANSFORM, difficulty_fn,
                                    batch_size=4, rng=random.Random(epoch), split="train")
        model.train()
        for img, label in train_samples:
            triplet = a_phys_transform(img)  # (3, C, H, W): clean, deg1, deg2
            label_t = torch.tensor([label])
            ce = F.cross_entropy(model(triplet[0:1]), label_t)
            jsd = _jsd(F.softmax(model(triplet[1:2]), dim=1), F.softmax(model(triplet[2:3]), dim=1))
            loss = ce + 0.1 * jsd
            assert torch.isfinite(loss), "loss diverged during the smoke-test training loop"
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # A gradient genuinely reached S-WFA's sample-gate MLP and cross-band gate,
    # not just the classification head sitting on top of a frozen wavelet path.
    assert model.wfa.gate.mlp[0].weight.grad is not None
    assert model.wfa.core.cross_gate.weight.grad is not None
    difficulty_values = controller.difficulty.as_dict()
    assert any(v != 0.5 for v in difficulty_values.values()), "difficulty EMA never moved off its init value"

    # ---- 4: Robustness Cube over the ID protocol, with a real (trained-a-little) model ----
    model.eval()
    cube = run_cube_protocol(eval_samples, BASE_TRANSFORM, _predict_fn(model), regime="id",
                              class_names=CLASS_NAMES, batch_size=4, min_samples_per_class=1)
    assert len(cube._conditions) == 1 + len(ID_TYPES) * N_SEVERITIES
    mpc = cube.overall_mPC()
    assert 0.0 <= mpc <= 1.0
    worst = cube.worst_slices(k=3, min_samples=1)
    assert len(worst) > 0
    auc = cube.severity_auc("jpeg")
    assert 0.0 <= auc <= 1.0

    clean_records = cube._conditions[(None, None)].records
    ci = group_bootstrap_ci(
        clean_records,
        lambda recs: macro_f1([r.pred_label for r in recs], [r.true_label for r in recs], len(CLASS_NAMES)),
        n_boot=200, rng=random.Random(0),
    )
    assert ci.lower <= ci.point + 1e-9 and ci.point - 1e-9 <= ci.upper

    # ---- 5: calibration + selective prediction, fit on a disjoint calibration split ----
    calib_imgs = torch.stack([BASE_TRANSFORM(img) for img, _ in calib_samples])
    calib_labels = torch.tensor([lbl for _, lbl in calib_samples])
    with torch.no_grad():
        calib_logits = model(calib_imgs)
    predictor = SelectivePredictor.fit(calib_logits, calib_labels, target_coverage=0.7, split="calibration")

    eval_imgs = torch.stack([BASE_TRANSFORM(img) for img, _ in eval_samples])
    eval_labels = torch.tensor([lbl for _, lbl in eval_samples])
    with torch.no_grad():
        eval_logits = model(eval_imgs)
    coverage, risk = predictor.coverage_and_risk(eval_logits, eval_labels)
    assert 0.0 <= coverage <= 1.0
    assert risk is None or 0.0 <= risk <= 1.0

    # ---- 6: ONNX export, PyTorch-vs-ONNX agreement, and latency benchmarking ----
    onnx_path = str(tmp_path / "model.onnx")
    export_to_onnx(model, torch.randn(2, 1, *IMG_SIZE), onnx_path)
    comparison = compare_pytorch_onnx(model, onnx_path, eval_imgs)
    assert comparison.max_abs_diff < 1e-3
    assert comparison.argmax_agreement_rate == pytest.approx(1.0)

    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    single_input = eval_imgs[:1].numpy()
    stats = benchmark_latency(lambda: session.run(None, {input_name: single_input}), n_warmup=2, n_iters=10)
    assert stats.p50_ms >= 0.0
    assert stats.throughput_qps > 0.0
    assert model_size_mb(onnx_path) > 0.0
