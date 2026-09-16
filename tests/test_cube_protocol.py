import pytest
import torch
from torchvision import transforms as T

from degradation.params import ID_TYPES, N_SEVERITIES, OOD_TYPES
from eval.cube_protocol import run_cube_protocol
from tests._synthetic import make_synthetic_samples

TRANSFORM = T.Compose([T.Resize((16, 16)), T.PILToTensor(), lambda t: t.float() / 255.0])
CLASS_NAMES = ["low", "high"]


def _stub_predict_fn(loader):
    """A 'model' that thresholds mean brightness — same trick as component 1's
    eval-protocol test, extended to also emit a (clipped, valid) confidence."""
    preds, confs, labels = [], [], []
    for imgs, lbls in loader:
        means = imgs.mean(dim=(1, 2, 3))
        pred = (means > 0.4).long()
        conf = (0.5 + (means - 0.4).abs()).clamp(0.0, 1.0)
        preds.extend(int(p) for p in pred.tolist())
        confs.extend(float(c) for c in conf.tolist())
        labels.extend(int(l) for l in lbls.tolist())
    return preds, confs, labels


def test_id_protocol_produces_clean_plus_twenty_one_conditions():
    samples = make_synthetic_samples(n_per_class=4, n_classes=2, size=(16, 16))
    cube = run_cube_protocol(samples, TRANSFORM, _stub_predict_fn, regime="id",
                              class_names=CLASS_NAMES, batch_size=4)
    assert len(cube._conditions) == 1 + len(ID_TYPES) * N_SEVERITIES


def test_ood_protocol_produces_clean_plus_twenty_four_conditions():
    samples = make_synthetic_samples(n_per_class=4, n_classes=2, size=(16, 16))
    cube = run_cube_protocol(samples, TRANSFORM, _stub_predict_fn, regime="ood",
                              class_names=CLASS_NAMES, batch_size=4)
    assert len(cube._conditions) == 1 + len(OOD_TYPES) * N_SEVERITIES


def test_rejects_unknown_regime():
    samples = make_synthetic_samples(n_per_class=2, n_classes=2, size=(16, 16))
    with pytest.raises(ValueError):
        run_cube_protocol(samples, TRANSFORM, _stub_predict_fn, regime="bogus", class_names=CLASS_NAMES)


def test_default_group_id_is_one_group_per_sample():
    samples = make_synthetic_samples(n_per_class=2, n_classes=2, size=(16, 16))
    cube = run_cube_protocol(samples, TRANSFORM, _stub_predict_fn, regime="id",
                              class_names=CLASS_NAMES, batch_size=4)
    clean = cube._conditions[(None, None)]
    assert len({r.group_id for r in clean.records}) == len(samples)


def test_custom_group_id_fn_groups_samples_together():
    """Models the GC10 case: several crops from one source image share a
    group_id, so bootstrap CIs (eval/bootstrap.py) resample at the image level."""
    samples = make_synthetic_samples(n_per_class=2, n_classes=2, size=(16, 16))
    cube = run_cube_protocol(samples, TRANSFORM, _stub_predict_fn, regime="id",
                              class_names=CLASS_NAMES, group_id_fn=lambda item: "shared_source_image",
                              batch_size=4)
    clean = cube._conditions[(None, None)]
    assert {r.group_id for r in clean.records} == {"shared_source_image"}


def test_worst_slices_and_severity_auc_work_end_to_end():
    samples = make_synthetic_samples(n_per_class=6, n_classes=2, size=(16, 16))
    cube = run_cube_protocol(samples, TRANSFORM, _stub_predict_fn, regime="id",
                              class_names=CLASS_NAMES, batch_size=4, min_samples_per_class=2)
    worst = cube.worst_slices(k=3)
    assert 0 < len(worst) <= 3
    auc = cube.severity_auc("jpeg")
    assert 0.0 <= auc <= 1.0
    assert 0.0 <= cube.overall_mPC() <= 1.0


def test_predict_fn_output_length_mismatch_is_caught():
    def bad_predict_fn(loader):
        return [0], [0.9], [0, 1]  # lengths disagree

    samples = make_synthetic_samples(n_per_class=2, n_classes=2, size=(16, 16))
    with pytest.raises(ValueError):
        run_cube_protocol(samples, TRANSFORM, bad_predict_fn, regime="id",
                           class_names=CLASS_NAMES, batch_size=4)
