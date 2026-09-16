import pytest

from eval.cube import RobustnessCube
from eval.metrics import macro_f1, per_class_f1
from eval.records import PredictionRecord

CLASS_NAMES = ["a", "b", "c"]


def _recs(preds, labels, confidences=None, prefix="s"):
    confidences = confidences or [0.9] * len(preds)
    return [
        PredictionRecord(f"{prefix}{i}", f"{prefix}{i}", labels[i], preds[i], confidences[i])
        for i in range(len(preds))
    ]


# ---- basic ingestion ----

def test_class_f1_matches_metrics_module_independently():
    labels, preds = [0, 0, 1, 1, 2, 2], [0, 1, 1, 1, 2, 0]
    cube = RobustnessCube(CLASS_NAMES)
    cube.add_condition(None, None, _recs(preds, labels))
    expected = per_class_f1(preds, labels, 3)
    for c in range(3):
        assert cube.class_f1(None, None, c) == pytest.approx(expected[c])
    assert cube.condition_macro_f1(None, None) == pytest.approx(macro_f1(preds, labels, 3))


def test_add_condition_rejects_duplicate():
    cube = RobustnessCube(CLASS_NAMES)
    cube.add_condition(None, None, _recs([0, 1], [0, 1]))
    with pytest.raises(ValueError):
        cube.add_condition(None, None, _recs([0, 1], [0, 1]))


def test_add_condition_rejects_empty():
    cube = RobustnessCube(CLASS_NAMES)
    with pytest.raises(ValueError):
        cube.add_condition(None, None, [])


def test_add_condition_rejects_out_of_range_label():
    cube = RobustnessCube(CLASS_NAMES)
    with pytest.raises(ValueError):
        cube.add_condition(None, None, _recs([0, 3], [0, 1]))  # 3 is out of range for 3 classes


def test_unknown_condition_raises_keyerror():
    cube = RobustnessCube(CLASS_NAMES)
    cube.add_condition(None, None, _recs([0, 1], [0, 1]))
    with pytest.raises(KeyError):
        cube.class_f1("motion_blur", 0, 0)


def test_empty_class_names_rejected():
    with pytest.raises(ValueError):
        RobustnessCube([])


# ---- overall mPC ----

def test_overall_mPC_averages_non_clean_conditions_and_excludes_clean():
    cube = RobustnessCube(["a", "b"])
    cube.add_condition(None, None, _recs([0, 1], [0, 1]))       # clean, f1=1.0 — must be excluded
    cube.add_condition("jpeg", 0, _recs([0, 0], [0, 1]))         # macro f1 = 1/3
    cube.add_condition("jpeg", 1, _recs([1, 1], [0, 1]))         # macro f1 = 1/3
    assert cube.overall_mPC() == pytest.approx(1 / 3, rel=1e-3)


def test_overall_mPC_requires_at_least_one_condition():
    cube = RobustnessCube(["a", "b"])
    with pytest.raises(RuntimeError):
        cube.overall_mPC()


# ---- worst-slice reporting ----

def test_worst_slices_excludes_slice_below_min_samples_floor():
    cube = RobustnessCube(CLASS_NAMES, min_samples_per_class=3)
    labels = [0, 0, 1, 1, 2]
    preds = [0, 0, 1, 1, 0]  # class 2 has only 1 true instance, predicted wrong (f1=0)
    cube.add_condition("jpeg", 0, _recs(preds, labels))
    worst = cube.worst_slices(k=10)
    assert all(not (s.deg_type == "jpeg" and s.class_idx == 2) for s in worst)


def test_worst_slices_includes_slice_meeting_floor_and_sorts_ascending():
    cube = RobustnessCube(CLASS_NAMES, min_samples_per_class=2)
    labels = [0, 0, 1, 1, 2, 2]
    preds = [0, 0, 1, 1, 0, 0]  # class 2: 2 true instances, both misclassified -> f1=0
    cube.add_condition("jpeg", 0, _recs(preds, labels))
    worst = cube.worst_slices(k=1)
    assert worst[0].deg_type == "jpeg" and worst[0].class_idx == 2 and worst[0].f1 == pytest.approx(0.0)
    assert worst[0].n_samples == 2


# ---- severity AUC ----

def test_severity_auc_requires_clean_condition():
    cube = RobustnessCube(CLASS_NAMES)
    cube.add_condition("jpeg", 0, _recs([0, 1], [0, 1]))
    with pytest.raises(RuntimeError):
        cube.severity_auc("jpeg", n_severities=1)


def test_severity_auc_of_flat_perfect_curve_equals_one():
    labels = [0, 0, 1, 1]
    perfect = _recs([0, 0, 1, 1], labels)
    cube = RobustnessCube(["a", "b"])
    cube.add_condition(None, None, perfect)
    for s in range(3):
        cube.add_condition("jpeg", s, perfect)
    assert cube.severity_auc("jpeg", n_severities=3) == pytest.approx(1.0)


def test_severity_auc_drops_when_performance_degrades_with_severity():
    labels = [0, 0, 1, 1]
    cube = RobustnessCube(["a", "b"])
    cube.add_condition(None, None, _recs([0, 0, 1, 1], labels))   # f1 = 1.0
    cube.add_condition("jpeg", 0, _recs([0, 0, 1, 1], labels))     # f1 = 1.0
    cube.add_condition("jpeg", 1, _recs([0, 0, 1, 0], labels))     # degraded
    cube.add_condition("jpeg", 2, _recs([1, 0, 0, 1], labels))     # degraded further
    assert cube.severity_auc("jpeg", n_severities=3) < 1.0


def test_severity_auc_per_class_matches_class_f1_on_flat_curve():
    labels = [0, 0, 1, 1]
    perfect = _recs([0, 0, 1, 1], labels)
    cube = RobustnessCube(["a", "b"])
    cube.add_condition(None, None, perfect)
    for s in range(2):
        cube.add_condition("jpeg", s, perfect)
    assert cube.severity_auc("jpeg", class_idx=0, n_severities=2) == pytest.approx(1.0)


# ---- bad cases ----

def test_bad_cases_high_confidence_error_sorted_descending():
    labels = [0, 1, 0, 1]
    preds = [0, 0, 1, 1]  # s1 wrong (pred0/true1), s2 wrong (pred1/true0)
    confidences = [0.9, 0.6, 0.95, 0.99]
    cube = RobustnessCube(["a", "b"])
    cube.add_condition("jpeg", 0, _recs(preds, labels, confidences))
    cases = cube.bad_cases("jpeg", 0, k=5, mode="high_confidence_error")
    assert [c.sample_id for c in cases] == ["s2", "s1"]


def test_bad_cases_largest_clean_drop_prioritizes_correct_to_wrong_flips():
    labels = [0, 1]
    cube = RobustnessCube(["a", "b"])
    cube.add_condition(None, None, _recs([0, 1], labels, [0.99, 0.99]))  # both correct in clean
    cube.add_condition("jpeg", 0, _recs([0, 0], labels, [0.9, 0.8]))     # s0 stays correct, s1 flips
    cases = cube.bad_cases("jpeg", 0, k=5, mode="largest_clean_drop")
    assert cases[0].sample_id == "s1"
    assert cases[0].clean_correct is True
    assert cases[0].cond_correct is False


def test_bad_cases_rejects_unknown_mode():
    cube = RobustnessCube(["a", "b"])
    cube.add_condition(None, None, _recs([0, 1], [0, 1]))
    cube.add_condition("jpeg", 0, _recs([0, 1], [0, 1]))
    with pytest.raises(ValueError):
        cube.bad_cases("jpeg", 0, mode="not_a_real_mode")
