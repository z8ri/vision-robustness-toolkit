import pytest

from eval.metrics import class_support, macro_f1, per_class_f1

LABELS = [0, 0, 1, 1, 2, 2]
PREDS = [0, 1, 1, 1, 2, 0]


def test_per_class_f1_hand_computed():
    f1s = per_class_f1(PREDS, LABELS, num_classes=3)
    assert f1s[0] == pytest.approx(0.5)
    assert f1s[1] == pytest.approx(0.8)
    assert f1s[2] == pytest.approx(2 / 3)


def test_macro_f1_is_mean_of_per_class():
    f1s = per_class_f1(PREDS, LABELS, num_classes=3)
    assert macro_f1(PREDS, LABELS, num_classes=3) == pytest.approx(sum(f1s) / 3)


def test_perfect_predictions_give_f1_one():
    labels = [0, 1, 2, 0, 1, 2]
    assert macro_f1(labels, labels, num_classes=3) == pytest.approx(1.0)


def test_class_absent_from_both_preds_and_labels_scores_zero_not_crash():
    """A class with zero true instances and zero predictions: precision and
    recall are both 0/0, defined as 0 here rather than raising or NaN-ing."""
    f1s = per_class_f1([0, 0], [0, 0], num_classes=3)
    assert f1s[2] == 0.0


def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        per_class_f1([0, 1], [0], num_classes=2)


def test_class_support_counts_true_instances():
    assert class_support(LABELS, 0) == 2
    assert class_support(LABELS, 1) == 2
    assert class_support(LABELS, 2) == 2
    assert class_support(LABELS, 5) == 0
