import math

import pytest
import torch

from calibration.metrics import aurc, brier_score, expected_calibration_error, nll, risk_coverage_curve


# ---- nll ----

def test_nll_zero_for_near_perfect_probabilities():
    labels = torch.tensor([0, 1, 2])
    probs = torch.tensor([
        [0.999, 0.0005, 0.0005],
        [0.0005, 0.999, 0.0005],
        [0.0005, 0.0005, 0.999],
    ])
    assert nll(probs, labels) == pytest.approx(0.0, abs=1e-2)


def test_nll_of_uniform_probabilities_equals_log_num_classes():
    n, c = 20, 4
    labels = torch.randint(0, c, (n,))
    probs = torch.full((n, c), 1.0 / c)
    assert nll(probs, labels) == pytest.approx(math.log(c), rel=1e-4)


# ---- brier_score ----

def test_brier_score_zero_for_perfect_one_hot_predictions():
    labels = torch.tensor([0, 1, 2])
    probs = torch.nn.functional.one_hot(labels, 3).float()
    assert brier_score(probs, labels, num_classes=3) == pytest.approx(0.0)


def test_brier_score_uniform_probabilities_hand_computed():
    """C=4 uniform probs (0.25 each): Brier per-sample = (1-0.25)^2 + 3*(0.25)^2
    = 0.5625 + 0.1875 = 0.75, regardless of the true label."""
    n, c = 10, 4
    labels = torch.randint(0, c, (n,))
    probs = torch.full((n, c), 0.25)
    assert brier_score(probs, labels, num_classes=c) == pytest.approx(0.75)


# ---- expected_calibration_error ----

def test_ece_near_zero_for_perfectly_calibrated_bins():
    """Two bins: confidence~0.9 with 90% accuracy, confidence~0.6 with 60%
    accuracy — a textbook perfectly-calibrated setup."""
    torch.manual_seed(0)
    n_per_bin = 100
    conf_hi = torch.full((n_per_bin,), 0.9)
    correct_hi = (torch.rand(n_per_bin) < 0.9).float()
    conf_lo = torch.full((n_per_bin,), 0.6)
    correct_lo = (torch.rand(n_per_bin) < 0.6).float()
    confidences = torch.cat([conf_hi, conf_lo])
    correct = torch.cat([correct_hi, correct_lo])
    assert expected_calibration_error(confidences, correct, n_bins=10) < 0.05


def test_ece_large_for_systematically_overconfident_predictions():
    """Confidence always 0.99 but accuracy only ~50% -> ECE close to 0.49."""
    torch.manual_seed(1)
    n = 500
    confidences = torch.full((n,), 0.99)
    correct = (torch.rand(n) < 0.5).float()
    ece = expected_calibration_error(confidences, correct, n_bins=10)
    assert ece == pytest.approx(0.49, abs=0.03)


def test_ece_shape_mismatch_raises():
    with pytest.raises(ValueError):
        expected_calibration_error(torch.rand(5), torch.rand(4))


# ---- risk_coverage_curve / aurc ----

def test_risk_at_full_coverage_equals_overall_error_rate():
    torch.manual_seed(0)
    n = 50
    confidences = torch.rand(n)
    correct = (torch.rand(n) < 0.7).float()
    curve = risk_coverage_curve(confidences, correct)
    coverage, risk = curve[-1]
    assert coverage == pytest.approx(1.0)
    assert risk == pytest.approx(1.0 - float(correct.mean()))


def test_risk_coverage_curve_perfect_confidence_ranking():
    """7 correct samples all ranked above 3 incorrect ones: risk must be exactly
    0 through coverage=0.7, then increase as wrong samples get folded in."""
    confidences = torch.tensor([0.99, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80, 0.5, 0.4, 0.3])
    correct = torch.tensor([1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.float32)
    curve = risk_coverage_curve(confidences, correct)
    for k in range(7):
        assert curve[k][1] == pytest.approx(0.0)
    assert curve[7][1] == pytest.approx(1 / 8)
    assert curve[8][1] == pytest.approx(2 / 9)
    assert curve[9][1] == pytest.approx(3 / 10)


def test_risk_coverage_curve_rejects_empty_input():
    with pytest.raises(ValueError):
        risk_coverage_curve(torch.tensor([]), torch.tensor([]))


def test_aurc_matches_manual_trapezoid_reference():
    torch.manual_seed(3)
    n = 30
    confidences = torch.rand(n)
    correct = (torch.rand(n) < 0.6).float()
    curve = risk_coverage_curve(confidences, correct)
    xs = [0.0] + [c for c, _ in curve]
    ys = [curve[0][1]] + [r for _, r in curve]
    reference = sum((ys[i] + ys[i + 1]) / 2 * (xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
    assert aurc(confidences, correct) == pytest.approx(reference, rel=1e-9)


def test_aurc_is_lower_for_better_ranked_confidence():
    """Same underlying correctness pattern, but one confidence ordering ranks
    correct samples first (good) and the other ranks them last (bad, adversarial
    ranking) — AURC must be strictly lower for the good ranking."""
    correct = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0, 0, 0], dtype=torch.float32)
    good_conf = torch.tensor([0.9, 0.9, 0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1])
    bad_conf = torch.tensor([0.1, 0.1, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9])
    assert aurc(good_conf, correct) < aurc(bad_conf, correct)
