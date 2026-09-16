import pytest
import torch

from calibration.selective import (
    Decision,
    SelectivePredictor,
    describe_decision,
    energy_confidence,
    msp_confidence,
    select_threshold_for_coverage,
)

CLASS_NAMES = ["scratch", "dent", "crack"]


# ---- confidence scores ----

def test_msp_confidence_is_max_probability():
    probs = torch.tensor([[0.7, 0.2, 0.1], [0.3, 0.3, 0.4]])
    assert torch.allclose(msp_confidence(probs), torch.tensor([0.7, 0.4]))


def test_energy_confidence_is_higher_for_more_peaked_logits():
    peaked = torch.tensor([[8.0, 0.0, 0.0]])
    flat = torch.tensor([[0.1, 0.05, 0.0]])
    assert float(energy_confidence(peaked)) > float(energy_confidence(flat))


# ---- threshold selection ----

def test_threshold_gives_approximately_the_target_coverage_on_the_same_data():
    torch.manual_seed(0)
    confidences = torch.rand(2000)
    threshold = select_threshold_for_coverage(confidences, target_coverage=0.9)
    realized = float((confidences >= threshold).float().mean())
    assert realized == pytest.approx(0.9, abs=0.02)


def test_threshold_rejects_invalid_coverage():
    with pytest.raises(ValueError):
        select_threshold_for_coverage(torch.rand(10), target_coverage=0.0)
    with pytest.raises(ValueError):
        select_threshold_for_coverage(torch.rand(10), target_coverage=1.5)


# ---- describe_decision: never claims an "unknown class" ----

def test_describe_decision_accepted_returns_class_name():
    d = Decision(pred_label=1, confidence=0.9, accepted=True)
    assert describe_decision(d, CLASS_NAMES) == "dent"


def test_describe_decision_rejected_recommends_review_not_unknown_class():
    d = Decision(pred_label=1, confidence=0.2, accepted=False)
    out = describe_decision(d, CLASS_NAMES)
    assert out == "拒绝判断 / 建议复核"
    assert out not in CLASS_NAMES
    assert "unknown" not in out.lower()


# ---- SelectivePredictor.fit split guard ----

def _synthetic_logits_labels(n=300, c=4, seed=0):
    torch.manual_seed(seed)
    labels = torch.randint(0, c, (n,))
    logits = torch.nn.functional.one_hot(labels, c).float() * 4.0
    logits += torch.randn(n, c) * 1.5  # add noise so it's not trivially perfect
    return logits, labels


def test_fit_rejects_non_calibration_split():
    logits, labels = _synthetic_logits_labels()
    with pytest.raises(ValueError):
        SelectivePredictor.fit(logits, labels, target_coverage=0.9, split="test")


def test_fit_rejects_unknown_score():
    logits, labels = _synthetic_logits_labels()
    with pytest.raises(ValueError):
        SelectivePredictor.fit(logits, labels, target_coverage=0.9, split="calibration", score="not_a_score")


def test_fit_with_msp_and_energy_both_produce_valid_predictors():
    logits, labels = _synthetic_logits_labels()
    for score in ("msp", "energy"):
        predictor = SelectivePredictor.fit(logits, labels, target_coverage=0.85, split="calibration", score=score)
        assert predictor.temperature > 0.0
        decisions = predictor.predict(logits)
        assert len(decisions) == logits.shape[0]
        assert all(isinstance(d, Decision) for d in decisions)


def test_predictor_constructor_rejects_non_positive_temperature():
    with pytest.raises(ValueError):
        SelectivePredictor(temperature=0.0, threshold=0.5)


# ---- coverage / risk behavior ----

def test_realized_coverage_on_calibration_data_matches_target():
    logits, labels = _synthetic_logits_labels(n=1000)
    predictor = SelectivePredictor.fit(logits, labels, target_coverage=0.8, split="calibration")
    coverage, _risk = predictor.coverage_and_risk(logits, labels)
    assert coverage == pytest.approx(0.8, abs=0.02)


def test_rejection_reduces_risk_below_full_coverage_error_rate():
    """The whole point of selective prediction: risk among *accepted* samples
    should be lower than the unconditional error rate, when confidence actually
    correlates with correctness (true here since noisy logits mean the
    low-margin samples are more likely to be genuinely wrong)."""
    logits, labels = _synthetic_logits_labels(n=1500, seed=7)
    predictor = SelectivePredictor.fit(logits, labels, target_coverage=0.7, split="calibration")

    full_coverage_error = float((logits.argmax(dim=1) != labels).float().mean())
    _coverage, risk = predictor.coverage_and_risk(logits, labels)
    assert risk < full_coverage_error


def test_coverage_and_risk_handles_zero_accepted_samples_gracefully():
    logits, labels = _synthetic_logits_labels(n=50)
    predictor = SelectivePredictor(temperature=1.0, threshold=1e9)  # nothing clears this bar
    coverage, risk = predictor.coverage_and_risk(logits, labels)
    assert coverage == 0.0
    assert risk is None
