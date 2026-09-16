import pytest
import torch

from calibration.temperature import apply_temperature, fit_temperature
from eval.metrics import macro_f1


def test_fit_temperature_rejects_bad_shapes():
    with pytest.raises(ValueError):
        fit_temperature(torch.randn(5), torch.randint(0, 3, (5,)))  # logits not 2D
    with pytest.raises(ValueError):
        fit_temperature(torch.randn(5, 3), torch.randint(0, 3, (4,)))  # length mismatch


def test_temperature_is_always_positive():
    torch.manual_seed(0)
    logits = torch.randn(50, 4)
    labels = torch.randint(0, 4, (50,))
    t = fit_temperature(logits, labels)
    assert t > 0.0


def test_softens_confidently_wrong_predictions():
    """Classic Guo et al. 2017 scenario: a model with large-margin logits that
    is sometimes confidently wrong should be softened (T > 1) to reduce NLL."""
    torch.manual_seed(0)
    n, c = 200, 5
    labels = torch.randint(0, c, (n,))
    logits = torch.nn.functional.one_hot(labels, c).float() * 8.0
    noisy = torch.rand(n) < 0.3
    wrong_labels = (labels + 1) % c
    logits[noisy] = torch.nn.functional.one_hot(wrong_labels[noisy], c).float() * 8.0

    t = fit_temperature(logits, labels)
    assert t > 1.0


def test_sharpens_underconfident_correct_predictions():
    """A model that is always right but barely above uniform confidence should
    be sharpened (T < 1) to reduce NLL."""
    torch.manual_seed(1)
    n, c = 200, 5
    labels = torch.randint(0, c, (n,))
    logits = torch.nn.functional.one_hot(labels, c).float() * 0.05

    t = fit_temperature(logits, labels)
    assert t < 1.0


def test_apply_temperature_rejects_non_positive_temperature():
    logits = torch.randn(4, 3)
    with pytest.raises(ValueError):
        apply_temperature(logits, 0.0)
    with pytest.raises(ValueError):
        apply_temperature(logits, -1.0)


def test_temperature_scaling_never_changes_argmax():
    """The load-bearing invariant behind 'temperature scaling通常不改变argmax'
    — checked directly on the fitted temperature from the confidently-wrong
    scenario above, not just on a friendly random case."""
    torch.manual_seed(0)
    n, c = 200, 5
    labels = torch.randint(0, c, (n,))
    logits = torch.nn.functional.one_hot(labels, c).float() * 8.0
    noisy = torch.rand(n) < 0.3
    wrong_labels = (labels + 1) % c
    logits[noisy] = torch.nn.functional.one_hot(wrong_labels[noisy], c).float() * 8.0

    t = fit_temperature(logits, labels)
    probs = apply_temperature(logits, t)
    assert torch.equal(probs.argmax(dim=1), logits.argmax(dim=1))


def test_temperature_scaling_does_not_change_macro_f1():
    """Direct test of the boundary claim '不能声称提高macro-F1/mPC': since
    argmax is unchanged, macro-F1 computed from calibrated probabilities must be
    bit-for-bit the same as from raw logits."""
    torch.manual_seed(2)
    n, c = 100, 4
    logits = torch.randn(n, c) * 3
    labels = torch.randint(0, c, (n,))
    t = fit_temperature(logits, labels)
    probs = apply_temperature(logits, t)

    f1_before = macro_f1(logits.argmax(dim=1).tolist(), labels.tolist(), c)
    f1_after = macro_f1(probs.argmax(dim=1).tolist(), labels.tolist(), c)
    assert f1_before == pytest.approx(f1_after)
