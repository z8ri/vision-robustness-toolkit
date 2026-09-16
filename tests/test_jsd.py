import pytest
import torch

from losses.jsd import jsd_loss


def test_jsd_is_zero_when_all_three_views_agree():
    torch.manual_seed(0)
    logits = torch.randn(8, 5)
    assert jsd_loss(logits, logits, logits).item() == pytest.approx(0.0, abs=1e-5)


def test_jsd_is_nonnegative_on_random_logits():
    torch.manual_seed(1)
    for _ in range(5):
        a, b, c = (torch.randn(4, 6) for _ in range(3))
        assert jsd_loss(a, b, c).item() >= -1e-6


def test_jsd_is_symmetric_under_permutation_of_the_three_views():
    torch.manual_seed(2)
    a, b, c = (torch.randn(4, 6) for _ in range(3))
    v_abc = jsd_loss(a, b, c).item()
    v_bca = jsd_loss(b, c, a).item()
    v_cab = jsd_loss(c, a, b).item()
    assert v_abc == pytest.approx(v_bca, abs=1e-5)
    assert v_abc == pytest.approx(v_cab, abs=1e-5)


def test_jsd_rejects_mismatched_shapes():
    a = torch.randn(4, 5)
    b = torch.randn(4, 6)
    c = torch.randn(4, 5)
    with pytest.raises(ValueError):
        jsd_loss(a, b, c)


def test_gradient_flows_through_all_three_views():
    torch.manual_seed(3)
    a = torch.randn(4, 5, requires_grad=True)
    b = torch.randn(4, 5, requires_grad=True)
    c = torch.randn(4, 5, requires_grad=True)
    jsd_loss(a, b, c).backward()
    assert a.grad is not None and b.grad is not None and c.grad is not None
