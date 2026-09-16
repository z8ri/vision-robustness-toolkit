import pytest

from eval.split_guard import FinalizeGuard


def test_first_finalize_of_test_split_succeeds():
    g = FinalizeGuard()
    g.finalize("test")
    assert g.is_finalized("test")


def test_second_finalize_of_test_split_raises_without_override():
    g = FinalizeGuard()
    g.finalize("test")
    with pytest.raises(RuntimeError):
        g.finalize("test")


def test_second_finalize_allowed_with_explicit_override():
    g = FinalizeGuard()
    g.finalize("test")
    g.finalize("test", allow_repeat=True)  # should not raise
    assert g.is_finalized("test")


def test_ood_split_is_guarded_the_same_way():
    g = FinalizeGuard()
    g.finalize("ood")
    with pytest.raises(RuntimeError):
        g.finalize("ood")


def test_train_and_val_splits_are_never_guarded():
    g = FinalizeGuard()
    g.finalize("train")
    g.finalize("train")  # must not raise, any number of times
    g.finalize("val")
    g.finalize("val")


def test_different_guard_instances_are_independent():
    g1, g2 = FinalizeGuard(), FinalizeGuard()
    g1.finalize("test")
    g2.finalize("test")  # a fresh guard must not inherit g1's state
