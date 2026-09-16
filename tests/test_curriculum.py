import pytest

from augmentation.curriculum import SeverityCurriculum


def test_default_schedule_unlocks_in_thirds():
    c = SeverityCurriculum(n_severities=3)
    assert c.unlocked_severities(0.0) == [0]
    assert c.unlocked_severities(0.2) == [0]
    assert c.unlocked_severities(1 / 3) == [0, 1]
    assert c.unlocked_severities(0.5) == [0, 1]
    assert c.unlocked_severities(2 / 3) == [0, 1, 2]
    assert c.unlocked_severities(1.0) == [0, 1, 2]


def test_progress_is_clamped_to_unit_interval():
    c = SeverityCurriculum(n_severities=3)
    assert c.unlocked_severities(-5.0) == c.unlocked_severities(0.0)
    assert c.unlocked_severities(5.0) == c.unlocked_severities(1.0)


def test_severity_zero_must_unlock_at_progress_zero():
    with pytest.raises(ValueError):
        SeverityCurriculum(n_severities=3, unlock_fractions=(0.1, 0.4, 0.8))


def test_unlock_fractions_must_be_non_decreasing():
    with pytest.raises(ValueError):
        SeverityCurriculum(n_severities=3, unlock_fractions=(0.0, 0.8, 0.4))


def test_unlock_fractions_length_must_match_n_severities():
    with pytest.raises(ValueError):
        SeverityCurriculum(n_severities=3, unlock_fractions=(0.0, 0.5))


def test_all_severities_available_immediately_is_a_valid_degenerate_case():
    """unlock_fractions=(0,0,0) => the "no curriculum" baseline used to prove
    A-PhysDeg's plumbing reduces to component 1's static sampling."""
    c = SeverityCurriculum(n_severities=3, unlock_fractions=(0.0, 0.0, 0.0))
    assert c.unlocked_severities(0.0) == [0, 1, 2]
