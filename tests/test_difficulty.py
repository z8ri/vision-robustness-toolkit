import random

import pytest

from augmentation.difficulty import (
    DifficultyEMA,
    cap_and_renormalize,
    mixed_distribution,
    weighted_sample_without_replacement,
)

DEG_TYPES = ["a", "b", "c"]


# ---- DifficultyEMA ----

def test_ema_initializes_all_cells_to_init_value():
    d = DifficultyEMA(DEG_TYPES, n_severities=3, init_value=0.5)
    for t in DEG_TYPES:
        for s in range(3):
            assert d.get(t, s) == 0.5


def test_ema_converges_toward_repeated_observation():
    d = DifficultyEMA(DEG_TYPES, n_severities=3, decay=0.9, init_value=0.5)
    for _ in range(200):
        d.update("a", 1, 1.0)
    assert d.get("a", 1) > 0.99
    # untouched cells stay at init value
    assert d.get("b", 1) == 0.5


def test_ema_update_clamps_out_of_range_difficulty():
    d = DifficultyEMA(DEG_TYPES, n_severities=3, decay=0.5, init_value=0.5)
    d.update("a", 0, 5.0)   # should clamp to 1.0
    assert d.get("a", 0) == pytest.approx(0.5 * 0.5 + 0.5 * 1.0)
    d2 = DifficultyEMA(DEG_TYPES, n_severities=3, decay=0.5, init_value=0.5)
    d2.update("a", 0, -5.0)  # should clamp to 0.0
    assert d2.get("a", 0) == pytest.approx(0.5 * 0.5 + 0.5 * 0.0)


def test_ema_update_rejects_unknown_cell():
    d = DifficultyEMA(DEG_TYPES, n_severities=3)
    with pytest.raises(KeyError):
        d.update("not_a_type", 0, 0.5)


def test_ema_rejects_invalid_decay_and_init():
    with pytest.raises(ValueError):
        DifficultyEMA(DEG_TYPES, n_severities=3, decay=1.0)
    with pytest.raises(ValueError):
        DifficultyEMA(DEG_TYPES, n_severities=3, init_value=1.5)


def test_difficulty_values_are_plain_floats_not_tensors():
    """'困难度只作为停止梯度的采样统计' — trivially true here since a plain
    Python float was never part of an autograd graph to begin with."""
    d = DifficultyEMA(DEG_TYPES, n_severities=3)
    d.update("a", 0, 0.7)
    assert isinstance(d.get("a", 0), float)


# ---- cap_and_renormalize ----

def test_cap_and_renormalize_sums_to_one_and_respects_cap():
    weights = [0.9, 0.05, 0.03, 0.02]
    capped = cap_and_renormalize(weights, cap=0.3)
    assert sum(capped) == pytest.approx(1.0)
    assert all(w <= 0.3 + 1e-9 for w in capped)


def test_cap_and_renormalize_noop_when_already_under_cap():
    weights = [0.25, 0.25, 0.25, 0.25]
    capped = cap_and_renormalize(weights, cap=0.5)
    assert capped == pytest.approx(weights)


def test_cap_and_renormalize_rejects_infeasible_cap():
    with pytest.raises(ValueError):
        cap_and_renormalize([1.0, 1.0, 1.0, 1.0], cap=0.1)  # cap*n=0.4 < 1


def test_cap_and_renormalize_handles_all_zero_weights():
    capped = cap_and_renormalize([0.0, 0.0, 0.0], cap=0.5)
    assert capped == pytest.approx([1 / 3, 1 / 3, 1 / 3])


# ---- mixed_distribution ----

class _StubDifficulty:
    def __init__(self, values):
        self.values = values

    def get(self, t, s):
        return self.values[(t, s)]


def test_mix_alpha_zero_is_uniform_regardless_of_difficulty():
    cells = [("a", 0), ("b", 0), ("c", 0)]
    stub = _StubDifficulty({("a", 0): 0.99, ("b", 0): 0.01, ("c", 0): 0.5})
    dist = mixed_distribution(cells, stub, mix_alpha=0.0, prob_cap=0.9)
    for c in cells:
        assert dist[c] == pytest.approx(1 / 3)


def test_mix_alpha_one_is_proportional_to_difficulty_before_capping():
    cells = [("a", 0), ("b", 0)]
    stub = _StubDifficulty({("a", 0): 0.75, ("b", 0): 0.25})
    dist = mixed_distribution(cells, stub, mix_alpha=1.0, prob_cap=0.99)
    assert dist[("a", 0)] > dist[("b", 0)]
    assert dist[("a", 0)] == pytest.approx(0.75)
    assert dist[("b", 0)] == pytest.approx(0.25)


def test_mixed_distribution_respects_prob_cap():
    cells = [(f"t{i}", 0) for i in range(10)]
    values = {c: 0.01 for c in cells}
    values[cells[0]] = 100.0  # one cell dominates raw difficulty
    stub = _StubDifficulty(values)
    dist = mixed_distribution(cells, stub, mix_alpha=1.0, prob_cap=0.2)
    assert all(p <= 0.2 + 1e-9 for p in dist.values())
    assert sum(dist.values()) == pytest.approx(1.0)


def test_mixed_distribution_empty_cells_returns_empty_dict():
    assert mixed_distribution([], _StubDifficulty({}), mix_alpha=0.5, prob_cap=0.5) == {}


def test_mixed_distribution_rejects_invalid_mix_alpha():
    with pytest.raises(ValueError):
        mixed_distribution([("a", 0)], _StubDifficulty({("a", 0): 0.5}), mix_alpha=1.5, prob_cap=0.9)


# ---- weighted_sample_without_replacement ----

def test_weighted_sample_returns_k_distinct_items():
    rng = random.Random(0)
    items = ["a", "b", "c", "d"]
    weights = [1.0, 1.0, 1.0, 1.0]
    chosen = weighted_sample_without_replacement(rng, items, weights, k=3)
    assert len(chosen) == 3
    assert len(set(chosen)) == 3
    assert set(chosen) <= set(items)


def test_weighted_sample_k_larger_than_pool_returns_whole_pool():
    rng = random.Random(0)
    chosen = weighted_sample_without_replacement(rng, ["a", "b"], [1.0, 1.0], k=5)
    assert sorted(chosen) == ["a", "b"]


def test_weighted_sample_favors_heavier_weights_on_average():
    rng = random.Random(0)
    hits = {"heavy": 0, "light": 0}
    for _ in range(500):
        rng2 = random.Random(rng.random())
        chosen = weighted_sample_without_replacement(rng2, ["heavy", "light"], [0.95, 0.05], k=1)
        hits[chosen[0]] += 1
    assert hits["heavy"] > hits["light"]


def test_weighted_sample_all_zero_weights_falls_back_to_uniform_without_crashing():
    rng = random.Random(0)
    chosen = weighted_sample_without_replacement(rng, ["a", "b", "c"], [0.0, 0.0, 0.0], k=2)
    assert len(chosen) == 2
    assert len(set(chosen)) == 2
