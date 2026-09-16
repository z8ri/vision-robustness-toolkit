"""augmentation/difficulty.py — per-(type, severity) difficulty EMA + a bounded
mixture-of-(uniform, difficulty) sampling distribution.

Difficulty values here are plain Python floats, never torch tensors touched by
autograd — "困难度只作为停止梯度的采样统计" (design doc U8) is true by
construction: there is nothing to detach because a plain float was never part of
a graph.
"""
import random
from typing import Dict, List, Sequence, Tuple

Cell = Tuple[str, int]  # (deg_type, severity)


class DifficultyEMA:
    """Exponential moving average of a difficulty signal (e.g. 1 - accuracy, or a
    normalized loss) per (deg_type, severity) cell."""

    def __init__(self, deg_types: Sequence[str], n_severities: int, decay: float = 0.9,
                 init_value: float = 0.5):
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"decay must be in [0, 1), got {decay}")
        if not 0.0 <= init_value <= 1.0:
            raise ValueError(f"init_value must be in [0, 1], got {init_value}")
        self.decay = decay
        self._values: Dict[Cell, float] = {
            (t, s): init_value for t in deg_types for s in range(n_severities)
        }

    def update(self, deg_type: str, severity: int, difficulty: float) -> None:
        key = (deg_type, severity)
        if key not in self._values:
            raise KeyError(f"unknown cell {key}")
        difficulty = max(0.0, min(1.0, float(difficulty)))
        old = self._values[key]
        self._values[key] = self.decay * old + (1.0 - self.decay) * difficulty

    def get(self, deg_type: str, severity: int) -> float:
        return self._values[(deg_type, severity)]

    def as_dict(self) -> Dict[Cell, float]:
        return dict(self._values)


def cap_and_renormalize(weights: Sequence[float], cap: float) -> List[float]:
    """Water-filling: clip every weight to <= cap, redistribute the removed mass
    proportionally among the still-uncapped weights, repeat until stable. Returns
    a list summing to 1 with every entry <= cap (guaranteed as long as
    cap * len(weights) >= 1).
    """
    n = len(weights)
    if n == 0:
        return []
    total = sum(weights)
    if total <= 0:
        weights = [1.0] * n
        total = float(n)
    if cap * n < 1.0 - 1e-9:
        raise ValueError(f"prob_cap={cap} is infeasible for n={n} cells (need cap*n >= 1)")

    p = [w / total for w in weights]
    fixed = [False] * n
    for _ in range(n):
        over = [i for i in range(n) if not fixed[i] and p[i] > cap + 1e-12]
        if not over:
            break
        for i in over:
            p[i] = cap
            fixed[i] = True
        remaining_mass = 1.0 - sum(p[i] for i in range(n) if fixed[i])
        free = [i for i in range(n) if not fixed[i]]
        if not free:
            break
        free_sum = sum(p[i] for i in free)
        if free_sum > 0:
            for i in free:
                p[i] = p[i] / free_sum * remaining_mass
        else:
            for i in free:
                p[i] = remaining_mass / len(free)
    return p


def mixed_distribution(cells: Sequence[Cell], difficulty: DifficultyEMA,
                        mix_alpha: float, prob_cap: float) -> Dict[Cell, float]:
    """(1 - mix_alpha) * uniform + mix_alpha * difficulty-proportional, then capped.

    mix_alpha=0.0 -> pure uniform over `cells` (the "fixed curriculum, no
    difficulty" ablation). mix_alpha=1.0 -> pure difficulty-proportional before
    capping (uniform floor disappears, so the prob_cap is the only thing left
    guarding against a runaway hard-example cell).
    """
    if not 0.0 <= mix_alpha <= 1.0:
        raise ValueError(f"mix_alpha must be in [0, 1], got {mix_alpha}")
    n = len(cells)
    if n == 0:
        return {}
    uniform_w = 1.0 / n
    diff_vals = [difficulty.get(*c) for c in cells]
    diff_sum = sum(diff_vals)
    if diff_sum > 0:
        mixed = [(1 - mix_alpha) * uniform_w + mix_alpha * (d / diff_sum) for d in diff_vals]
    else:
        mixed = [uniform_w] * n
    capped = cap_and_renormalize(mixed, prob_cap)
    return dict(zip(cells, capped))


def weighted_sample_without_replacement(rng: random.Random, items: Sequence, weights: Sequence[float],
                                         k: int) -> list:
    """Sequential weighted sampling without replacement. Falls back to a uniform
    draw among the remaining pool if all remaining weights are (numerically) 0,
    rather than dividing by zero — this can happen when mix_alpha=1.0 and a type
    has driven every one of its unlocked cells' difficulty to 0."""
    items = list(items)
    weights = [max(0.0, float(w)) for w in weights]
    k = min(k, len(items))
    chosen = []
    for _ in range(k):
        total = sum(weights)
        if total <= 0:
            idx = rng.randrange(len(items))
        else:
            r = rng.random() * total
            upto = 0.0
            idx = len(items) - 1
            for i, w in enumerate(weights):
                upto += w
                if upto >= r:
                    idx = i
                    break
        chosen.append(items.pop(idx))
        weights.pop(idx)
    return chosen
