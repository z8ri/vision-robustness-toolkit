"""eval/bootstrap.py — source-image group bootstrap confidence intervals.

Resamples whole groups with replacement (not individual records) so that
records sharing one source image (e.g. several GC10 crops) always move
together — a naive per-record bootstrap would treat those crops as independent
evidence and understate the true uncertainty.
"""
import random
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from .records import PredictionRecord

MetricFn = Callable[[Sequence[PredictionRecord]], float]


@dataclass(frozen=True)
class BootstrapCI:
    point: float
    lower: float
    upper: float
    alpha: float
    n_boot: int
    draws: Tuple[float, ...]  # every bootstrap replicate's metric value, sorted


def group_bootstrap_ci(
    records: Sequence[PredictionRecord],
    metric_fn: MetricFn,
    n_boot: int = 1000,
    alpha: float = 0.05,
    rng: Optional[random.Random] = None,
) -> BootstrapCI:
    if not records:
        raise ValueError("group_bootstrap_ci needs at least one record")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    rng = rng or random.Random()

    by_group: dict = {}
    for r in records:
        by_group.setdefault(r.group_id, []).append(r)
    groups: List[str] = list(by_group.keys())
    n_groups = len(groups)

    point = metric_fn(records)

    draws: List[float] = []
    for _ in range(n_boot):
        resampled_groups = [groups[rng.randrange(n_groups)] for _ in range(n_groups)]
        resampled_records: List[PredictionRecord] = []
        for g in resampled_groups:
            resampled_records.extend(by_group[g])
        draws.append(metric_fn(resampled_records))

    draws.sort()
    lo_idx = int((alpha / 2) * n_boot)
    hi_idx = min(n_boot - 1, int((1 - alpha / 2) * n_boot))
    return BootstrapCI(point=point, lower=draws[lo_idx], upper=draws[hi_idx], alpha=alpha, n_boot=n_boot,
                        draws=tuple(draws))
