"""augmentation/curriculum.py — "先轻后重" severity curriculum.

A single, global unlock schedule over severity indices: severity 0 is always
available (training starts on the lightest level), and each heavier severity
unlocks once training progress crosses its threshold fraction. Applied
identically to all 7 in-distribution degradation types — the design doc does not
call for per-type schedules, and a single shared schedule is the simplest thing
that satisfies "先轻后重地开放 severity".
"""
from typing import List, Sequence


class SeverityCurriculum:
    def __init__(self, n_severities: int, unlock_fractions: Sequence[float] = (0.0, 1 / 3, 2 / 3)):
        if len(unlock_fractions) != n_severities:
            raise ValueError(
                f"unlock_fractions must have one entry per severity level "
                f"({n_severities}), got {len(unlock_fractions)}"
            )
        if unlock_fractions[0] != 0.0:
            raise ValueError("severity 0 (lightest) must unlock at progress=0.0 ('先轻后重')")
        if list(unlock_fractions) != sorted(unlock_fractions):
            raise ValueError("unlock_fractions must be non-decreasing (heavier severities unlock no earlier)")
        self.n_severities = n_severities
        self.unlock_fractions = tuple(unlock_fractions)

    def unlocked_severities(self, progress: float) -> List[int]:
        """progress: fraction of training completed, in [0, 1] (e.g. epoch / total_epochs)."""
        progress = max(0.0, min(1.0, progress))
        return [s for s, frac in enumerate(self.unlock_fractions) if frac <= progress]
