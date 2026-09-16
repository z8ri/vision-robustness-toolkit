"""eval/split_guard.py — a process-discipline check, not a security boundary.

U9's "防泄漏边界": training statistics only serve training; validation is for
model/hyperparameter selection; once a protocol is frozen, test/OOD are used for
the final report ONLY — peeking at test, tweaking something, and re-running
"the same independent test" is exactly the anti-pattern this catches. This is a
runtime nudge (it can't stop someone from constructing a fresh guard to route
around it), not a cryptographic guarantee — documented as such rather than
oversold.
"""
_GUARDED_SPLITS = ("test", "ood")


class FinalizeGuard:
    def __init__(self):
        self._finalized: set = set()

    def finalize(self, split: str, allow_repeat: bool = False) -> None:
        if split in _GUARDED_SPLITS and split in self._finalized and not allow_repeat:
            raise RuntimeError(
                f"split={split!r} was already finalized once on this guard. Re-finalizing without "
                "allow_repeat=True would let a 'final' report double as a tuning signal — construct a "
                "fresh guard (and, ideally, re-collect the held-out data) instead of calling this twice."
            )
        self._finalized.add(split)

    def is_finalized(self, split: str) -> bool:
        return split in self._finalized
