"""eval/cube.py — Robustness Cube: class x degradation-type x severity slices.

Implements U9 from INTERVIEW_VALUE_UPGRADES.md: a single mPC number can hide a
minority class collapsing under one specific (degradation, severity); this
module keeps the full "类别 x 退化类型 x severity" decomposition addressable,
plus the four reporting primitives the design doc names explicitly —
worst-slice (with a minimum-sample floor so a 2-sample slice can't masquerade
as "the worst"), performance-severity AUC, group-bootstrap CIs (bootstrap.py),
and bad-case ranking.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .metrics import class_support as _class_support
from .metrics import macro_f1, per_class_f1
from .records import PredictionRecord

ConditionKey = Tuple[Optional[str], Optional[int]]  # (deg_type, severity); (None, None) == clean


@dataclass(frozen=True)
class ConditionResult:
    deg_type: Optional[str]
    severity: Optional[int]
    records: Tuple[PredictionRecord, ...]

    @property
    def label(self) -> str:
        if self.deg_type is None:
            return "clean"
        return f"{self.deg_type}/s{self.severity + 1}"


@dataclass(frozen=True)
class WorstSlice:
    deg_type: Optional[str]
    severity: Optional[int]
    class_idx: int
    f1: float
    n_samples: int


@dataclass(frozen=True)
class BadCase:
    sample_id: str
    true_label: int
    clean_pred: int
    clean_confidence: float
    clean_correct: bool
    cond_pred: int
    cond_confidence: float
    cond_correct: bool


class RobustnessCube:
    def __init__(self, class_names: Sequence[str], min_samples_per_class: int = 5):
        if not class_names:
            raise ValueError("class_names must be non-empty")
        self.class_names = list(class_names)
        self.num_classes = len(class_names)
        self.min_samples_per_class = min_samples_per_class
        self._conditions: Dict[ConditionKey, ConditionResult] = {}

    # ---- ingestion ----

    def add_condition(self, deg_type: Optional[str], severity: Optional[int],
                       records: Sequence[PredictionRecord]) -> None:
        key = (deg_type, severity)
        if key in self._conditions:
            raise ValueError(
                f"condition {key} already recorded; build a fresh RobustnessCube "
                "rather than silently overwriting an existing slice"
            )
        if not records:
            raise ValueError(f"condition {key} has no records")
        for r in records:
            if not 0 <= r.true_label < self.num_classes:
                raise ValueError(f"true_label {r.true_label} out of range for {self.num_classes} classes")
            if not 0 <= r.pred_label < self.num_classes:
                raise ValueError(f"pred_label {r.pred_label} out of range for {self.num_classes} classes")
        self._conditions[key] = ConditionResult(deg_type, severity, tuple(records))

    def _condition(self, deg_type: Optional[str], severity: Optional[int]) -> ConditionResult:
        key = (deg_type, severity)
        if key not in self._conditions:
            raise KeyError(f"condition {key} was never added to this cube")
        return self._conditions[key]

    # ---- slice metrics ----

    def class_f1(self, deg_type: Optional[str], severity: Optional[int], class_idx: int) -> float:
        cond = self._condition(deg_type, severity)
        preds = [r.pred_label for r in cond.records]
        labels = [r.true_label for r in cond.records]
        return per_class_f1(preds, labels, self.num_classes)[class_idx]

    def condition_macro_f1(self, deg_type: Optional[str], severity: Optional[int]) -> float:
        cond = self._condition(deg_type, severity)
        preds = [r.pred_label for r in cond.records]
        labels = [r.true_label for r in cond.records]
        return macro_f1(preds, labels, self.num_classes)

    def class_support(self, deg_type: Optional[str], severity: Optional[int], class_idx: int) -> int:
        cond = self._condition(deg_type, severity)
        return _class_support([r.true_label for r in cond.records], class_idx)

    def overall_mPC(self) -> float:
        """Mean over every non-clean condition's macro-F1 — the same aggregate
        eval/protocol.py's ProtocolResult.mPC reports, kept consistent here so
        the Cube is a strict refinement, not a competing number."""
        non_clean = [k for k in self._conditions if k != (None, None)]
        if not non_clean:
            raise RuntimeError("overall_mPC needs at least one non-clean condition")
        return sum(self.condition_macro_f1(*k) for k in non_clean) / len(non_clean)

    # ---- worst-slice reporting ----

    def worst_slices(self, k: int = 5, min_samples: Optional[int] = None) -> List[WorstSlice]:
        floor = self.min_samples_per_class if min_samples is None else min_samples
        candidates: List[WorstSlice] = []
        for (deg_type, severity), cond in self._conditions.items():
            for c in range(self.num_classes):
                n = self.class_support(deg_type, severity, c)
                if n < floor:
                    continue
                f1 = self.class_f1(deg_type, severity, c)
                candidates.append(WorstSlice(deg_type, severity, c, f1, n))
        candidates.sort(key=lambda s: s.f1)
        return candidates[:k]

    # ---- performance-severity AUC ----

    def severity_auc(self, deg_type: str, class_idx: Optional[int] = None,
                      n_severities: int = 3) -> float:
        """Trapezoidal area under the F1-vs-severity curve, x-axis = [clean=0,
        s1=1, ..., sN=N], normalized by the x-range so the result stays on the
        [0, 1] F1 scale (a flat curve at F1=v has severity_auc == v)."""
        if (None, None) not in self._conditions:
            raise RuntimeError("severity_auc needs the clean condition to anchor x=0")
        xs = list(range(n_severities + 1))
        ys = []
        for x in xs:
            dt, sev = (None, None) if x == 0 else (deg_type, x - 1)
            ys.append(self.condition_macro_f1(dt, sev) if class_idx is None else self.class_f1(dt, sev, class_idx))
        area = sum((ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))
        return area / (xs[-1] - xs[0])

    # ---- bad-case ranking ----

    def bad_cases(self, deg_type: str, severity: int, k: int = 10,
                  mode: str = "high_confidence_error") -> List[BadCase]:
        if mode not in ("high_confidence_error", "largest_clean_drop"):
            raise ValueError(f"unknown mode {mode!r}")
        cond = self._condition(deg_type, severity)

        if mode == "high_confidence_error":
            wrong = [r for r in cond.records if not r.correct]
            wrong.sort(key=lambda r: r.confidence, reverse=True)
            return [
                BadCase(r.sample_id, r.true_label, r.pred_label, r.confidence, False,
                        r.pred_label, r.confidence, False)
                for r in wrong[:k]
            ]

        # "largest_clean_drop": needs the clean condition to compare against.
        clean = self._condition(None, None)
        clean_by_id = {r.sample_id: r for r in clean.records}
        cases = []
        for r in cond.records:
            c = clean_by_id.get(r.sample_id)
            if c is None:
                continue
            cases.append(BadCase(r.sample_id, r.true_label, c.pred_label, c.confidence, c.correct,
                                  r.pred_label, r.confidence, r.correct))
        # Prioritize flips from correct-in-clean to wrong-here; among those, the
        # samples the model was *most confidently right about* in clean are the
        # most surprising failures.
        cases.sort(key=lambda b: ((1 if b.clean_correct and not b.cond_correct else 0), b.clean_confidence),
                   reverse=True)
        return cases[:k]
