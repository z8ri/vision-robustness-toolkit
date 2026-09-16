"""calibration/selective.py — calibrated selective prediction (accept / reject).

Design doc U10: calibrated MSP (max softmax probability) is the primary
rejection score; energy score is an available alternative, never the default.
When a sample's score falls below the validation-selected threshold, the output
is "reject / recommend human review" — never "predicted as an unknown class"
(there is no such class), and never a claim that a human-review system already
exists downstream (that is the caller's integration to build, not this module's
to assume).

`SelectivePredictor.fit` requires `split="calibration"` explicitly, mirroring
the split guards in augmentation/a_physdeg.py (split="train") and
eval/split_guard.py (test/ood) — the same discipline applied to a third split
role: a threshold must never be picked by peeking at what would make test/OOD
numbers look best.
"""
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import torch

from .temperature import apply_temperature, fit_temperature

ScoreFn = Callable[[torch.Tensor], torch.Tensor]  # logits (N,C) -> confidence (N,), higher = more confident


def msp_confidence(probs: torch.Tensor) -> torch.Tensor:
    return probs.max(dim=1).values


def energy_confidence(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """-E(x) with E(x) = -T * logsumexp(z/T); returned with the sign flipped so
    higher == more confident, the same convention as msp_confidence."""
    return temperature * torch.logsumexp(logits / temperature, dim=1)


def select_threshold_for_coverage(confidences: torch.Tensor, target_coverage: float) -> float:
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError(f"target_coverage must be in (0, 1], got {target_coverage}")
    return float(torch.quantile(confidences, 1.0 - target_coverage))


@dataclass(frozen=True)
class Decision:
    pred_label: int
    confidence: float
    accepted: bool


def describe_decision(decision: Decision, class_names: Sequence[str]) -> str:
    if not decision.accepted:
        return "拒绝判断 / 建议复核"
    return class_names[decision.pred_label]


class SelectivePredictor:
    def __init__(self, temperature: float, threshold: float, score_fn: Optional[ScoreFn] = None):
        if temperature <= 0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.temperature = temperature
        self.threshold = threshold
        self.score_fn = score_fn or (lambda logits: msp_confidence(apply_temperature(logits, self.temperature)))

    @classmethod
    def fit(
        cls,
        calibration_logits: torch.Tensor,
        calibration_labels: torch.Tensor,
        target_coverage: float,
        split: str = "calibration",
        score: str = "msp",
    ) -> "SelectivePredictor":
        if split != "calibration":
            raise ValueError(
                f"SelectivePredictor.fit(split={split!r}) refused: the threshold must only ever be chosen "
                "on a designated calibration set, never on validation-for-model-selection or on test/OOD "
                "('冻结模型,仅在...校准混合上拟合'). Pass split='calibration' explicitly."
            )
        if score not in ("msp", "energy"):
            raise ValueError(f"score must be 'msp' or 'energy', got {score!r}")

        temperature = fit_temperature(calibration_logits, calibration_labels)
        if score == "msp":
            confidences = msp_confidence(apply_temperature(calibration_logits, temperature))
            score_fn: ScoreFn = lambda logits, t=temperature: msp_confidence(apply_temperature(logits, t))
        else:
            confidences = energy_confidence(calibration_logits, temperature)
            score_fn = lambda logits, t=temperature: energy_confidence(logits, t)

        threshold = select_threshold_for_coverage(confidences, target_coverage)
        return cls(temperature, threshold, score_fn)

    def predict(self, logits: torch.Tensor) -> List[Decision]:
        probs = apply_temperature(logits, self.temperature)
        preds = probs.argmax(dim=1)
        confidences = self.score_fn(logits)
        return [
            Decision(int(preds[i]), float(confidences[i]), bool(confidences[i] >= self.threshold))
            for i in range(logits.shape[0])
        ]

    def coverage_and_risk(self, logits: torch.Tensor, labels: torch.Tensor):
        """Realized coverage (fraction accepted) and risk (error rate *among
        accepted samples* — not 1 - macro_f1, and not a promise this holds under
        a different distribution than the one it was measured on)."""
        decisions = self.predict(logits)
        accepted = [d for d in decisions if d.accepted]
        coverage = len(accepted) / len(decisions) if decisions else 0.0
        if not accepted:
            return coverage, None
        errors = sum(1 for d, y in zip(decisions, labels.tolist()) if d.accepted and d.pred_label != y)
        risk = errors / len(accepted)
        return coverage, risk
