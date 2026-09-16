"""eval/records.py — the atomic unit the Robustness Cube is built from."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionRecord:
    """One model prediction under one evaluation condition (clean, or a
    specific (deg_type, severity) cell).

    `group_id` is the *source-image* group, not the sample id: several samples
    (e.g. multiple crops from one GC10 source image) can share a group_id, so
    that bootstrap resampling (eval/bootstrap.py) can resample at the group
    level and stay leakage-safe the same way the train/test split itself must be
    (see project docs on GC10's image-level split).
    """

    sample_id: str
    group_id: str
    true_label: int
    pred_label: int
    confidence: float  # max predicted-class probability, in [0, 1]

    def __post_init__(self):
        if not -1e-6 <= self.confidence <= 1.0 + 1e-6:
            raise ValueError(f"confidence must be in [0, 1], got {self.confidence}")

    @property
    def correct(self) -> bool:
        return self.pred_label == self.true_label
