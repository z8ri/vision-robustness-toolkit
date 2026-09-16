"""eval/metrics.py — from-scratch multiclass per-class / macro F1 (no sklearn
dependency, consistent with the rest of this repo)."""
from typing import List, Sequence


def per_class_f1(preds: Sequence[int], labels: Sequence[int], num_classes: int) -> List[float]:
    if len(preds) != len(labels):
        raise ValueError(f"preds and labels must be the same length, got {len(preds)} vs {len(labels)}")
    f1s = []
    for c in range(num_classes):
        tp = sum(1 for p, y in zip(preds, labels) if p == c and y == c)
        fp = sum(1 for p, y in zip(preds, labels) if p == c and y != c)
        fn = sum(1 for p, y in zip(preds, labels) if p != c and y == c)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1s.append(f1)
    return f1s


def macro_f1(preds: Sequence[int], labels: Sequence[int], num_classes: int) -> float:
    f1s = per_class_f1(preds, labels, num_classes)
    return sum(f1s) / len(f1s) if f1s else 0.0


def class_support(labels: Sequence[int], class_idx: int) -> int:
    """Number of ground-truth instances of `class_idx` — what `min_samples`
    gating in RobustnessCube.worst_slices checks against."""
    return sum(1 for y in labels if y == class_idx)
