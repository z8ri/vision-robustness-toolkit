"""calibration/metrics.py — NLL, Brier, ECE, risk-coverage curve, AURC.

Deliberately does NOT implement conformal prediction / RAPS: the design doc is
explicit that a conformal-style coverage guarantee would be false advertising
under distribution shift ("Conformal/RAPS在分布偏移下不能被写成无条件覆盖保证"),
and this project's whole point is honest characterization over impressive-
sounding claims — so it is left out rather than implemented and mis-described.
"""
from typing import List, Tuple

import torch


def nll(probs: torch.Tensor, labels: torch.Tensor, eps: float = 1e-12) -> float:
    picked = probs.gather(1, labels.view(-1, 1)).squeeze(1).clamp(min=eps)
    return float(-picked.log().mean())


def brier_score(probs: torch.Tensor, labels: torch.Tensor, num_classes: int) -> float:
    onehot = torch.nn.functional.one_hot(labels, num_classes).float()
    return float(((probs - onehot) ** 2).sum(dim=1).mean())


def expected_calibration_error(confidences: torch.Tensor, correct: torch.Tensor, n_bins: int = 10) -> float:
    if confidences.shape != correct.shape:
        raise ValueError("confidences and correct must have the same shape")
    correct = correct.float()
    n = confidences.shape[0]
    edges = torch.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (confidences >= lo) & (confidences <= hi if i == n_bins - 1 else confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += (mask.float().sum() / n) * (bin_conf - bin_acc).abs()
    return float(ece)


def risk_coverage_curve(confidences: torch.Tensor, correct: torch.Tensor) -> List[Tuple[float, float]]:
    """Sort by confidence descending; at coverage=k/N (the k most confident
    samples), risk = error rate within that accepted subset."""
    if confidences.shape[0] == 0:
        raise ValueError("risk_coverage_curve needs at least one sample")
    order = torch.argsort(confidences, descending=True)
    correct_sorted = correct[order].float()
    n = correct_sorted.shape[0]
    cum_correct = torch.cumsum(correct_sorted, dim=0)
    return [((k + 1) / n, 1.0 - float(cum_correct[k]) / (k + 1)) for k in range(n)]


def aurc(confidences: torch.Tensor, correct: torch.Tensor) -> float:
    """Trapezoidal area under the risk-coverage curve over coverage in [0, 1],
    extending the curve flat back to coverage=0 (using the risk at the smallest
    observed coverage as a conservative anchor)."""
    curve = risk_coverage_curve(confidences, correct)
    xs = [0.0] + [c for c, _ in curve]
    ys = [curve[0][1]] + [r for _, r in curve]
    return sum((ys[i] + ys[i + 1]) / 2 * (xs[i + 1] - xs[i]) for i in range(len(xs) - 1))
