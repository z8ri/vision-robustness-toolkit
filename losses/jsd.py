"""losses/jsd.py — three-way Jensen-Shannon consistency loss over
(clean, deg1, deg2), the term `train.py` pairs with CE(clean).

JSD(clean, deg1, deg2) = (1/3) * sum_i KL(p_i || M), where M is the mean of
the three softmax distributions (AugMix-style). `F.kl_div(log_M, p_i)`
computes exactly `KL(p_i || M)` when called as `kl_div(input=log-prob,
target=prob)`, so the three-term sum below is a direct implementation of that
definition — not an approximation of it.
"""
import torch
import torch.nn.functional as F


def jsd_loss(logits_clean: torch.Tensor, logits_deg1: torch.Tensor, logits_deg2: torch.Tensor,
             eps: float = 1e-8) -> torch.Tensor:
    if not (logits_clean.shape == logits_deg1.shape == logits_deg2.shape):
        raise ValueError(
            f"clean/deg1/deg2 logits must share a shape, got "
            f"{tuple(logits_clean.shape)}, {tuple(logits_deg1.shape)}, {tuple(logits_deg2.shape)}"
        )
    p_clean = F.softmax(logits_clean, dim=1)
    p_deg1 = F.softmax(logits_deg1, dim=1)
    p_deg2 = F.softmax(logits_deg2, dim=1)
    log_mixture = ((p_clean + p_deg1 + p_deg2) / 3.0).clamp(min=eps).log()
    return (
        F.kl_div(log_mixture, p_clean, reduction="batchmean")
        + F.kl_div(log_mixture, p_deg1, reduction="batchmean")
        + F.kl_div(log_mixture, p_deg2, reduction="batchmean")
    ) / 3.0
