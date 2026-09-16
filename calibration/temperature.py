"""calibration/temperature.py — single-scalar temperature scaling (Guo et al. 2017).

Deliberately operates on *pre-computed* (logits, labels) tensors only — this
module never holds a reference to a model, so "冻结模型" (the model must not
move while calibrating) is true by construction: there is nothing here with a
handle to update backbone weights.

`fit_temperature` optimizes T via LBFGS minimizing NLL, parameterized as
T = exp(log_T) so positivity holds without clamping. Softmax(z/T) has the exact
same argmax as softmax(z) for any T > 0 — that invariant (design doc: "temperature
scaling通常不改变argmax") is asserted in tests/test_temperature.py, not just
claimed here.
"""
import torch


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor,
                     max_iter: int = 50, lr: float = 0.01) -> float:
    if logits.ndim != 2:
        raise ValueError(f"logits must be (N, C), got shape {tuple(logits.shape)}")
    if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
        raise ValueError(f"labels must be (N,) matching logits' N={logits.shape[0]}, got {tuple(labels.shape)}")

    logits = logits.detach()
    labels = labels.detach().long()
    log_t = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_t], lr=lr, max_iter=max_iter)
    ce = torch.nn.CrossEntropyLoss()

    def closure():
        optimizer.zero_grad()
        t = log_t.exp()
        loss = ce(logits / t, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_t.exp().item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    return torch.softmax(logits / temperature, dim=-1)
