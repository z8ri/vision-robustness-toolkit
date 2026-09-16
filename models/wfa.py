"""models/wfa.py — WFACore (original WFA) + SpectralGate + S-WFA.

Implements WFA_VNEXT_DESIGN.md exactly:

    x -> Haar DWT -> LL channel attention + HF spatial attention
                     + one-directional HF->LL cross-gate -> IDWT -> R_WFA(x)

    E_b = mean(z_b^2) for b in {LL, LH, HL, HH}      (raw DWT sub-bands of x)
    p_b = (E_b + eps) / (sum_j E_j + 4*eps)
    s(x) = [log p_LL, log p_LH, log p_HL, log p_HH]
    g(x) = sigmoid(MLP(s(x))),  MLP: 4 -> gate_hidden -> 1

    y = x + g(x) * gamma * R_WFA(x)     (gamma: per-channel LayerScale, init 1e-6)

`WFACore` is the "material validation" original WFA computation (reimplemented,
not copied) — it never changes between original WFA and S-WFA. `SpectralGate` is
the one new piece S-WFA adds. `SWFA` composes both and reduces to *exactly* the
original WFA when `use_gate=False` (or when the gate is forced to 1.0) and to an
exact identity when the gate is forced to 0.0 — both are asserted in
tests/test_wfa.py, not just claimed in prose.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn

from .wavelet import haar_dwt2d, haar_idwt2d


class WFACore(nn.Module):
    """Original WFA: LL channel attention + HF spatial attention + HF->LL cross-gate.

    Returns the reconstruction R_WFA(x), *not* combined with any residual or
    LayerScale — that combination is `SWFA`'s job, so this module is reusable by
    both the "original WFA" baseline and the S-WFA candidate without duplication.
    """

    def __init__(self, channels: int, reduction: int = 32, hf_kernel: int = 3):
        super().__init__()
        # Clamp to >=1 so a small `channels` with the default reduction=32 still
        # builds a valid (if less compressive) bottleneck instead of raising.
        hidden = max(1, channels // reduction)

        # Low-frequency channel attention (SE-style gate on the LL band).
        self.ll_fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.ll_fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

        # High-frequency spatial attention, shared across LH/HL/HH.
        self.hf_fuse = nn.Conv2d(3 * channels, channels, kernel_size=1)
        self.hf_bn = nn.BatchNorm2d(channels)
        self.hf_conv = nn.Conv2d(channels, 1, kernel_size=hf_kernel, padding=hf_kernel // 2)

        # One-directional HF -> LL cross-band gate.
        self.cross_gate = nn.Conv2d(2 * channels, channels, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(
        self, x: torch.Tensor, return_raw_bands: bool = False
    ):
        ll, lh, hl, hh = haar_dwt2d(x)

        # LL channel attention.
        s = self.sigmoid(self.ll_fc2(self.relu(self.ll_fc1(self.gap(ll)))))  # (B, C, 1, 1)
        ll_refined = s * ll

        # HF spatial attention (one shared map applied to all three detail bands).
        hf_cat = torch.cat([lh, hl, hh], dim=1)  # (B, 3C, H/2, W/2)
        a = self.sigmoid(self.hf_conv(self.relu(self.hf_bn(self.hf_fuse(hf_cat)))))  # (B, 1, H/2, W/2)
        lh_r, hl_r, hh_r = a * lh, a * hl, a * hh

        # One-directional HF -> LL cross-band gate.
        hf_avg = (lh_r + hl_r + hh_r) / 3.0
        gate_in = self.gap(torch.cat([ll_refined, hf_avg], dim=1))  # (B, 2C, 1, 1)
        g = self.sigmoid(self.cross_gate(gate_in))  # (B, C, 1, 1)
        ll_final = g * ll_refined

        u = haar_idwt2d(ll_final, lh_r, hl_r, hh_r)
        if return_raw_bands:
            return u, (ll, lh, hl, hh)
        return u


class SpectralGate(nn.Module):
    """The one thing S-WFA adds: a sample-level scalar g(x) in (0, 1) from the
    normalized log-energy proportions of x's four raw Haar sub-bands."""

    def __init__(self, hidden: int = 8, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mlp = nn.Sequential(
            nn.Linear(4, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )

    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        energies = torch.stack(
            [b.pow(2).mean(dim=(1, 2, 3)) for b in (ll, lh, hl, hh)], dim=1
        )  # (B, 4)
        proportions = (energies + self.eps) / (energies.sum(dim=1, keepdim=True) + 4 * self.eps)
        log_proportions = torch.log(proportions)  # (B, 4)
        g = torch.sigmoid(self.mlp(log_proportions))  # (B, 1)
        return g.view(-1, 1, 1, 1)


class SWFA(nn.Module):
    """y = x + g(x) * gamma * R_WFA(x).

    `use_gate=False` reproduces the original WFA exactly (g(x) == 1 always,
    matching `WFA_VNEXT_DESIGN.md` §3: "y = gamma*u + z" with no sample gate) —
    use the `original_wfa()` factory below for that case, so the intent is
    explicit at the call site rather than a boolean flag a reader has to look up.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 32,
        hf_kernel: int = 3,
        gate_hidden: int = 8,
        layerscale_init: float = 1e-6,
        use_gate: bool = True,
    ):
        super().__init__()
        self.core = WFACore(channels, reduction=reduction, hf_kernel=hf_kernel)
        self.use_gate = use_gate
        self.gate = SpectralGate(hidden=gate_hidden) if use_gate else None
        self.gamma = nn.Parameter(torch.full((channels,), float(layerscale_init)))

    def forward(
        self,
        x: torch.Tensor,
        force_gate: Optional[float] = None,
        return_gate: bool = False,
    ):
        u, raw_bands = self.core(x, return_raw_bands=True)

        if force_gate is not None:
            g = x.new_full((x.shape[0], 1, 1, 1), float(force_gate))
        elif self.use_gate:
            g = self.gate(*raw_bands)
        else:
            g = x.new_ones((x.shape[0], 1, 1, 1))

        gamma = self.gamma.view(1, -1, 1, 1)
        y = x + g * gamma * u
        return (y, g) if return_gate else y


def original_wfa(channels: int, reduction: int = 32, hf_kernel: int = 3,
                  layerscale_init: float = 1e-6) -> SWFA:
    """Original WFA (material validation baseline): SWFA with the sample gate
    disabled, i.e. g(x) == 1 for every sample — the g(x)->1 endpoint of S-WFA."""
    return SWFA(channels, reduction=reduction, hf_kernel=hf_kernel,
                layerscale_init=layerscale_init, use_gate=False)
