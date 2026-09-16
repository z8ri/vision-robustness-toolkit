"""models/wavelet.py — single-level 2D Haar DWT/IDWT via fixed depthwise conv.

Fresh implementation (not copied from the frozen paper repo): the four Haar
analysis filters are the outer products of the 1D orthonormal low/high-pass Haar
filters, applied depthwise (groups=C) with stride 2. Because the Haar basis is
orthonormal, the synthesis (inverse) transform is exactly the transpose of the
analysis transform with the same filters — that identity is what
`haar_idwt2d(*haar_dwt2d(x)) == x` (up to floating point) is testing in
tests/test_wavelet.py.

Only even H, W are supported (ConvNeXt Stage-3 feature maps are always a power of
two in this project's use case); odd sizes raise a clear error rather than
silently padding, since silent padding would break the round-trip identity.

`HaarDWT`/`HaarIDWT` register their conv kernels as buffers at construction time
(channels fixed then, not inferred from the input at every forward call) rather
than building them inline in `forward` — this is not a style preference: the
inline-construction version (a `.expand(channels, ...).contiguous()` built fresh
every forward pass) fails ONNX export with "convolution for kernel of unknown
shape" (found empirically — see deploy/ component's export smoke test), because
the legacy TorchScript exporter can't statically resolve a conv weight that's
recomputed inside the traced graph. Pre-registering the kernel as a buffer is
exactly the "改写成可导出的固定卷积" the vNext design docs call for.
"""
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D orthonormal Haar low/high-pass filters.
_LOW = torch.tensor([1.0, 1.0]) / (2 ** 0.5)
_HIGH = torch.tensor([1.0, -1.0]) / (2 ** 0.5)

# 2D separable filters = outer(row_filter, col_filter), each L2-normalized to 1.
_HAAR_KERNELS = {
    "LL": torch.outer(_LOW, _LOW),
    "LH": torch.outer(_LOW, _HIGH),
    "HL": torch.outer(_HIGH, _LOW),
    "HH": torch.outer(_HIGH, _HIGH),
}
_BAND_ORDER = ("LL", "LH", "HL", "HH")


def _depthwise_kernel(band: str, channels: int) -> torch.Tensor:
    """(C, 1, 2, 2) depthwise kernel: the same fixed 2x2 filter on every channel."""
    k = _HAAR_KERNELS[band]
    return k.view(1, 1, 2, 2).expand(channels, 1, 2, 2).contiguous()


class HaarDWT(nn.Module):
    """(B, C, H, W) -> (LL, LH, HL, HH), each (B, C, H/2, W/2). `channels` is fixed
    at construction (its kernels are registered buffers, an ONNX-export
    requirement — see module docstring); requires even H, W."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        for band in _BAND_ORDER:
            self.register_buffer(f"kernel_{band}", _depthwise_kernel(band, channels))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, c, h, w = x.shape
        if c != self.channels:
            raise ValueError(f"HaarDWT was built for channels={self.channels}, got input with C={c}")
        if h % 2 != 0 or w % 2 != 0:
            raise ValueError(f"HaarDWT requires even H, W; got H={h}, W={w}")
        bands = [
            F.conv2d(x, getattr(self, f"kernel_{band}"), stride=2, groups=c)
            for band in _BAND_ORDER
        ]
        return tuple(bands)  # LL, LH, HL, HH


class HaarIDWT(nn.Module):
    """Inverse of HaarDWT: 4x (B, C, H/2, W/2) -> (B, C, H, W)."""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        for band in _BAND_ORDER:
            self.register_buffer(f"kernel_{band}", _depthwise_kernel(band, channels))

    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        c = ll.shape[1]
        if c != self.channels:
            raise ValueError(f"HaarIDWT was built for channels={self.channels}, got input with C={c}")
        out = None
        for band, sub in zip(_BAND_ORDER, (ll, lh, hl, hh)):
            kernel = getattr(self, f"kernel_{band}")
            contribution = F.conv_transpose2d(sub, kernel, stride=2, groups=c)
            out = contribution if out is None else out + contribution
        return out


def haar_dwt2d(x: torch.Tensor):
    """Free-function convenience wrapper around HaarDWT for direct-tensor use
    (e.g. tests/exploration) where building a persistent module isn't needed.
    Not used inside any exported model's forward path — see the module
    docstring for why that matters."""
    return HaarDWT(x.shape[1]).to(device=x.device, dtype=x.dtype)(x)


def haar_idwt2d(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """Free-function convenience wrapper around HaarIDWT; see haar_dwt2d."""
    return HaarIDWT(ll.shape[1]).to(device=ll.device, dtype=ll.dtype)(ll, lh, hl, hh)
