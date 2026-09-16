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
"""
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


def _depthwise_kernel(band: str, channels: int, device, dtype) -> torch.Tensor:
    """(C, 1, 2, 2) depthwise kernel: the same fixed 2x2 filter on every channel."""
    k = _HAAR_KERNELS[band].to(device=device, dtype=dtype)
    return k.view(1, 1, 2, 2).expand(channels, 1, 2, 2).contiguous()


def haar_dwt2d(x: torch.Tensor):
    """(B, C, H, W) -> (LL, LH, HL, HH), each (B, C, H/2, W/2). Requires even H, W."""
    b, c, h, w = x.shape
    if h % 2 != 0 or w % 2 != 0:
        raise ValueError(f"haar_dwt2d requires even H, W; got H={h}, W={w}")
    bands = []
    for band in _BAND_ORDER:
        kernel = _depthwise_kernel(band, c, x.device, x.dtype)
        bands.append(F.conv2d(x, kernel, stride=2, groups=c))
    return tuple(bands)  # LL, LH, HL, HH


def haar_idwt2d(ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
    """Inverse of haar_dwt2d: 4x (B, C, H/2, W/2) -> (B, C, H, W)."""
    c = ll.shape[1]
    out = None
    for band, sub in zip(_BAND_ORDER, (ll, lh, hl, hh)):
        kernel = _depthwise_kernel(band, c, sub.device, sub.dtype)
        contribution = F.conv_transpose2d(sub, kernel, stride=2, groups=c)
        out = contribution if out is None else out + contribution
    return out


class HaarDWT(nn.Module):
    """Thin nn.Module wrapper so it composes naturally inside an nn.Sequential-style model."""

    def forward(self, x: torch.Tensor):
        return haar_dwt2d(x)


class HaarIDWT(nn.Module):
    def forward(self, ll: torch.Tensor, lh: torch.Tensor, hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        return haar_idwt2d(ll, lh, hl, hh)
