"""degradation/ops.py — pixel-level degradation operators.

Every operator takes a grayscale PIL "L" image, a 0-based severity index
(0..N_SEVERITIES-1), and (for stochastic operators) an integer seed, and returns a
grayscale PIL "L" image of the same size. Stochastic operators are seeded so that
evaluation is reproducible per-image: the same (image index, degradation type,
severity) always yields the same corrupted image.

This module is a fresh implementation of standard, textbook image-degradation
techniques (box/disk-kernel blur, additive/multiplicative noise, JPEG re-encoding,
resize-based downsampling and pixelation, gamma/tone remapping) — it does not copy
code from the frozen paper repository, only reuses the same physical vocabulary.
"""
import io
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageFilter

from .params import ID_PARAMS, OOD_PARAMS

_ArrayFn = Callable[[Image.Image, int, int], Image.Image]


def _to_arr(img: Image.Image) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


def _to_img(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))  # 2D uint8 -> mode "L"


def _conv2d_same(img: Image.Image, kernel: np.ndarray) -> Image.Image:
    """Apply a single 2D kernel via same-padding convolution (grayscale)."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    arr = _to_arr(img)
    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    k = torch.from_numpy(kernel.astype(np.float32)).unsqueeze(0).unsqueeze(0)
    out = F.conv2d(t, k, padding=(pad_h, pad_w))
    return _to_img(out.squeeze(0).squeeze(0).numpy())


# ============================================================
# In-distribution (ID) operators — trained on by PhysDeg.
# ============================================================

def motion_blur(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    """Horizontal box-kernel blur (relative motion between sensor and stock)."""
    length = ID_PARAMS["motion_blur"]["kernel_sizes"][severity]
    if length <= 1:
        return img
    length += 1 - (length % 2)  # force odd
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0 / length
    return _conv2d_same(img, kernel)


def defocus_blur(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    """Disk-kernel (defocus) blur."""
    radius = ID_PARAMS["defocus_blur"]["radii"][severity]
    size = 2 * radius + 1
    yy, xx = np.ogrid[-radius:size - radius, -radius:size - radius]
    disk = (xx * xx + yy * yy <= radius * radius).astype(np.float32)
    disk /= disk.sum()
    return _conv2d_same(img, disk)


def gaussian_noise(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    """Additive Gaussian sensor noise, on the [0, 1] pixel scale."""
    sigma = ID_PARAMS["gaussian_noise"]["sigmas"][severity]
    rng = np.random.default_rng(seed)
    arr = _to_arr(img) / 255.0
    arr = arr + rng.normal(0.0, sigma, size=arr.shape)
    return _to_img(np.clip(arr, 0, 1) * 255.0)


def jpeg(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    """Lossy JPEG compression (block-DCT quantization)."""
    quality = ID_PARAMS["jpeg"]["qualities"][severity]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    out = Image.open(buf).convert("L").copy()
    buf.close()
    return out


def downsample(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    """Anti-aliased resolution loss: downsample then restore to the working size."""
    scale = ID_PARAMS["downsample"]["scales"][severity]
    w, h = img.size
    dw, dh = max(1, int(w * scale)), max(1, int(h * scale))
    down = img.resize((dw, dh), resample=Image.BILINEAR)
    return down.resize((w, h), resample=Image.BILINEAR)


def brightness(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    """Additive illumination shift."""
    offset = ID_PARAMS["brightness"]["offsets"][severity] * 255.0
    return _to_img(_to_arr(img) + offset)


def contrast(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    """Mean-centered contrast reduction (sensor response gain)."""
    factor = ID_PARAMS["contrast"]["factors"][severity]
    arr = _to_arr(img)
    m = arr.mean()
    return _to_img((arr - m) * factor + m)


ID_OPS: Dict[str, _ArrayFn] = {
    "motion_blur": motion_blur,
    "defocus_blur": defocus_blur,
    "gaussian_noise": gaussian_noise,
    "jpeg": jpeg,
    "downsample": downsample,
    "brightness": brightness,
    "contrast": contrast,
}


def apply_id_degradation(img: Image.Image, deg_type: str, severity: int, seed: int = 0) -> Image.Image:
    return ID_OPS[deg_type](img, severity, seed)


# ============================================================
# Out-of-distribution (OOD) operators — never applied during training.
# ============================================================

def gaussian_blur(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    radius = OOD_PARAMS["gaussian_blur"]["radius"][severity]
    return img.filter(ImageFilter.GaussianBlur(radius))


def glass_blur(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    """Gaussian blur + local random pixel displacement ("frosted glass")."""
    sigma, delta = OOD_PARAMS["glass_blur"]["sigma_delta"][severity]
    blurred = img.filter(ImageFilter.GaussianBlur(sigma))
    a = np.asarray(blurred, dtype=np.uint8)
    h, w = a.shape
    rng = np.random.default_rng(seed + 101)
    dy = rng.integers(-delta, delta + 1, size=(h, w))
    dx = rng.integers(-delta, delta + 1, size=(h, w))
    ys = np.clip(np.arange(h)[:, None] + dy, 0, h - 1)
    xs = np.clip(np.arange(w)[None, :] + dx, 0, w - 1)
    a = a[ys, xs]
    return Image.fromarray(a).filter(ImageFilter.GaussianBlur(sigma))


def zoom_blur(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    """Average of progressively zoomed-in center crops."""
    zmax = OOD_PARAMS["zoom_blur"]["zmax"][severity]
    w, h = img.size
    acc = _to_arr(img).copy()
    count = 1
    z = 1.02
    while z <= zmax + 1e-6:
        zw, zh = int(round(w * z)), int(round(h * z))
        zoomed = img.resize((zw, zh), Image.BILINEAR)
        left, top = (zw - w) // 2, (zh - h) // 2
        cropped = zoomed.crop((left, top, left + w, top + h))
        acc += _to_arr(cropped)
        count += 1
        z += 0.02
    return _to_img(acc / count)


def gamma(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    """Non-linear tone response (camera/display gamma); g>1 darkens midtones."""
    g = OOD_PARAMS["gamma"]["g"][severity]
    arr = _to_arr(img) / 255.0
    return _to_img(np.power(arr, g) * 255.0)


def low_light(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    """Multiplicative under-exposure."""
    gain = OOD_PARAMS["low_light"]["gain"][severity]
    return _to_img(_to_arr(img) * gain)


def speckle_noise(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    c = OOD_PARAMS["speckle_noise"]["c"][severity]
    arr = _to_arr(img) / 255.0
    rng = np.random.default_rng(seed + 211)
    arr = arr + arr * rng.normal(0.0, c, size=arr.shape)
    return _to_img(np.clip(arr, 0, 1) * 255.0)


def impulse_noise(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    amount = OOD_PARAMS["impulse_noise"]["amount"][severity]
    a = np.asarray(img).copy()
    rng = np.random.default_rng(seed + 307)
    n = int(amount * a.size)
    ys = rng.integers(0, a.shape[0], n); xs = rng.integers(0, a.shape[1], n)
    a[ys, xs] = 255
    ys = rng.integers(0, a.shape[0], n); xs = rng.integers(0, a.shape[1], n)
    a[ys, xs] = 0
    return Image.fromarray(a)


def pixelate(img: Image.Image, severity: int, seed: int = 0) -> Image.Image:
    scale = OOD_PARAMS["pixelate"]["scale"][severity]
    w, h = img.size
    small = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BOX)
    return small.resize((w, h), Image.NEAREST)


OOD_OPS: Dict[str, _ArrayFn] = {
    "gaussian_blur": gaussian_blur,
    "glass_blur": glass_blur,
    "zoom_blur": zoom_blur,
    "gamma": gamma,
    "low_light": low_light,
    "speckle_noise": speckle_noise,
    "impulse_noise": impulse_noise,
    "pixelate": pixelate,
}


def apply_ood_degradation(img: Image.Image, deg_type: str, severity: int, seed: int = 0) -> Image.Image:
    return OOD_OPS[deg_type](img, severity, seed)
