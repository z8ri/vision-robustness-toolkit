"""Synthetic-data helpers shared by tests. No real dataset needed to run these."""
import numpy as np
from PIL import Image


def make_synthetic_image(size=(64, 64), mean=128.0, seed=0) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.clip(rng.normal(mean, 20.0, size=size), 0, 255).astype(np.uint8)
    return Image.fromarray(arr)  # 2D uint8 -> mode "L"


def make_synthetic_samples(n_per_class=3, n_classes=2, size=(64, 64)):
    """Returns a list of (PIL.Image, label) — class k images are brighter by
    k * 40, so a trivial mean-brightness threshold can separate them for the
    macro-F1 stub used in eval-protocol tests."""
    samples = []
    idx = 0
    for label in range(n_classes):
        for _ in range(n_per_class):
            img = make_synthetic_image(size=size, mean=80.0 + 40.0 * label, seed=idx)
            samples.append((img, label))
            idx += 1
    return samples
