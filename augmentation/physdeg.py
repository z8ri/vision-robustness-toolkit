"""augmentation/physdeg.py — static PhysDeg: (clean, deg1, deg2) triplet generator.

This is the "static" sampling strategy: each degraded view is a chain of 2-3 types
drawn uniformly from the 7 in-distribution degradations (degradation/params.py),
each at a uniformly drawn severity, composed in physical acquisition order.

`_sample_chain` is the single extension point: component 2 (A-PhysDeg) subclasses
this and overrides only `_sample_chain` to draw from a curriculum-weighted,
difficulty-EMA-weighted distribution instead of uniform — everything else
(the clean/deg1/deg2 triplet construction, the physical-order composition,
the training objective) stays identical, so the two are a controlled, one-variable
comparison rather than two unrelated pipelines.
"""
import random
from typing import Callable, List, Optional, Sequence, Tuple

import torch
from PIL import Image

from degradation.ops import apply_id_degradation
from degradation.params import ID_TYPES, N_SEVERITIES, PHYSICAL_ORDER

Chain = List[Tuple[str, int]]  # [(deg_type, severity_idx), ...] in physical order


class PhysDegTransform:
    """Generates a stacked (3, C, H, W) tensor: clean, deg1, deg2.

    Args:
        base_transform: standard preprocessing (resize/crop + to-tensor + normalize).
        deg_types: pool of in-distribution degradation type names to sample from.
        chain_range: (min, max) number of degradation types per chain.
        rng: optional random.Random instance for reproducibility; a fresh
            `random.Random()` is created per call otherwise (module-level `random`
            state is shared with everything else in the process, which makes
            multi-worker DataLoader determinism awkward — an explicit rng avoids
            that footgun).
    """

    def __init__(
        self,
        base_transform: Callable[[Image.Image], torch.Tensor],
        deg_types: Sequence[str] = ID_TYPES,
        chain_range: Tuple[int, int] = (2, 3),
        rng: Optional[random.Random] = None,
    ):
        unknown = set(deg_types) - set(ID_TYPES)
        if unknown:
            raise ValueError(f"Unknown in-distribution degradation type(s): {unknown}")
        self.base_transform = base_transform
        self.deg_types = list(deg_types)
        self.chain_range = chain_range
        self.rng = rng or random.Random()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        clean = self.base_transform(img)
        deg1 = self.base_transform(self._apply_chain(img, self._sample_chain()))
        deg2 = self.base_transform(self._apply_chain(img, self._sample_chain()))
        return torch.stack([clean, deg1, deg2], dim=0)  # (3, C, H, W)

    # ---- extension point (A-PhysDeg overrides only this) ----
    def _sample_chain(self) -> Chain:
        """Static/uniform sampling: draw n types uniformly, severity uniformly."""
        n = min(self.rng.randint(*self.chain_range), len(self.deg_types))
        selected = self.rng.sample(self.deg_types, n)
        selected.sort(key=lambda t: PHYSICAL_ORDER.index(t))
        return [(t, self.rng.randrange(N_SEVERITIES)) for t in selected]

    def _apply_chain(self, img: Image.Image, chain: Chain) -> Image.Image:
        for deg_type, severity in chain:
            img = apply_id_degradation(img, deg_type, severity, seed=self.rng.randrange(2**31))
        return img
