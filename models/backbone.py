"""models/backbone.py — ConvNeXt-Tiny with S-WFA inserted into Stage 3.

Matches the real integration point this project's S-WFA is meant for: Stage 3
of `torchvision.models.convnext_tiny` (`.features[5]`) is 9 `CNBlock`s at 384
channels. An attention module goes *after* individual blocks within that one
stage — not scattered across all four stages — specifically after block index
4 and block index 8 (the last block). This was confirmed directly against the
installed torchvision build (0.23.0): `features[5]` does have exactly 9
children, and running a dummy `(1,3,224,224)` input through `features[:5]`
does land on `(1, 384, 14, 14)` — checked, not assumed, before writing this.

This module has never been run against the real private dataset (not on this
machine) or a GPU. It's built to the real architectural convention so that a
GPU + the real dataset directory is the only thing missing to actually train
it, not the wiring itself.
"""
from typing import Sequence

import torch
import torch.nn as nn
from torchvision.models import convnext_tiny

from .wfa import SWFA

STAGE3_CHANNELS = 384
DEFAULT_WFA_POSITIONS = (4, 8)


class ConvNeXtTinySWFA(nn.Module):
    def __init__(
        self,
        num_classes: int,
        wfa_positions: Sequence[int] = DEFAULT_WFA_POSITIONS,
        wfa_reduction: int = 32,
        wfa_hf_kernel: int = 3,
        pretrained: bool = False,
    ):
        super().__init__()
        base = convnext_tiny(weights="IMAGENET1K_V1" if pretrained else None)
        stage3_blocks = list(base.features[5].children())
        if not stage3_blocks:
            raise RuntimeError("torchvision's convnext_tiny Stage 3 (features[5]) has no blocks")
        if any(p < 0 or p >= len(stage3_blocks) for p in wfa_positions):
            raise ValueError(
                f"wfa_positions {tuple(wfa_positions)} out of range for Stage 3's "
                f"{len(stage3_blocks)} blocks (valid: 0..{len(stage3_blocks) - 1})"
            )

        self.features_pre_stage3 = nn.Sequential(*list(base.features[:5]))
        self.stage3_blocks = nn.ModuleList(stage3_blocks)
        self.features_post_stage3 = nn.Sequential(*list(base.features[6:]))
        self.wfa_positions = set(wfa_positions)
        self.wfa_modules = nn.ModuleDict({
            str(p): SWFA(STAGE3_CHANNELS, reduction=wfa_reduction, hf_kernel=wfa_hf_kernel)
            for p in self.wfa_positions
        })

        self.avgpool = base.avgpool
        head_in_features = base.classifier[2].in_features
        self.classifier = nn.Sequential(
            base.classifier[0],  # LayerNorm2d over the pooled 768-channel feature
            base.classifier[1],  # Flatten
            nn.Linear(head_in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features_pre_stage3(x)
        for i, block in enumerate(self.stage3_blocks):
            x = block(x)
            if i in self.wfa_positions:
                x = self.wfa_modules[str(i)](x)
        x = self.features_post_stage3(x)
        x = self.avgpool(x)
        return self.classifier(x)
