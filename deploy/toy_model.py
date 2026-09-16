"""deploy/toy_model.py — a small, CPU-friendly model that actually exercises
S-WFA, for the ONNX export/verification/benchmark pipeline (U5).

This is deliberately NOT the paper repo's ConvNeXt-Tiny + WFA production model —
that backbone isn't part of this vNext component set, and this machine has
neither the private dataset nor a GPU to make benchmarking it meaningful. What's
actually in question here is whether *this component's own contribution*
(SWFA, built on HaarDWT/HaarIDWT) survives export and inference through ONNX
Runtime — that's what ToyDefectClassifier exists to prove, honestly scoped to
what this machine can actually demonstrate.
"""
import torch.nn as nn

from models.wfa import SWFA


class ToyDefectClassifier(nn.Module):
    def __init__(self, num_classes: int = 5, in_channels: int = 3,
                 stem_channels: int = 8, wfa_channels: int = 16,
                 reduction: int = 4, hf_kernel: int = 3):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(stem_channels, wfa_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.wfa = SWFA(wfa_channels, reduction=reduction, hf_kernel=hf_kernel)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(wfa_channels, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.wfa(x)
        return self.head(x)
