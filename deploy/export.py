"""deploy/export.py — ONNX export with a dynamic batch axis by default."""
import warnings
from typing import Sequence

import torch
import torch.nn as nn


def export_to_onnx(
    model: nn.Module,
    dummy_input: torch.Tensor,
    path: str,
    opset: int = 17,
    input_names: Sequence[str] = ("input",),
    output_names: Sequence[str] = ("output",),
    dynamic_batch: bool = True,
) -> None:
    model.eval()
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {input_names[0]: {0: "batch"}, output_names[0]: {0: "batch"}}
    with warnings.catch_warnings():
        # torch.onnx.export's TracerWarnings here are about static shape-assertion
        # branches (e.g. HaarDWT's channel/even-size checks) being baked in as
        # constants, which is the correct, intended behavior for fixed
        # architectural invariants — not a sign of a broken trace.
        warnings.simplefilter("ignore", category=torch.jit.TracerWarning)
        torch.onnx.export(
            model, dummy_input, path,
            export_params=True,
            opset_version=opset,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_axes=dynamic_axes,
        )
