"""deploy/export.py — ONNX export with a dynamic batch axis by default.

Explicitly pins `dynamo=False` (the legacy TorchScript-based tracer). PyTorch
2.9 flips `torch.onnx.export`'s default to the newer `torch.export`/`onnxscript`
-based exporter, a genuinely different code path this project has never
exercised — every claim in this repo about export correctness (the HaarDWT/
HaarIDWT buffer fix, the PyTorch-vs-ONNX-Runtime agreement tests, the
clean/ID/OOD regression check) was verified against the legacy tracer
specifically. Pinning it in code, rather than relying on whatever a given
`pip install torch` happens to default to, keeps "verified" meaning what it
says instead of silently starting to mean "probably still fine on a code path
nobody has run." (Discovered the hard way: this repo's own CI failed on a
newer torch with `ModuleNotFoundError: No module named 'onnxscript'`, since
the new exporter is an optional extra this project doesn't depend on.)
"""
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
            dynamo=False,
        )
