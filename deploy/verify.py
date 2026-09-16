"""deploy/verify.py — PyTorch vs. ONNX Runtime output alignment + provider check.

`check_no_fallback` is empirically grounded, not guessed: onnxruntime's
profiling JSON (`SessionOptions.enable_profiling=True`) emits one `*_kernel_time`
event per executed node with `args["provider"]` set to the execution provider
that actually ran it — verified directly against this onnxruntime version before
writing this function (fence/session bookkeeping events don't carry a
`provider` field, so filtering on `"provider" in args` is how the real kernel
executions are picked out from the bookkeeping noise).
"""
import json
import os
from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


@dataclass(frozen=True)
class ExportComparison:
    max_abs_diff: float
    mean_abs_diff: float
    argmax_agreement_rate: float
    n_samples: int


def compare_pytorch_onnx(
    model: nn.Module,
    onnx_path: str,
    inputs: torch.Tensor,
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> ExportComparison:
    import onnxruntime as ort

    model.eval()
    with torch.no_grad():
        torch_out = model(inputs).numpy()

    session = ort.InferenceSession(onnx_path, providers=list(providers))
    input_name = session.get_inputs()[0].name
    onnx_out = session.run(None, {input_name: inputs.numpy()})[0]

    if torch_out.shape != onnx_out.shape:
        raise ValueError(f"shape mismatch: PyTorch {torch_out.shape} vs ONNX {onnx_out.shape}")

    diff = np.abs(torch_out - onnx_out)
    agreement = float((torch_out.argmax(axis=1) == onnx_out.argmax(axis=1)).mean())
    return ExportComparison(
        max_abs_diff=float(diff.max()),
        mean_abs_diff=float(diff.mean()),
        argmax_agreement_rate=agreement,
        n_samples=int(inputs.shape[0]),
    )


@dataclass(frozen=True)
class ProviderUsageReport:
    requested_providers: Tuple[str, ...]
    used_providers: Dict[str, int]  # provider name -> executed-kernel count
    no_fallback: bool  # True iff every executed kernel ran on requested_providers[0]


def check_no_fallback(
    onnx_path: str,
    inputs: torch.Tensor,
    providers: Sequence[str] = ("CPUExecutionProvider",),
) -> ProviderUsageReport:
    """Runs one inference with ORT profiling enabled and inspects which
    execution provider actually ran each node — catching the case where a
    requested (e.g. GPU) provider silently falls back to CPU for unsupported ops.
    The scenario this exists for (requesting CUDAExecutionProvider and catching
    a fallback) isn't testable on this machine (no GPU); tests here exercise the
    same code path with CPUExecutionProvider, where the interesting assertion is
    that `used_providers` and `no_fallback` faithfully reflect what ran, not that
    a fallback was caught (there is none to catch without a second provider)."""
    import onnxruntime as ort

    if not providers:
        raise ValueError("providers must be non-empty")

    so = ort.SessionOptions()
    so.enable_profiling = True
    session = ort.InferenceSession(onnx_path, sess_options=so, providers=list(providers))
    input_name = session.get_inputs()[0].name
    session.run(None, {input_name: inputs.numpy()})
    profile_path = session.end_profiling()

    try:
        with open(profile_path) as f:
            events = json.load(f)
    finally:
        if os.path.exists(profile_path):
            os.remove(profile_path)

    used: Dict[str, int] = {}
    for event in events:
        provider = event.get("args", {}).get("provider")
        if provider:
            used[provider] = used.get(provider, 0) + 1

    primary = providers[0]
    no_fallback = len(used) > 0 and set(used.keys()) == {primary}
    return ProviderUsageReport(requested_providers=tuple(providers), used_providers=used, no_fallback=no_fallback)
