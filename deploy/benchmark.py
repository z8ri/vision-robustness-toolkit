"""deploy/benchmark.py — batch=1 latency, model size, and a CPU memory proxy.

Backend-agnostic on purpose: `benchmark_latency` takes a zero-argument
`predict_fn` closure, so the exact same function times a PyTorch model
(`lambda: model(x)`), an ONNX Runtime CPU session, or — once real hardware is
available — a CUDA session, without this module needing to know which.

Numbers produced by running this on this development machine are CPU numbers,
not the RTX 4090 / CUDA EP figures the vNext result account names as a future
target — this module proves the *measurement mechanism* works, not that those
specific target numbers have been reproduced here.
"""
import os
import platform
import resource
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class LatencyStats:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    throughput_qps: float
    n_iters: int
    n_warmup: int


def benchmark_latency(predict_fn: Callable[[], None], n_warmup: int = 10, n_iters: int = 100) -> LatencyStats:
    if n_iters <= 0:
        raise ValueError(f"n_iters must be positive, got {n_iters}")
    if n_warmup < 0:
        raise ValueError(f"n_warmup must be non-negative, got {n_warmup}")

    for _ in range(n_warmup):
        predict_fn()

    times_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        predict_fn()
        t1 = time.perf_counter()
        times_ms.append((t1 - t0) * 1000.0)

    times_sorted = sorted(times_ms)

    def _pct(p: float) -> float:
        idx = min(len(times_sorted) - 1, int(p * len(times_sorted)))
        return times_sorted[idx]

    mean_ms = sum(times_ms) / len(times_ms)
    return LatencyStats(
        p50_ms=_pct(0.50), p95_ms=_pct(0.95), p99_ms=_pct(0.99),
        mean_ms=mean_ms,
        throughput_qps=(1000.0 / mean_ms) if mean_ms > 0 else float("inf"),
        n_iters=n_iters, n_warmup=n_warmup,
    )


def model_size_mb(path: str) -> float:
    return os.path.getsize(path) / (1024 ** 2)


def peak_memory_mb(fn: Callable[[], None]) -> float:
    """A coarse CPU memory proxy: process peak resident-set size (RSS) after
    running `fn`, via `resource.getrusage`. `ru_maxrss` is a *process-lifetime*
    high-water mark (monotonically non-decreasing), not scoped to just `fn` —
    so this is only meaningful as a before/after delta or a relative comparison
    across calls in the same process, never as an exact per-call figure. macOS
    reports `ru_maxrss` in bytes, Linux in KB; handled below."""
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    fn()
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    unit_bytes = 1 if platform.system() == "Darwin" else 1024
    delta = max(0, after - before)  # ru_maxrss is a monotonic high-water mark; delta==0 is a valid "no new peak"
    return delta * unit_bytes / (1024 ** 2)
