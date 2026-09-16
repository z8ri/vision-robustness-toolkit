import time

import pytest

from deploy.benchmark import benchmark_latency, model_size_mb, peak_memory_mb


def test_benchmark_latency_runs_the_requested_iteration_count():
    calls = {"n": 0}

    def predict_fn():
        calls["n"] += 1

    stats = benchmark_latency(predict_fn, n_warmup=5, n_iters=20)
    assert calls["n"] == 25  # 5 warmup + 20 timed
    assert stats.n_iters == 20
    assert stats.n_warmup == 5


def test_percentiles_are_ordered_and_throughput_matches_mean():
    def predict_fn():
        time.sleep(0.0005)

    stats = benchmark_latency(predict_fn, n_warmup=2, n_iters=30)
    assert stats.p50_ms <= stats.p95_ms <= stats.p99_ms
    assert stats.mean_ms > 0
    assert stats.throughput_qps == pytest.approx(1000.0 / stats.mean_ms, rel=1e-6)


def test_benchmark_latency_rejects_invalid_iteration_counts():
    with pytest.raises(ValueError):
        benchmark_latency(lambda: None, n_iters=0)
    with pytest.raises(ValueError):
        benchmark_latency(lambda: None, n_warmup=-1)


def test_model_size_mb_matches_a_known_file_size(tmp_path):
    path = tmp_path / "blob.bin"
    path.write_bytes(b"0" * (2 * 1024 * 1024))  # exactly 2 MiB
    assert model_size_mb(str(path)) == pytest.approx(2.0, rel=1e-6)


def test_peak_memory_mb_is_non_negative_and_does_not_crash():
    def allocate_something():
        _ = [0] * 2_000_000  # a few MB of Python objects

    value = peak_memory_mb(allocate_something)
    assert isinstance(value, float)
    assert value >= 0.0
