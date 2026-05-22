import pytest

from quant_ml_lab.benchmarking import benchmark_latency


def test_benchmark_latency_returns_percentiles():
    report = benchmark_latency(lambda: 1 + 1, iterations=5)

    assert report.iterations == 5
    assert report.p50_ms >= 0
    assert report.p95_ms >= report.p50_ms
    assert report.max_ms >= report.p99_ms


def test_benchmark_latency_rejects_non_positive_iterations():
    with pytest.raises(ValueError):
        benchmark_latency(lambda: None, iterations=0)
