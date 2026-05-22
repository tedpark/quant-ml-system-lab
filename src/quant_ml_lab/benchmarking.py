from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np


@dataclass(frozen=True)
class LatencyReport:
    iterations: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "iterations": self.iterations,
            "p50_ms": self.p50_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "max_ms": self.max_ms,
        }


def benchmark_latency(fn, iterations: int = 500) -> LatencyReport:
    if iterations <= 0:
        raise ValueError("iterations must be positive")

    timings: list[float] = []
    for _ in range(iterations):
        start = perf_counter()
        fn()
        timings.append((perf_counter() - start) * 1000.0)

    values = np.asarray(timings, dtype=float)
    return LatencyReport(
        iterations=iterations,
        p50_ms=float(np.percentile(values, 50)),
        p95_ms=float(np.percentile(values, 95)),
        p99_ms=float(np.percentile(values, 99)),
        max_ms=float(values.max()),
    )
