# Serving Latency

This report measures the local deterministic demo prediction contract. It does not benchmark a production model or checkpoint.

## Protocol

- Endpoint shape: `POST /predict`
- Measurement target: local `demo_predict` function
- Iterations: 1000
- Production checkpoints: excluded

## Result

| scope | iterations | p50_ms | p95_ms | p99_ms | max_ms |
| --- | --- | --- | --- | --- | --- |
| demo_predict | 1000 | 0.002750 | 0.002917 | 0.003291 | 0.177125 |

## Limitations

- Local timings vary by machine and current system load.
- This is a schema/contract benchmark, not a live model benchmark.
