# Benchmarks

This repository uses synthetic or sample data only. Benchmark outputs are intended to validate system behavior, not to advertise a live trading strategy.

## Sample Pair Baseline

Command:

```bash
make sample-backtest
```

Current sample output:

```text
dataset: synthetic_pair
train total_return: -0.2302
train sharpe: -0.9646
train max_drawdown: -0.3164
test total_return: -0.1908
test sharpe: -1.6692
test max_drawdown: -0.2559
```

Interpretation:

- This is not a profitable strategy claim.
- The baseline exists to verify time splits, position generation, transaction costs, and metric computation.
- Future examples can compare regime filters and risk controls against this baseline without exposing production logic.

## Disclosure

Production universes, feature recipes, thresholds, checkpoints, live broker paths, and raw performance records are intentionally excluded.

## Walk-Forward Demo

Command:

```bash
make walk-forward
```

Purpose:

- verify time-ordered folds
- separate train and test windows
- report fold-level metrics
- provide a safe benchmark artifact without production parameters

## Monitoring Demo

Command:

```bash
make monitoring-report
```

Purpose:

- produce drift metrics on synthetic/sample features
- compute rolling performance metrics from the public baseline
- show production ML monitoring patterns without exposing live telemetry

## Latency Demo

Command:

```bash
make latency-benchmark
```

Purpose:

- measure local demo prediction contract latency
- report p50/p95/p99/max
- avoid benchmarking any production model or checkpoint
