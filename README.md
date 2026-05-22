# Quant ML System Lab

Public companion project for financial ML engineering.

This repository is intentionally **sanitized**. It demonstrates the engineering architecture of a financial ML system using synthetic or sample data. Production strategy parameters, live trading rules, broker integrations, private datasets, model checkpoints, and proprietary alpha research are intentionally excluded.

## Purpose

This lab focuses on the engineering layer around quantitative ML systems:

- walk-forward validation
- leakage-aware evaluation
- baseline strategy comparison
- regime-aware modeling patterns
- distributional RL risk-control concepts
- CVaR-aware position sizing examples
- MLflow-style experiment tracking
- FastAPI-style model serving
- monitoring and drift-report skeletons

The goal is not to publish a live trading strategy. The goal is to show how a production-style quant ML system can be structured, tested, evaluated, and served.

## Public / Private Boundary

### Included

- synthetic/sample OHLCV data
- simplified baseline strategies
- validation and benchmark scaffolding
- model-training skeletons
- serving API examples
- tests
- architecture documentation

### Excluded

- production asset universe
- pair-selection rules
- feature weights
- entry/exit thresholds
- live broker execution code
- production configuration
- private market data cache
- trained production checkpoints
- raw live-trading performance records

## Planned Structure

```text
quant-ml-system-lab/
  data/
    sample/
  src/
    data/
    features/
    validation/
    models/
    risk/
    serving/
    monitoring/
  tests/
  docs/
    architecture.md
    validation.md
    benchmarks.md
    serving.md
```

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
make test
make sample-backtest
make walk-forward
make monitoring-report
make experiment-demo
make latency-benchmark
```

The examples write local reports to:

```text
reports/sample_backtest.json
reports/walk_forward.json
reports/monitoring_report.json
reports/experiment_demo.json
reports/latency_benchmark.json
```

The generated report is intentionally not committed because it is an execution artifact.

Run the sanitized demo API:

```bash
make serve
```

Example request:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"request_id":"demo-1","features":[0.4,0.3,0.2]}'
```

## Roadmap

1. Create a minimal reproducible validation pipeline. Done.
2. Add a synthetic pair-trading baseline. Done.
3. Add walk-forward evaluation examples. Done.
4. Add CVaR-aware risk-control examples. Done.
5. Add model serving and monitoring skeletons. Done.
6. Add local experiment tracking and registry manifests. Done.
7. Publish benchmark reports using non-proprietary data. In progress.

## Positioning

This project is designed to demonstrate production ML engineering skills for financial time-series systems:

```text
financial data -> features -> model training -> walk-forward evaluation -> risk control -> serving -> monitoring
```

It is a portfolio and research-engineering artifact, not investment advice and not a trading recommendation.

## Current Status

- Synthetic pair data generator: implemented
- Time-ordered train/test split: implemented
- Mean-reversion baseline: implemented
- Transaction-cost-aware metrics: implemented
- CVaR utility examples: implemented
- Tests: implemented
- Walk-forward evaluation: implemented
- Serving schemas: implemented
- FastAPI demo API: implemented
- Monitoring metrics and report: implemented
- Local experiment tracking: implemented
- Manifest-only model registry: implemented
- Local latency benchmark: implemented
- Dockerfile and CI workflow: implemented
