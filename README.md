# Quant ML System Lab

[![CI](https://github.com/tedpark/quant-ml-system-lab/actions/workflows/ci.yml/badge.svg)](https://github.com/tedpark/quant-ml-system-lab/actions/workflows/ci.yml)

Public companion project for financial ML engineering.

This repository is intentionally **sanitized**. It demonstrates the engineering architecture of a financial ML system using synthetic or sample data. Production strategy parameters, live trading rules, broker integrations, private datasets, model checkpoints, and proprietary alpha research are intentionally excluded.

## Purpose

This lab focuses on the engineering layer around quantitative ML systems:

- walk-forward validation
- leakage-aware evaluation
- baseline strategy comparison
- regime-aware modeling patterns
- distributional RL risk-control concepts
- RL fundamentals labs
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

## Repository Structure

```text
quant-ml-system-lab/
  src/
    quant_ml_lab/
      api.py
      benchmarking.py
      data.py
      experiments.py
      monitoring.py
      regime.py
      rl.py
      sac.py
      risk.py
      serving.py
      sizing.py
      validation.py
      walk_forward.py
  tests/
  examples/
  docs/
    ai_quant_strategy.md
    architecture.md
    benchmarks.md
    benchmark_reports/
    career_strategy.md
    deployment.md
    experiments.md
    learning_lab.md
    monitoring.md
    portfolio_artifacts.md
    public_boundary.md
    recruiting_notes.md
    serving.md
    validation.md
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
make q-learning-gridworld
make sac-bandit
make all-reports
```

The examples write local reports to:

```text
reports/sample_backtest.json
reports/walk_forward.json
reports/monitoring_report.json
reports/experiment_demo.json
reports/latency_benchmark.json
reports/baseline_vs_regime_filter.json
reports/cvar_sizing.json
reports/rl_sizing_comparison.json
reports/q_learning_gridworld.json
reports/sac_bandit.json
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
8. Build an ML/AI/RL learning lab with reproducible experiments. In progress.

## Positioning

This project is designed to demonstrate production ML engineering skills for financial time-series systems:

```text
financial data -> features -> model training -> walk-forward evaluation -> risk control -> serving -> monitoring
```

It is a portfolio and research-engineering artifact, not investment advice and not a trading recommendation.

See also:

- [AI Quant Strategy](docs/ai_quant_strategy.md)
- [Career Strategy](docs/career_strategy.md)
- [Portfolio Artifact Plan](docs/portfolio_artifacts.md)
- [ML / AI / RL Learning Lab](docs/learning_lab.md)
- [Benchmark Reports](docs/benchmark_reports/)
- [Recruiting Notes](docs/recruiting_notes.md)
- [Public / Private Boundary](docs/public_boundary.md)

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
- RL fundamentals lab: implemented
- SAC concept lab: implemented
