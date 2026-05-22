# Reproducibility

This repository is designed to be cloned, installed, tested, and run without private data.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## Core Verification

```bash
make test
```

## Generate Reports

```bash
make all-reports
```

This writes execution artifacts to `reports/` and generated Markdown benchmark summaries to `docs/benchmark_reports/`.

## Public Data Boundary

All examples use synthetic/sample data. The repo does not include:

- production asset universe
- private feature recipes
- live broker execution code
- trained production checkpoints
- raw live-trading performance records

## Determinism

Synthetic data examples use fixed seeds where a report depends on generated data. Latency results may vary by machine and current system load.

## Expected Outputs

After `make all-reports`, the following files should exist:

```text
reports/sample_backtest.json
reports/walk_forward.json
reports/baseline_vs_regime_filter.json
reports/cvar_sizing.json
reports/rl_sizing_comparison.json
reports/monitoring_report.json
reports/experiment_demo.json
reports/latency_benchmark.json
docs/benchmark_reports/baseline_vs_regime_filter.md
docs/benchmark_reports/cvar_sizing.md
docs/benchmark_reports/rl_sizing_comparison.md
docs/benchmark_reports/serving_latency.md
```
