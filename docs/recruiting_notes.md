# Recruiting Notes

This repository is meant to support conversations for roles such as:

- Financial ML Engineer
- Production ML Engineer
- Quant Developer
- Research Engineer
- Trading Systems Engineer
- MLOps Engineer in fintech

## Positioning

```text
Production ML engineer focused on financial time-series systems: validation, risk control, serving, monitoring, and experiment tracking.
```

Expanded AI quant positioning:

```text
I build production-style AI quant systems that connect financial data pipelines,
model training, walk-forward validation, risk-aware decisioning, inference serving,
and operational monitoring.
```

## What This Repo Demonstrates

- ability to structure a Python ML project
- leakage-aware validation mindset
- transaction-cost-aware backtesting
- walk-forward evaluation
- risk metric implementation
- model serving contracts
- monitoring and drift reporting
- local experiment tracking
- CI and deployment skeletons

## Portfolio Artifacts

The repo is organized around five hiring artifacts:

1. executable GitHub repository
2. benchmark reports
3. model serving / demo layer
4. technical writing series
5. resume / portfolio / book landing-page material

See `docs/portfolio_artifacts.md` for the execution checklist and completion criteria.

For the broader graduate-school and overseas-career strategy, see `docs/career_strategy.md`.

## What This Repo Does Not Claim

- profitable trading strategy
- investment advice
- production trading system
- access to private market data
- live broker integration

## Interview Talking Points

1. Financial ML is mostly an evaluation problem before it is a modeling problem.
2. The public baseline is intentionally simple and synthetic; the engineering workflow is the artifact.
3. Production strategy details are excluded to protect proprietary research.
4. The same structure can be extended to real data, model training, model registry, and deployment.
5. Monitoring includes drift and rolling performance because model quality can degrade after deployment.

## Interview Pitch

```text
In financial ML, the first problem is not model complexity but evaluation error.
This project starts with leakage-aware walk-forward validation and transaction-cost-aware baselines.
Then it adds the production ML layer around the research workflow: experiment tracking,
model metadata, serving contracts, latency benchmarks, and monitoring reports.
The public repository is intentionally sanitized, so it demonstrates the engineering architecture
without disclosing private strategy parameters or live trading logic.
```
