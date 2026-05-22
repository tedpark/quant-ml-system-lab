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
