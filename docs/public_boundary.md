# Public Boundary

This project is designed as a public portfolio artifact. It intentionally separates engineering demonstration from production trading research.

## Public

- synthetic data generation
- simplified baseline strategy
- time-ordered validation
- walk-forward evaluation
- CVaR/risk-control utilities
- monitoring metrics
- local experiment tracking
- manifest-only model registry
- FastAPI demo API
- tests and CI

## Private

- production asset universe
- production features
- production thresholds
- model checkpoints
- broker execution paths
- live trading configuration
- real account state
- raw live performance records
- private parameter search spaces

## Rationale

Quantitative trading systems should not publish proprietary alpha. The public repo therefore demonstrates reproducible engineering patterns without disclosing live strategy logic.

This boundary is intentional and should be preserved in future changes.
