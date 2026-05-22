# AI Quant Strategy

This document explains the intended direction of this lab: not "AI predicts stocks", but a production-style financial ML system that connects validation, model training, risk-aware decisioning, serving, and monitoring.

## Thesis

The strongest direction for this project is:

```text
Regime Detection
  + Pair Trading
  + Distributional RL
  + CVaR Risk Control
  + Production Serving
```

The goal is not to build a black-box model that issues buy/sell calls. The goal is to build a system that:

1. detects market regimes,
2. generates constrained trading candidates,
3. uses ML/RL to rank, filter, or size those candidates,
4. controls tail risk with distribution-aware metrics,
5. validates the full workflow with walk-forward evaluation,
6. exposes the model through a serving interface,
7. monitors drift, latency, and rolling quality.

## Target Role Fit

This repository is best aligned with roles such as:

- Financial ML Engineer
- AI Quant Developer
- Production Quant ML Engineer
- ML Engineer, Trading Systems
- Quant Developer with ML focus
- Research Engineer in fintech or trading systems

The positioning is deliberately engineering-heavy:

```text
I build production-style AI quant systems that connect financial data pipelines,
model training, walk-forward validation, risk-aware decisioning, inference serving,
and operational monitoring.
```

## The Five AI Quant Problems

### 1. Prediction

Questions:

- What is the next-period return distribution?
- Is volatility increasing?
- Which regime is the market in?

Candidate models:

- LightGBM / XGBoost / CatBoost
- HMM
- Kalman filters
- temporal deep learning models

Principle:

Prediction accuracy is not enough. The model must improve a transaction-cost-aware decision process.

### 2. Signal Generation

Questions:

- When should a strategy enter?
- When should it exit?
- Which pairs or assets are valid candidates?
- Which regimes invalidate the signal?

Candidate approaches:

- mean-reversion baseline
- z-score spread signal
- HMM regime filter
- Kalman spread model
- ranking model
- meta-labeling

Principle:

The model should constrain or improve a trading workflow, not replace all domain logic with a black box.

### 3. Position Sizing

Questions:

- How much exposure should the strategy take?
- Should exposure be reduced in high-tail-risk states?
- Should the trade be vetoed?

Candidate approaches:

- SAC for continuous sizing
- PPO as a stable policy baseline
- QR-DQN / distributional RL for return quantiles
- CVaR / Expected Shortfall
- Kelly-style sizing

Principle:

RL is most defensible here when it is constrained to sizing or risk multipliers, not unconstrained trade generation.

### 4. Strategy Selection

Questions:

- Should the system use mean reversion, momentum, or stay flat?
- Which model is reliable in the current regime?

Candidate approaches:

- regime-aware ensembles
- contextual bandits
- Thompson Sampling
- model routing

Principle:

The system should model when a strategy is likely to fail, not just when it works.

### 5. Operations and Risk Monitoring

Questions:

- Is the model distribution drifting?
- Is latency increasing?
- Is rolling performance degrading?
- Are drawdowns outside the expected range?

Candidate components:

- model registry
- serving schemas
- latency benchmarks
- PSI / KS drift checks
- rolling PnL and risk reports
- alerting hooks

Principle:

Production ML credibility comes from the lifecycle around the model, not from the model alone.

## What This Project Avoids

This repository intentionally avoids common but weak AI-trading claims:

- "LSTM predicts stock prices"
- "Transformer predicts tomorrow's close"
- "RL learns to trade end-to-end"
- "Backtest return is 300%"
- "AI recommends buy/sell"
- "Accuracy proves the model works"

For financial ML, validation quality matters more than headline returns.

## Recommended Build Order

### 1. Validation First

Before adding models, establish the evaluation protocol:

- time-ordered train/test splits
- walk-forward validation
- look-ahead-bias checks
- transaction costs
- slippage assumptions
- baseline comparisons
- failure-case reporting

### 2. Baselines

Start with non-ML baselines:

- no-trade baseline
- equal-weight baseline
- z-score mean reversion
- simple momentum
- simple pair trading

The model must beat a clear baseline after costs.

### 3. Regime Filter

Add regime-aware gating:

- HMM or volatility regime model
- regime-specific performance report
- high-volatility risk reduction
- drawdown-by-regime analysis

The key question:

```text
When does this strategy work, and when does it fail?
```

### 4. ML Ranking

Before deep learning, use fast and interpretable tabular models:

- LightGBM
- XGBoost
- CatBoost
- feature importance
- SHAP analysis

Use these for candidate ranking, meta-labeling, or filter scoring.

### 5. RL for Sizing

Use RL in a constrained role:

```text
regime model + pair signal -> trade candidate
RL policy -> position size / risk multiplier
CVaR module -> reduce / floor / veto
execution layer -> order candidate
```

Compare SAC, PPO, and QR-DQN under the same environment, split, and cost assumptions.

### 6. Production Layer

Make the system portfolio-grade:

- FastAPI-style inference
- model manifest or registry
- versioned model metadata
- schema validation
- hot-reload pattern
- latency benchmark
- monitoring report

## System Shape

```text
Market Data
  -> Data Pipeline
  -> Feature Construction
  -> Baseline Signals
  -> Regime Detection
  -> Pair / Asset Ranking
  -> RL Position Sizing
  -> CVaR Risk Control
  -> Walk-Forward Validation
  -> Model Registry
  -> Inference API
  -> Monitoring / Alerting
```

## Portfolio Evidence

The most useful public artifacts are:

1. a reproducible benchmark report,
2. an executable GitHub repository,
3. technical writing that explains tradeoffs and failures.

### Benchmark Report

Include:

- baseline vs regime-filtered strategy
- SAC vs PPO vs QR-DQN
- CVaR sizing before/after
- transaction-cost sensitivity
- drawdown and tail-risk analysis
- latency and model-size measurements

Metrics:

- Sharpe
- Sortino
- maximum drawdown
- win rate
- turnover
- transaction-cost impact
- CVaR 5% / 10%
- hit rate by regime
- p95 inference latency

### Executable Repo

The project should remain easy to verify:

```bash
make test
make sample-backtest
make walk-forward
make experiment-demo
make latency-benchmark
make serve
```

### Technical Writing

Recommended articles:

1. `Why Financial ML Fails Without Walk-Forward Validation`
2. `Regime-Aware Pair Trading with HMM`
3. `QR-DQN and CVaR for Tail-Risk-Aware Position Sizing`
4. `SAC vs PPO vs QR-DQN for Trading Position Sizing`
5. `Serving RL Policies with FastAPI and MLflow`

## Interview Story

Weak story:

```text
I used AI to predict stock prices.
```

Strong story:

```text
In financial ML, the first problem is not model complexity but evaluation error.
I started with leakage-aware walk-forward validation and transaction-cost-aware baselines.
Then I added regime detection to identify when a strategy should be active.
I constrained RL to position sizing rather than unconstrained trading decisions.
For tail risk, I used distributional outputs and CVaR-aware sizing.
Finally, I connected the workflow to serving, registry metadata, latency benchmarks,
and monitoring reports so the model could be evaluated as a production system.
```

## Boundary

This public lab is not a live trading system and does not publish proprietary strategy details. It is a sanitized engineering artifact that demonstrates how a quant ML system can be structured, tested, evaluated, served, and monitored.
