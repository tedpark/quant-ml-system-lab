# Strategy Candidate: Pair RL Sizer

This is the first public strategy-candidate pipeline in the lab.

## Strategy Contract

The public strategy candidate is:

```text
pair mean reversion signal
-> forward-only HMM regime features
-> SAC position sizing overlay
-> validation gates
-> paper-trading signal
```

It is a research strategy candidate, not a live trading system.

## What It Does

- Builds a baseline pair mean-reversion signal.
- Fits HMM emission parameters only on the training split.
- Uses forward-only regime probabilities on later data.
- Trains SAC only as a risk/sizing overlay.
- Provides richer RL state features: z-score, HMM probability, regime transition, baseline position, spread momentum, spread volatility, recent baseline PnL, and baseline drawdown.
- Penalizes turnover, high-volatility exposure, and drawdown in the public reward.
- Runs multiple seeds.
- Selects the best validation policy by validation Sharpe.
- Saves the selected policy checkpoint.
- Emits a latest target pair signal.
- Marks the signal as approved only if infrastructure and risk gates pass.
- Reports regime-conditioned policy behavior.

## Command

```bash
make pair-rl-strategy
```

Output:

```text
reports/pair_rl_strategy.json
artifacts/strategy_checkpoints/pair_rl_seed_*.pt
```

Walk-forward robustness check:

```bash
make pair-rl-strategy-walk-forward
```

Output:

```text
reports/pair_rl_strategy_walk_forward.json
```

## Gates

Infrastructure gates:

- validation rows are sufficient
- all seed metrics are finite
- best checkpoint exists
- position bounds stay within `[-1, 1]`

Risk gates:

- validation drawdown is within the configured limit
- validation trade count is sufficient
- validation Sharpe beats the baseline when required
- regime response is large enough when required

If any required gate fails, `trade_ready` is false and the latest signal is not approved for paper trading.

## Regime Behavior Analysis

The strategy report includes a `regime_behavior` section. It answers the question:

```text
Did the learned policy behave differently in normal and high-volatility regimes?
```

The report compares:

- rows by regime
- mean HMM high-volatility probability
- mean baseline absolute position
- mean SAC absolute position
- mean SAC multiplier
- regime-specific return
- regime-specific Sharpe
- regime-specific drawdown
- regime-specific turnover
- regime-specific trade count

The field `learned_regime_response` classifies the policy as:

- `defensive_sizing_in_high_vol`
- `aggressive_sizing_in_high_vol`
- `neutral_or_mixed_sizing`

This makes the RL behavior auditable. If the policy increases high-volatility exposure, the report should show it directly instead of hiding it behind aggregate Sharpe.

Important distinction:

- `mean_sac_abs_position` can increase simply because the baseline signal is active more often in that regime.
- `mean_active_multiplier` is the better measure of whether SAC itself changed sizing when a baseline position existed.
- A strategy should not claim regime-specific learning unless the active multiplier shift is economically meaningful and robust across splits.

The current public gate requires a minimum absolute active multiplier shift before treating the policy as regime-responsive.

## Robustness Status

The single split can pass `trade_ready`, but that is not enough. The walk-forward report includes separate robustness gates:

- trade-ready rate across folds
- mean Sharpe delta
- mean total return delta
- majority of positive return-delta folds
- majority of positive Sharpe-delta folds

The latest local walk-forward result is not robust-ready. It reduced losses versus baseline across folds, but Sharpe improvement was not consistent.

## Public Boundary

This pipeline deliberately excludes:

- production universe selection
- private feature recipes
- proprietary thresholds
- live broker routing
- order execution logic
- capital allocation
- real PnL claims

The point is to make the strategy workflow real without leaking strategy edge.
