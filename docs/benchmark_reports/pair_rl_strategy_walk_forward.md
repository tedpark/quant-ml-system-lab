# Pair RL Strategy Walk-Forward Report

This report checks whether the public pair RL strategy candidate survives multiple time splits.

## Command

```bash
make pair-rl-strategy-walk-forward
```

Output:

```text
reports/pair_rl_strategy_walk_forward.json
```

## Latest Local Result

Summary:

- Folds: 3
- Trade-ready folds: 1
- Trade-ready rate: 0.3333333333333333
- Mean total return: -0.04189361046460364
- Mean Sharpe: -1.9624507129976634
- Worst max drawdown: -0.08068158427427996
- Mean total return delta vs baseline: 0.03844859587141414
- Mean Sharpe delta vs baseline: -0.13335791895013735
- Positive return-delta folds: 3
- Positive Sharpe-delta folds: 1
- Mean active multiplier shift: 0.011643322620120355

Robustness gates:

- Trade-ready rate >= 50%: false
- Mean Sharpe delta positive: false
- Mean total return delta positive: true
- Positive return-delta majority: true
- Positive Sharpe-delta majority: false

Result:

```text
robust_ready = false
```

## Interpretation

The single split result improved after adding richer state features and drawdown-aware reward. The walk-forward result is more cautious:

- The strategy reduced losses versus the baseline in every fold.
- It did not improve Sharpe consistently.
- Only one of three folds passed the full trade-ready gates.
- The average active multiplier shift was small.

This means the current model is a promising risk-reduction overlay, but it is not robust enough to call a stable regime-policy strategy yet.

## Next Work

The next improvement should target robustness, not another single-split gain:

- add fold-level model selection
- test more synthetic market regimes
- add stricter slippage/cost stress
- compare against deterministic CVaR and volatility-sizing baselines
- require walk-forward gates before paper-trading approval
