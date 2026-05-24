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
- Runs multiple seeds.
- Selects the best validation policy by validation Sharpe.
- Saves the selected policy checkpoint.
- Emits a latest target pair signal.
- Marks the signal as approved only if infrastructure and risk gates pass.

## Command

```bash
make pair-rl-strategy
```

Output:

```text
reports/pair_rl_strategy.json
artifacts/strategy_checkpoints/pair_rl_seed_*.pt
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

If any required gate fails, `trade_ready` is false and the latest signal is not approved for paper trading.

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
