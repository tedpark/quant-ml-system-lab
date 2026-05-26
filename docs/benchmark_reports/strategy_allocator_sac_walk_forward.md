# Strategy Allocator SAC Walk-Forward

This report summarizes walk-forward validation for the SAC continuous strategy allocator.

Command:

```bash
make strategy-allocator-sac-walk-forward
```

Output:

```text
reports/strategy_allocator_sac_walk_forward.json
```

## Purpose

The goal is to test whether the SAC allocator survives time-split validation better than a single train/validation split.

Each fold:

- fits the HMM regime model only on past data
- trains SAC on a later train sub-window
- validates on the next unseen walk-forward window
- compares SAC against the rule-based selector

## Summary

- folds: `3`
- mean validation Sharpe: `0.6084243033462294`
- mean rule-based Sharpe: `0.5807367743726034`
- mean Sharpe delta: `0.027687528973625853`
- mean total return: `0.02844080906462998`
- mean rule-based total return: `0.021053766802069756`
- mean total return delta: `0.007387042262560224`
- positive Sharpe delta folds: `1`
- positive return delta folds: `1`
- mean weight concentration: `0.20283660692982594`
- robust-ready: `false`

## Interpretation

The SAC allocator slightly improved mean Sharpe and mean total return, but it only beat the rule-based selector in `1 / 3` folds. This is not robust enough.

The useful result is diagnostic:

- SAC is not collapsing into one strategy.
- The allocator is learning diversified weights.
- The improvement is not stable across market windows.
- The current dataset and reward design are still too weak for a tradable claim.

Next required work:

- multi-seed walk-forward
- multi-pair synthetic generator
- transaction-cost stress
- reward ablation
- offline-RL action coverage diagnostics
