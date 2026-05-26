# SAC Allocator Reward Ablation

This report removes one public reward component at a time from the SAC strategy allocator.
The goal is to identify whether reward shaping is improving robustness or hiding a fragile
policy behind hand-tuned penalties.

## Summary

- cases: `10`
- ablations: `5`
- best ablation by Sharpe: `no_drawdown_penalty`
- worst ablation by Sharpe: `no_concentration_penalty`
- best mean Sharpe delta: `-0.1789425443042961`
- full reward mean Sharpe delta: `-0.1791830938487189`
- best minus full Sharpe delta: `0.0002405495444227912`
- mean Sharpe delta: `-0.17923910952356698`
- worst Sharpe delta: `-0.40148032323206045`
- mean total return delta: `0.000780051480848698`
- robust case rate: `0.0`
- robust-ready: `False`

## Case Matrix

| ablation | dataset | sharpe_delta | return_delta | positive_sharpe_folds | positive_return_folds | robust_ready |
| --- | --- | --- | --- | --- | --- | --- |
| full_reward | synthetic_seed_404 | -0.401351 | -0.006013 | 0 | 0 | False |
| no_turnover_penalty | synthetic_seed_404 | -0.400381 | -0.005982 | 0 | 0 | False |
| no_high_vol_penalty | synthetic_seed_404 | -0.401007 | -0.005994 | 0 | 0 | False |
| no_drawdown_penalty | synthetic_seed_404 | -0.401480 | -0.006016 | 0 | 0 | False |
| no_concentration_penalty | synthetic_seed_404 | -0.400862 | -0.005986 | 0 | 0 | False |
| full_reward | synthetic_seed_405 | 0.042985 | 0.007555 | 1 | 2 | False |
| no_turnover_penalty | synthetic_seed_405 | 0.041939 | 0.007552 | 1 | 2 | False |
| no_high_vol_penalty | synthetic_seed_405 | 0.042266 | 0.007532 | 1 | 2 | False |
| no_drawdown_penalty | synthetic_seed_405 | 0.043595 | 0.007599 | 1 | 2 | False |
| no_concentration_penalty | synthetic_seed_405 | 0.041904 | 0.007553 | 1 | 2 | False |

## Interpretation

If removing a penalty improves the result, the current reward is probably over-shaped for
this public environment. If all ablations remain unstable, the bottleneck is more likely
data coverage and environment design than a single reward coefficient.
