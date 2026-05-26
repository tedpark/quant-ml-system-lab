# Strategy Candidate Benchmark

This report decomposes the public strategy family on multi-regime synthetic data.
It identifies which simple candidate policies are hard baselines for a learned
allocator to beat.

## Summary

- cases: `3`
- mean selected Sharpe: `-0.9164348741317294`
- mean selected minus best Sharpe: `-0.9164348741317294`
- worst selected minus best Sharpe: `-1.0888504377611108`
- selected matches best cases: `0`
- strongest candidate by mean Sharpe: `no_trade`
- weakest regime counts: `{'calm_mean_reverting': 2, 'slow_reversion': 1}`
- benchmark-ready: `False`

## Dataset Cases

| dataset | selected_sharpe | best_candidate | best_candidate_sharpe | selected_minus_best_sharpe | weakest_regime |
| --- | --- | --- | --- | --- | --- |
| regime_seed_501 | -0.632546 | no_trade | 0.000000 | -0.632546 | slow_reversion |
| regime_seed_502 | -1.088850 | no_trade | 0.000000 | -1.088850 | calm_mean_reverting |
| regime_seed_503 | -1.027909 | no_trade | 0.000000 | -1.027909 | calm_mean_reverting |

## Candidate Averages

| candidate | mean_sharpe | mean_total_return | best_count |
| --- | --- | --- | --- |
| no_trade | 0.000000 | 0.000000 | 3 |
| mean_reversion_full | -0.889875 | -0.181651 | 0 |
| mean_reversion_low_risk | -0.887674 | -0.090856 | 0 |
| volatility_defensive | -1.148242 | -0.159622 | 0 |
| cvar_defensive | -0.988996 | -0.090194 | 0 |

## Regime Decomposition

| dataset | regime | selected_sharpe | selected_total_return | selected_trades |
| --- | --- | --- | --- | --- |
| regime_seed_501 | calm_mean_reverting | -0.442343 | -0.017217 | 25 |
| regime_seed_501 | slow_reversion | -0.787267 | -0.040770 | 21 |
| regime_seed_501 | trend_moderate | 0.000000 | 0.000000 | 0 |
| regime_seed_502 | calm_mean_reverting | -1.278084 | -0.044877 | 30 |
| regime_seed_502 | slow_reversion | -0.968146 | -0.044420 | 24 |
| regime_seed_502 | trend_moderate | 0.000000 | 0.000000 | 0 |
| regime_seed_503 | calm_mean_reverting | -1.459341 | -0.046749 | 23 |
| regime_seed_503 | slow_reversion | -0.796610 | -0.040679 | 11 |
| regime_seed_503 | trend_moderate | 0.000000 | 0.000000 | 0 |

## Interpretation

If a simple candidate repeatedly beats the selector or RL allocator, the next model
should not be tuned until the benchmark gap is explained. This protects the lab from
mistaking model complexity for strategy quality.
