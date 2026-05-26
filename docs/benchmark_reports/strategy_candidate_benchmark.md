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
- benchmark-ready: `False`

## Dataset Cases

| dataset | selected_sharpe | best_candidate | best_candidate_sharpe | selected_minus_best_sharpe |
| --- | --- | --- | --- | --- |
| regime_seed_501 | -0.632546 | no_trade | 0.000000 | -0.632546 |
| regime_seed_502 | -1.088850 | no_trade | 0.000000 | -1.088850 |
| regime_seed_503 | -1.027909 | no_trade | 0.000000 | -1.027909 |

## Candidate Averages

| candidate | mean_sharpe | mean_total_return | best_count |
| --- | --- | --- | --- |
| no_trade | 0.000000 | 0.000000 | 3 |
| mean_reversion_full | -0.889875 | -0.181651 | 0 |
| mean_reversion_low_risk | -0.887674 | -0.090856 | 0 |
| volatility_defensive | -1.148242 | -0.159622 | 0 |
| cvar_defensive | -0.988996 | -0.090194 | 0 |

## Interpretation

If a simple candidate repeatedly beats the selector or RL allocator, the next model
should not be tuned until the benchmark gap is explained. This protects the lab from
mistaking model complexity for strategy quality.
