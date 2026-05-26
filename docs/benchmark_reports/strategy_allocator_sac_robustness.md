# SAC Allocator Robustness Matrix

This report stress-tests the public SAC strategy allocator across synthetic data seeds,
SAC random seeds, and transaction-cost assumptions. It is intended to falsify fragile
RL results before any private research or paper-trading workflow uses the pattern.

## Summary

- cases: `8`
- mean Sharpe delta: `-0.16407816642347073`
- median Sharpe delta: `-0.16017712382241217`
- worst Sharpe delta: `-0.4055403832446698`
- mean total return delta: `0.0012976537192070092`
- worst total return delta: `-0.006128065861054743`
- positive Sharpe case rate: `0.5`
- positive return case rate: `0.5`
- robust case rate: `0.0`
- robust-ready: `False`

## Case Matrix

| dataset | sac_seed | cost_bps | sharpe_delta | return_delta | positive_sharpe_folds | positive_return_folds | robust_ready |
| --- | --- | --- | --- | --- | --- | --- | --- |
| synthetic_seed_404 | 61 | 2.000000 | -0.405540 | -0.006128 | 0 | 0 | False |
| synthetic_seed_404 | 62 | 2.000000 | -0.374419 | -0.004851 | 0 | 0 | False |
| synthetic_seed_404 | 61 | 5.000000 | -0.395135 | -0.005977 | 0 | 0 | False |
| synthetic_seed_404 | 62 | 5.000000 | -0.363932 | -0.004719 | 0 | 0 | False |
| synthetic_seed_405 | 61 | 2.000000 | 0.044200 | 0.008149 | 1 | 2 | False |
| synthetic_seed_405 | 62 | 2.000000 | 0.069400 | 0.008035 | 1 | 2 | False |
| synthetic_seed_405 | 61 | 5.000000 | 0.043578 | 0.007963 | 1 | 2 | False |
| synthetic_seed_405 | 62 | 5.000000 | 0.069224 | 0.007910 | 1 | 2 | False |

## Interpretation

If the robustness gate is false, the correct conclusion is not that SAC is impossible.
The correct conclusion is that the current public environment, data coverage, reward,
and validation design are still insufficient for strategy claims.
