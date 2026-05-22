# Baseline vs Regime Filter

This report compares a synthetic mean-reversion pair baseline against a public high-volatility regime filter.

## Protocol

- Dataset: synthetic pair data
- Split: first 70% train, final 30% test
- Transaction cost: 2.00 bps per turnover unit
- Regime threshold: estimated on the train split only
- Regime proxy: rolling spread volatility
- High-volatility exposure multiplier: 0.35

## Result

High-volatility share in test: 0.173077

| variant | total_return | sharpe | sortino | max_drawdown | win_rate | turnover | trades |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | -0.190815 | -1.669204 | -2.124536 | -0.255905 | 0.432990 | 5.000000 | 5 |
| regime_filtered | -0.190983 | -1.744430 | -2.029259 | -0.252890 | 0.432990 | 8.250000 | 12 |

## Limitations

- Synthetic data does not represent a production universe.
- The regime filter is a public volatility proxy, not a proprietary HMM model.
- No parameter search is performed in this public example.
