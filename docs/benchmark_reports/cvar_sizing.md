# CVaR Sizing

This report compares an unscaled synthetic pair baseline with a public distributional-RL-style CVaR sizing proxy.

## Protocol

- Dataset: synthetic pair data
- Baseline: mean-reversion pair signal
- Risk-control proxy: rolling lower-tail empirical CVaR
- Position multiplier: reduced when downside tail risk exceeds target

## Result

| variant | total_return | sharpe | max_drawdown | cvar_5 | cvar_10 | turnover | mean_multiplier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline | -0.190815 | -1.669204 | -0.255905 | -0.028005 | -0.022291 | 5.000000 | 1.000000 |
| qrdqn_cvar_proxy | -0.114110 | -1.844515 | -0.150250 | -0.016253 | -0.013312 | 3.646987 | 0.709410 |

## Limitations

- This is a rolling empirical CVaR proxy, not a trained production QR-DQN checkpoint.
- The example demonstrates the risk-control contract, not alpha generation.
- Synthetic data does not represent a production universe.
