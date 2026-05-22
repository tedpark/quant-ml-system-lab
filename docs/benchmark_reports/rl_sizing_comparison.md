# RL Sizing Comparison

This report compares sanitized position-sizing policies under one shared synthetic pair test split.

## Protocol

- Dataset: synthetic pair data
- Shared signal: same baseline mean-reversion positions
- Shared cost model: transaction-cost-aware turnover
- RL role: position sizing / risk multiplier only
- Public boundary: deterministic proxies are used instead of private trained checkpoints

## Result

| policy | total_return | sharpe | sortino | max_drawdown | win_rate | turnover | mean_multiplier | min_multiplier | max_multiplier |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | -0.190815 | -1.669204 | -2.124536 | -0.255905 | 0.432990 | 5.000000 | 1.000000 | 1.000000 | 1.000000 |
| sac_proxy | -0.124408 | -1.570104 | -2.033796 | -0.174161 | 0.432990 | 10.478977 | 0.516913 | 0.250000 | 0.928861 |
| ppo_proxy | -0.117612 | -1.453074 | -1.798901 | -0.174878 | 0.432990 | 15.088365 | 0.484888 | 0.250000 | 1.000000 |
| qrdqn_cvar_proxy | -0.114110 | -1.844515 | -2.134858 | -0.150250 | 0.432990 | 3.646987 | 0.709410 | 0.371915 | 1.000000 |

## Policy Descriptions

- `base`: Unscaled baseline position.
- `sac_proxy`: Continuous tanh-style sizing proxy based on signal strength.
- `ppo_proxy`: Clipped sizing proxy based on signal strength.
- `qrdqn_cvar_proxy`: Distributional-RL-style CVaR proxy using rolling lower-tail returns.

## Limitations

- The public repo uses deterministic sizing proxies instead of private trained policies.
- This report validates comparison structure and metrics, not live trading performance.
- A production comparison would add fixed seeds, checkpoints, and full training logs.
