# Strategy Selector DQN Demo

This report summarizes the learned discrete strategy-selector scaffold.

Command:

```bash
make strategy-selector-dqn-demo
```

Output:

```text
reports/strategy_selector_dqn_demo.json
artifacts/strategy_checkpoints/strategy_selector_dqn.pt
```

## Purpose

The DQN selector is the first learned version of the meta-controller contract:

```text
state -> strategy choice
```

The action space is deliberately discrete and public:

- `no_trade`
- `mean_reversion_full`
- `mean_reversion_low_risk`
- `volatility_defensive`
- `cvar_defensive`

This is still not a live strategy. It is a training scaffold for testing whether a learned selector can beat simple rule-based selection under held-out validation and later walk-forward gates.

## Demo Result

Rows:

- train rows: `159`
- validation rows: `87`

DQN validation metrics:

- total return: `-0.0007502419429273877`
- Sharpe: `-0.024163421476573885`
- max drawdown: `-0.0580629834875529`
- turnover: `9.0`
- trades: `9`

Rule-based selector validation metrics:

- total return: `-0.016180094534256173`
- Sharpe: `-0.6856862476669808`
- max drawdown: `-0.0556060554082175`
- turnover: `5.607587765579333`
- trades: `17`

Random selector validation metrics:

- total return: `-0.05254006287906887`
- Sharpe: `-1.5893581968590802`
- max drawdown: `-0.09927946438334212`
- turnover: `17.747141774085883`
- trades: `26`

DQN validation selection counts:

- `mean_reversion_full`: `72`
- `no_trade`: `11`
- `mean_reversion_low_risk`: `2`
- `volatility_defensive`: `1`
- `cvar_defensive`: `1`

Rule-based validation selection counts:

- `cvar_defensive`: `74`
- `volatility_defensive`: `13`

Training diagnostics:

- loss tail mean: `2.0426711434993193e-05`
- q-value tail mean: `0.026724759340286255`
- q-value tail absolute mean: `0.02748905373737216`
- validation action concentration: `0.8275862068965517`
- beats random Sharpe: `true`
- beats rule-based Sharpe: `true`
- q-values are bounded: `true`
- validation is not single-action by the current gate: `true`

## Interpretation

The DQN selector improved validation return and Sharpe versus the current rule-based selector in this single demo split, but it did so mostly by selecting `mean_reversion_full`. That is not yet robust evidence of regime-aware learning.

The training split Sharpe was much higher than validation Sharpe, so this result must be treated as overfit-prone until walk-forward checks are added.

The validation action concentration is close to the current failure threshold. This means the model is not freely discovering a balanced regime policy yet; it is near a one-action collapse and must be monitored in every future run.

Required next gates:

- walk-forward DQN selector report
- seed stability report
- random selector baseline
- action distribution coverage report
- transaction-cost stress report
- rule-based selector and individual-candidate benchmarks
