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

DQN validation selection counts:

- `mean_reversion_full`: `72`
- `no_trade`: `11`
- `mean_reversion_low_risk`: `2`
- `volatility_defensive`: `1`
- `cvar_defensive`: `1`

Rule-based validation selection counts:

- `cvar_defensive`: `74`
- `volatility_defensive`: `13`

## Interpretation

The DQN selector improved validation return and Sharpe versus the current rule-based selector in this single demo split, but it did so mostly by selecting `mean_reversion_full`. That is not yet robust evidence of regime-aware learning.

The training split Sharpe was much higher than validation Sharpe, so this result must be treated as overfit-prone until walk-forward checks are added.

Required next gates:

- walk-forward DQN selector report
- seed stability report
- random selector baseline
- action distribution coverage report
- transaction-cost stress report
- rule-based selector and individual-candidate benchmarks
