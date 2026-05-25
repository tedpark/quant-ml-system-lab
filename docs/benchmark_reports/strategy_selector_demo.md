# Strategy Selector Demo

This report summarizes the public regime-aware strategy selector demo.

Command:

```bash
make strategy-selector-demo
```

Output:

```text
reports/strategy_selector_demo.json
```

## Purpose

The selector is a modular scaffold for the next RL architecture:

```text
regime model
-> strategy family
-> meta-controller
-> risk budget and execution constraints
-> walk-forward validation
```

The current implementation is deliberately rule-based. It is not a live strategy and not yet a learned RL meta-controller.

## Candidate Family

- `no_trade`
- `mean_reversion_full`
- `mean_reversion_low_risk`
- `volatility_defensive`
- `cvar_defensive`

## Demo Result

Baseline metrics:

- total return: `-0.008486678752498311`
- Sharpe: `-0.04672005285178897`
- max drawdown: `-0.11935204698174917`
- turnover: `11.0`
- trades: `11`

Selected policy metrics:

- total return: `-0.020134249591297726`
- Sharpe: `-0.2738501462822829`
- max drawdown: `-0.04317433598481957`
- turnover: `5.964368253400552`
- trades: `23`

Selection counts:

- `volatility_defensive`: `118`
- `mean_reversion_low_risk`: `54`
- `no_trade`: `45`
- `cvar_defensive`: `11`

## Interpretation

The scaffold reduced drawdown but did not improve risk-adjusted return. This is the correct kind of public result for the current stage: it validates the module boundary and audit output without claiming alpha.

The learned RL version should be required to beat this selector and the individual candidate strategies under walk-forward validation before any paper-trading claim.
