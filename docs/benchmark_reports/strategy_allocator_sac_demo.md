# Strategy Allocator SAC Demo

This report summarizes the SAC-only continuous strategy allocator scaffold.

Command:

```bash
make strategy-allocator-sac-demo
```

Output:

```text
reports/strategy_allocator_sac_demo.json
artifacts/strategy_checkpoints/strategy_allocator_sac.pt
```

## Purpose

The SAC allocator replaces discrete strategy selection with continuous allocation across the public strategy family:

```text
state -> continuous SAC action -> softmax strategy weights -> weighted position
```

The strategy family is:

- `no_trade`
- `mean_reversion_full`
- `mean_reversion_low_risk`
- `volatility_defensive`
- `cvar_defensive`

This is not a live strategy. It is a scaffold for learning risk allocation without forcing the agent into one discrete action.

## Demo Result

Rows:

- train rows: `159`
- validation rows: `87`

SAC validation metrics:

- total return: `-0.01970259896837645`
- Sharpe: `-0.9180280393066089`
- max drawdown: `-0.05414547018742388`
- turnover: `4.295274177875963`
- trades: `36`

Rule-based selector validation metrics:

- total return: `-0.016180094534256173`
- Sharpe: `-0.6856862476669808`
- max drawdown: `-0.0556060554082175`
- turnover: `5.607587765579333`
- trades: `17`

Validation mean weights:

- `no_trade`: `0.2749402455360753`
- `mean_reversion_full`: `0.20174930521576054`
- `mean_reversion_low_risk`: `0.12479226927418932`
- `volatility_defensive`: `0.17336365825201586`
- `cvar_defensive`: `0.22515452172195888`

Training diagnostics:

- actor loss tail mean: `-0.6892110526561737`
- critic loss tail mean: `0.0017581223743036388`
- alpha final: `0.007218679878860712`
- alpha loss tail mean: `-38.505323600769046`
- validation weight concentration: `0.21261754789544707`
- beats rule-based Sharpe: `false`

## Interpretation

SAC is the better architectural fit for continuous risk allocation, but this demo does not beat the rule-based selector on validation Sharpe.

The useful result is that SAC did not collapse into one action. It learned a diversified allocation across the strategy family. The weaker validation Sharpe means the reward, state, and validation protocol need more work before SAC can be treated as a stronger result.

Required next gates:

- walk-forward SAC allocator report
- multi-seed SAC allocator report
- transaction-cost stress report
- reward ablation report
- comparison against the best individual candidate, not only the rule-based selector
