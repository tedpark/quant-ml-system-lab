# Pair RL Strategy Candidate Report

This report documents the latest local run of the public strategy-candidate pipeline.

## Command

```bash
make pair-rl-strategy
```

## Latest Local Result

Generated report:

```text
reports/pair_rl_strategy.json
```

Latest run summary:

- Strategy: `pair_mean_reversion_hmm_sac_sizer`
- Best seed: 11
- Best checkpoint: `artifacts/strategy_checkpoints/pair_rl_seed_11.pt`
- Baseline validation total return: 0.009170647738297122
- Strategy validation total return: 0.007170995580799255
- Total return delta: -0.0019996521574978665
- Baseline validation Sharpe: 0.1341620041952139
- Strategy validation Sharpe: 0.20684762813321383
- Sharpe delta: 0.07268562393799993
- Baseline max drawdown: -0.07140219072869236
- Strategy max drawdown: -0.03564709099961416
- Drawdown delta: 0.0357550997290782
- `trade_ready`: true

Latest signal:

```json
{
  "date": "2022-11-29",
  "baseline_position": 1.0,
  "sized_position": 0.5025628805160522,
  "leg_a_target": -0.5025628805160522,
  "leg_b_target": 0.5025628805160522,
  "high_vol_prob": 4.709740889115226e-15,
  "approved_for_paper_trading": true
}
```

## Interpretation

The strategy candidate passed the configured paper-trading gates in this synthetic run. It did not beat baseline on total return, but it improved validation Sharpe and max drawdown.

That means the current candidate is a risk-adjusted sizing overlay, not an alpha engine. The next useful work is walk-forward strategy selection, stricter cost/slippage modeling, and a paper-trading adapter that emits orders without live execution.

## Boundary

This remains a public research artifact. It does not include live broker integration, real capital allocation, production universe selection, private feature recipes, or claims about future performance.
