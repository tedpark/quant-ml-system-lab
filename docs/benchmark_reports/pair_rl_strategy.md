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
- Learned regime response: `neutral_or_mixed_sizing`

Regime behavior:

| regime | rows | active_rows | mean_high_vol_prob | mean_sac_abs_position | mean_active_multiplier | total_return | sharpe | max_drawdown | trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 40 | 24 | 0.03532591365515776 | 0.3019430189335253 | 0.5032383648892088 | -0.030739827311168977 | -2.069416450883833 | -0.05006033067001314 | 27 |
| high_vol | 46 | 41 | 0.9603616153996503 | 0.4532663151723049 | 0.5085426950713665 | 0.039113154507111725 | 2.0692775020019494 | -0.031196437151460477 | 41 |

Regime interpretation:

- High-volatility rows had higher average absolute SAC exposure, but the baseline signal was also active more often in high-volatility rows.
- Conditional on an active baseline signal, SAC used almost the same multiplier in both regimes: normal `0.5032`, high-vol `0.5085`.
- The active multiplier shift is only `0.0053`, so the learned behavior is classified as neutral or mixed, not meaningfully aggressive or defensive.
- In this synthetic run, high-volatility rows produced better return and Sharpe, but the current SAC policy mostly acted like a roughly 50% sizing overlay rather than a strong regime router.

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

That means the current candidate is a risk-adjusted sizing overlay, not an alpha engine. It also does not yet show strong regime-specific policy behavior. The next useful work is walk-forward strategy selection, stricter cost/slippage modeling, a richer reward/state design, and a paper-trading adapter that emits orders without live execution.

## Boundary

This remains a public research artifact. It does not include live broker integration, real capital allocation, production universe selection, private feature recipes, or claims about future performance.
