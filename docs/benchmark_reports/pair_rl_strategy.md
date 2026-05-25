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
- Strategy validation total return: 0.010779211741699601
- Total return delta: 0.0016085640034024795
- Baseline validation Sharpe: 0.1341620041952139
- Strategy validation Sharpe: 0.31331942713430083
- Sharpe delta: 0.17915742293908693
- Baseline max drawdown: -0.07140219072869236
- Strategy max drawdown: -0.037391711317835585
- Drawdown delta: 0.034010479410856775
- `trade_ready`: true
- Learned regime response: `aggressive_sizing_in_high_vol`

Regime behavior:

| regime | rows | active_rows | mean_high_vol_prob | mean_sac_abs_position | mean_active_multiplier | total_return | sharpe | max_drawdown | trades |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| normal | 40 | 24 | 0.03532591365515776 | 0.2743015220970847 | 0.4571692034951411 | -0.02706606692239355 | -2.02504513432481 | -0.04457508600761184 | 27 |
| high_vol | 46 | 41 | 0.9603616153996503 | 0.4653825935219293 | 0.5221365683416768 | 0.03889809716511783 | 1.9807230097225488 | -0.035827588216041595 | 41 |

Regime interpretation:

- High-volatility rows had higher average absolute SAC exposure, but the baseline signal was also active more often in high-volatility rows.
- Conditional on an active baseline signal, SAC used a lower multiplier in normal rows and a higher multiplier in high-volatility rows: normal `0.4572`, high-vol `0.5221`.
- The active multiplier shift is `0.0650`, so the learned behavior is classified as aggressive high-volatility sizing.
- The improved state and reward design produced a clearer regime-conditioned policy than the prior near-constant overlay.

Latest signal:

```json
{
  "date": "2022-11-29",
  "baseline_position": 1.0,
  "sized_position": 0.451859787106514,
  "leg_a_target": -0.451859787106514,
  "leg_b_target": 0.451859787106514,
  "high_vol_prob": 4.709740889115226e-15,
  "approved_for_paper_trading": true
}
```

## Interpretation

The strategy candidate passed the configured paper-trading gates in this synthetic run. It beat baseline on total return, Sharpe, and max drawdown.

That means the current candidate is a risk-adjusted sizing overlay with emerging regime-conditioned behavior, not a standalone alpha engine. The next useful work is walk-forward strategy selection, stricter cost/slippage modeling, and a paper-trading adapter that emits orders without live execution.

## Boundary

This remains a public research artifact. It does not include live broker integration, real capital allocation, production universe selection, private feature recipes, or claims about future performance.
