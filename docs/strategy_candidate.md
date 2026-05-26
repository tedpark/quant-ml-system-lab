# Strategy Candidate: Pair RL Sizer

This is the first public strategy-candidate pipeline in the lab.

## Strategy Contract

The public strategy candidate is:

```text
pair mean reversion signal
-> forward-only HMM regime features
-> SAC position sizing overlay
-> validation gates
-> paper-trading signal
```

It is a research strategy candidate, not a live trading system.

For the research diagnosis and next architecture direction, see `docs/rl_trading_research_notes.md`.

## What It Does

- Builds a baseline pair mean-reversion signal.
- Fits HMM emission parameters only on the training split.
- Uses forward-only regime probabilities on later data.
- Trains SAC only as a risk/sizing overlay.
- Provides richer RL state features: z-score, HMM probability, regime transition, baseline position, spread momentum, spread volatility, recent baseline PnL, and baseline drawdown.
- Penalizes turnover, high-volatility exposure, and drawdown in the public reward.
- Runs multiple seeds.
- Selects the best validation policy by validation Sharpe.
- Saves the selected policy checkpoint.
- Emits a latest target pair signal.
- Marks the signal as approved only if infrastructure and risk gates pass.
- Reports regime-conditioned policy behavior.

## Command

```bash
make pair-rl-strategy
```

Output:

```text
reports/pair_rl_strategy.json
artifacts/strategy_checkpoints/pair_rl_seed_*.pt
```

Walk-forward robustness check:

```bash
make pair-rl-strategy-walk-forward
```

Output:

```text
reports/pair_rl_strategy_walk_forward.json
```

Regime-aware strategy selector scaffold:

```bash
make strategy-selector-demo
```

Output:

```text
reports/strategy_selector_demo.json
```

Learned DQN selector scaffold:

```bash
make strategy-selector-dqn-demo
```

Output:

```text
reports/strategy_selector_dqn_demo.json
artifacts/strategy_checkpoints/strategy_selector_dqn.pt
```

Candidate benchmark decomposition:

```bash
make strategy-candidate-benchmark
```

Output:

```text
reports/strategy_candidate_benchmark.json
docs/benchmark_reports/strategy_candidate_benchmark.md
```

SAC continuous allocator scaffold:

```bash
make strategy-allocator-sac-demo
```

Output:

```text
reports/strategy_allocator_sac_demo.json
artifacts/strategy_checkpoints/strategy_allocator_sac.pt
```

SAC allocator walk-forward:

```bash
make strategy-allocator-sac-walk-forward
```

Output:

```text
reports/strategy_allocator_sac_walk_forward.json
```

SAC allocator robustness matrix:

```bash
make strategy-allocator-sac-robustness
```

Output:

```text
reports/strategy_allocator_sac_robustness.json
docs/benchmark_reports/strategy_allocator_sac_robustness.md
```

SAC allocator reward ablation:

```bash
make strategy-allocator-sac-reward-ablation
```

Output:

```text
reports/strategy_allocator_sac_reward_ablation.json
docs/benchmark_reports/strategy_allocator_sac_reward_ablation.md
```

## Gates

Infrastructure gates:

- validation rows are sufficient
- all seed metrics are finite
- best checkpoint exists
- position bounds stay within `[-1, 1]`

Risk gates:

- validation drawdown is within the configured limit
- validation trade count is sufficient
- validation Sharpe beats the baseline when required
- regime response is large enough when required

If any required gate fails, `trade_ready` is false and the latest signal is not approved for paper trading.

## Regime Behavior Analysis

The strategy report includes a `regime_behavior` section. It answers the question:

```text
Did the learned policy behave differently in normal and high-volatility regimes?
```

The report compares:

- rows by regime
- mean HMM high-volatility probability
- mean baseline absolute position
- mean SAC absolute position
- mean SAC multiplier
- regime-specific return
- regime-specific Sharpe
- regime-specific drawdown
- regime-specific turnover
- regime-specific trade count

The field `learned_regime_response` classifies the policy as:

- `defensive_sizing_in_high_vol`
- `aggressive_sizing_in_high_vol`
- `neutral_or_mixed_sizing`

This makes the RL behavior auditable. If the policy increases high-volatility exposure, the report should show it directly instead of hiding it behind aggregate Sharpe.

Important distinction:

- `mean_sac_abs_position` can increase simply because the baseline signal is active more often in that regime.
- `mean_active_multiplier` is the better measure of whether SAC itself changed sizing when a baseline position existed.
- A strategy should not claim regime-specific learning unless the active multiplier shift is economically meaningful and robust across splits.

The current public gate requires a minimum absolute active multiplier shift before treating the policy as regime-responsive.

## Robustness Status

The single split can pass `trade_ready`, but that is not enough. The walk-forward report includes separate robustness gates:

- trade-ready rate across folds
- mean Sharpe delta
- mean total return delta
- majority of positive return-delta folds
- majority of positive Sharpe-delta folds

The latest local walk-forward result is not robust-ready. It reduced losses versus baseline across folds, but Sharpe improvement was not consistent.

Current decision:

```text
Do not treat this as a live strategy.
Keep it as a research scaffold.
Move toward RL meta-control and stronger walk-forward validation.
```

## Meta-Controller Scaffold

The next architecture is now represented by `src/quant_ml_lab/strategy_selector.py`.

Instead of asking SAC to directly solve the whole trading problem, the scaffold separates:

- candidate strategy construction
- regime-aware selection
- risk budget assignment
- selected-position backtest metrics
- selection counts for auditability

The public candidate family is:

- `no_trade`
- `mean_reversion_full`
- `mean_reversion_low_risk`
- `volatility_defensive`
- `cvar_defensive`

The current selector is deliberately rule-based. Its purpose is to create the module contract for a future learned RL selector:

```text
state -> strategy choice + risk budget
```

The learned version must beat both the individual candidates and the rule-based selector under walk-forward validation before it should be treated as a stronger result.

The first learned selector scaffold is `src/quant_ml_lab/strategy_selector_dqn.py`. It trains a DQN-style discrete policy over the same candidate family.

Current interpretation:

- It proves that a learned selector can be trained, evaluated, and checkpointed.
- It reports loss traces, reward traces, Q-value traces, random baseline comparison, candidate baselines, and action concentration.
- It is still a single-split scaffold.
- It should not be treated as robust until walk-forward, seed stability, random selector, and transaction-cost stress reports are added.

The SAC allocator is the preferred direction for continuous risk budgeting. It maps a continuous SAC action to softmax weights over the strategy family. The first demo does not beat the rule-based selector yet, but it avoids one-action collapse and gives a better module boundary for risk allocation.

The first SAC walk-forward report is still not robust-ready. Mean Sharpe delta is slightly positive, but only `1 / 3` folds beat the rule-based selector.

The SAC robustness matrix is stricter and currently fails:

- cases: `8`
- mean Sharpe delta: `-0.16407816642347073`
- positive Sharpe case rate: `0.5`
- positive return case rate: `0.5`
- robust case rate: `0.0`
- robust-ready: `false`

Current interpretation: the allocator can sometimes improve total return, but it is not stable enough across data seeds, SAC seeds, and cost assumptions. The right next step is not to add another RL algorithm; it is to improve data diversity, reward ablation, baseline decomposition, and offline-RL safety checks.

The first reward ablation also fails:

- cases: `10`
- best ablation by Sharpe: `no_drawdown_penalty`
- full reward mean Sharpe delta: `-0.1791830938487189`
- best minus full Sharpe delta: `0.0002405495444227912`
- robust case rate: `0.0`
- robust-ready: `false`

Current interpretation: removing one public penalty at a time barely changes the outcome. The problem is probably not a single reward coefficient. The stronger next move is multi-regime data generation and candidate-level benchmark decomposition.

The first multi-regime candidate benchmark is stricter:

- cases: `3`
- mean selected Sharpe: `-0.9164348741317294`
- mean selected minus best Sharpe: `-0.9164348741317294`
- strongest candidate by mean Sharpe: `no_trade`
- weakest regime counts: `{'calm_mean_reverting': 2, 'slow_reversion': 1}`
- benchmark-ready: `false`
- no-trade best rate: `1.0`
- selected positive Sharpe rate: `0.0`
- negative selected regime rate: `0.6666666666666666`
- RL allocation ready: `false`
- research decision: `candidate_signal_redesign_before_rl`

Current interpretation: the current candidate family is too weak under multi-regime stress. The selected policy also loses in calm mean-reverting and slow-reversion segments. A learned allocator should not be expected to create edge from weak candidates. The readiness gate now blocks more SAC tuning and points the next iteration toward stronger candidate signals, supervised/meta-label filters, and regime-level validation.

## Public Boundary

This pipeline deliberately excludes:

- production universe selection
- private feature recipes
- proprietary thresholds
- live broker routing
- order execution logic
- capital allocation
- real PnL claims

The point is to make the strategy workflow real without leaking strategy edge.
