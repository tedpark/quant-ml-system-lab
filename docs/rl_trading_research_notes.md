# RL Trading Research Notes

This note documents the current conclusion after testing the public HMM + SAC strategy candidate and reviewing related research.

For the deeper failure analysis and redesign decision, see:

```text
docs/rl_trading_failure_analysis.md
```

## Current Conclusion

The current structure should not be treated as a finished trading strategy.

It is useful as a research scaffold, but the walk-forward result shows that the model is not robust enough:

- Single split: improved total return, Sharpe, drawdown, and showed a meaningful high-volatility active multiplier shift.
- Walk-forward: reduced losses versus baseline across folds, but Sharpe improvement was inconsistent.
- Robustness status: `robust_ready = false`.

The main issue is not simply that RL "does not work." The more precise diagnosis is:

```text
current data + current environment + current validation protocol
are not enough to support a stable RL trading policy
```

## Evidence From The Current Lab

The latest single-split strategy result:

- baseline total return: `0.009170647738297122`
- strategy total return: `0.010779211741699601`
- baseline Sharpe: `0.1341620041952139`
- strategy Sharpe: `0.31331942713430083`
- baseline max drawdown: `-0.07140219072869236`
- strategy max drawdown: `-0.037391711317835585`
- learned regime response: `aggressive_sizing_in_high_vol`

The stricter walk-forward result:

- folds: `3`
- trade-ready folds: `1`
- trade-ready rate: `0.3333333333333333`
- mean total return delta: `0.03844859587141414`
- mean Sharpe delta: `-0.13335791895013735`
- positive return-delta folds: `3`
- positive Sharpe-delta folds: `1`
- robust-ready: `false`

Interpretation:

- The model behaves like a risk-reduction overlay more than a stable alpha engine.
- It reduced losses, but it did not consistently improve risk-adjusted returns.
- The regime response is not robust across folds.

## Why The Current Structure Is Weak

### 1. Data Is Too Small

RL is sample hungry. A single pair, daily frequency, and a small number of walk-forward folds do not provide enough state-action-reward diversity.

The current data is enough to test engineering contracts:

- train/validation split
- checkpoint save/reload
- walk-forward reporting
- regime behavior diagnostics

It is not enough to conclude that a learned policy is robust.

### 2. The Environment Is Too Narrow

The current action is a single continuous position multiplier:

```text
action -> multiplier in [0, 1]
final_position = baseline_position * multiplier
```

This is safer than letting RL directly buy/sell, but it also means the agent can only reshape an existing baseline signal. It cannot learn:

- strategy selection
- entry timing
- exit timing
- holding period control
- no-trade decisions independent of the baseline
- cross-asset risk budget

### 3. Offline RL Is A Harder Problem

Trading RL is mostly offline RL. The agent learns from historical data and cannot safely explore live markets.

That creates three problems:

- dataset coverage: the historical data may not contain enough examples of states/actions the policy wants to use
- distribution shift: the future market can differ from the training data
- overfitting: backtest gains can be false positives

### 4. The Baseline Can Dominate The RL Behavior

Because the current agent sizes a baseline position, regime exposure can increase simply because the baseline is active more often in a regime.

This is why the report now separates:

- absolute SAC exposure
- active baseline rows
- active multiplier
- active multiplier shift

The active multiplier shift is the better test of whether SAC itself changed behavior.

## Related Research

### Trading RL Surveys

[Deep Reinforcement Learning for Trading, A Critical Survey](https://www.mdpi.com/2306-5729/6/11/119)

This survey emphasizes that state representation, reward shaping, transaction costs, and generalization are central failure points in trading RL.

[Reinforcement Learning in Financial Markets](https://www.mdpi.com/2306-5729/4/3/110)

This survey argues that financial RL research needs careful baselines, comparison protocols, and realistic validation.

### Overfitting And Validation

[Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting](https://arxiv.org/abs/2209.05559)

This paper treats DRL backtest overfitting as a hypothesis-testing problem and argues that optimistic backtests can produce false positives.

### Offline RL

[Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)

This tutorial explains why learning from fixed historical data is difficult when the learned policy selects actions outside the behavior distribution.

[A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems](https://arxiv.org/abs/2203.01387)

This survey organizes offline RL failure modes and stresses dataset quality, coverage, and out-of-distribution action risk.

### Trading Frameworks

[FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance](https://arxiv.org/abs/2011.09607)

FinRL separates data, environment, agent, and backtesting layers. It also includes transaction cost, liquidity, and risk constraints.

[FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance](https://arxiv.org/abs/2111.09395)

This framework-level paper reinforces the need for reproducible pipelines rather than isolated strategy demos.

### Regime Switching And RL

[Regime-Based Portfolio Allocation Using Hidden Markov Models and Reinforcement Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5785443)

This work combines regime modeling with RL for asset allocation. The broader lesson is that RL is more natural as an allocation or risk-budget controller than as a raw alpha generator.

[Reinforcement learning for continuous-time mean-variance portfolio selection in a regime-switching market](https://www.sciencedirect.com/science/article/pii/S0165188923001938)

This paper studies RL under regime-switching assumptions and supports the idea that regime structure should be part of the formal problem design.

## Better Architecture

The current structure:

```text
baseline pair signal
-> HMM regime features
-> SAC multiplier
```

Recommended next structure:

```text
regime model
-> alpha/strategy family
-> RL meta-controller
-> risk budget and execution constraints
-> walk-forward and overfit rejection
```

The roles should be separated:

```text
ML models:
  prediction, signal quality, regime detection

RL policy:
  strategy selection, risk budget, sizing, exit control

Rules:
  execution, hard risk limits, kill switch, compliance constraints
```

## Implemented Scaffold

The first module for the recommended architecture is now implemented as a public, rule-based scaffold:

```text
src/quant_ml_lab/strategy_selector.py
```

It separates the strategy family from the selector:

```text
HMM/SAC research frame
-> candidate strategy family
-> regime-aware selector
-> selected position, selected return, selected equity
-> metrics and selection audit
```

The current candidate family is:

- `no_trade`
- `mean_reversion_full`
- `mean_reversion_low_risk`
- `volatility_defensive`
- `cvar_defensive`

Run:

```bash
make strategy-selector-demo
```

Output:

```text
reports/strategy_selector_demo.json
```

Important boundary:

```text
This is not yet a learned RL meta-controller.
It is the modular interface that a learned selector can replace later.
```

The current selector intentionally uses transparent public rules:

- no-trade when the baseline is inactive
- lower-risk mean reversion in normal conditions
- volatility defensive policy when high-volatility probability is high
- CVaR/drawdown defensive policy when baseline drawdown is elevated

The demo result reduced max drawdown versus the baseline, but did not improve Sharpe:

- baseline total return: `-0.008486678752498311`
- selected total return: `-0.020134249591297726`
- baseline Sharpe: `-0.04672005285178897`
- selected Sharpe: `-0.2738501462822829`
- baseline max drawdown: `-0.11935204698174917`
- selected max drawdown: `-0.04317433598481957`

Interpretation:

- The selector is behaving as a risk allocator, not an alpha engine.
- Drawdown control improved, but return quality did not.
- The next research task is to learn this selector under strict walk-forward and offline-RL safety gates.

## Learned Selector Scaffold

The first learned selector is implemented as a DQN-style discrete meta-controller:

```text
src/quant_ml_lab/strategy_selector_dqn.py
```

Run:

```bash
make strategy-selector-dqn-demo
```

Output:

```text
reports/strategy_selector_dqn_demo.json
artifacts/strategy_checkpoints/strategy_selector_dqn.pt
```

The learned selector action space is the same public candidate family:

- `no_trade`
- `mean_reversion_full`
- `mean_reversion_low_risk`
- `volatility_defensive`
- `cvar_defensive`

Single-split demo result:

- DQN validation total return: `-0.0007502419429273877`
- rule-based validation total return: `-0.016180094534256173`
- DQN validation Sharpe: `-0.024163421476573885`
- rule-based validation Sharpe: `-0.6856862476669808`
- DQN validation max drawdown: `-0.0580629834875529`
- rule-based validation max drawdown: `-0.0556060554082175`
- random validation Sharpe: `-1.5893581968590802`
- DQN loss tail mean: `2.0426711434993193e-05`
- DQN q-value tail absolute mean: `0.02748905373737216`
- DQN validation action concentration: `0.8275862068965517`

Interpretation:

- The DQN selector can now be trained, evaluated, checkpointed, and compared against the rule-based selector.
- The current demo is a single split, not robust evidence.
- The learned policy mostly selected `mean_reversion_full`, so this is not yet strong regime-aware behavior.
- The action concentration is close to the current gate, so one-action collapse remains a real risk.
- Walk-forward, seed stability, random-selector baseline, and cost stress are required next.

## SAC Allocator Scaffold

The SAC-only allocator is implemented as:

```text
src/quant_ml_lab/strategy_selector_sac.py
```

Run:

```bash
make strategy-allocator-sac-demo
```

Output:

```text
reports/strategy_allocator_sac_demo.json
artifacts/strategy_checkpoints/strategy_allocator_sac.pt
```

This structure is a better fit for continuous risk allocation:

```text
state
-> continuous SAC action
-> softmax strategy weights
-> weighted position
```

Single-split demo result:

- SAC validation total return: `-0.01970259896837645`
- rule-based validation total return: `-0.016180094534256173`
- SAC validation Sharpe: `-0.9180280393066089`
- rule-based validation Sharpe: `-0.6856862476669808`
- SAC validation max drawdown: `-0.05414547018742388`
- rule-based validation max drawdown: `-0.0556060554082175`
- SAC validation weight concentration: `0.21261754789544707`

Interpretation:

- SAC is the preferred architecture for risk budget and allocation.
- This first SAC allocator does not yet beat the rule-based selector.
- It did avoid one-action collapse by spreading weight across candidate strategies.
- Next work should focus on walk-forward SAC, multi-seed stability, and reward ablation.

Walk-forward SAC allocator result:

- folds: `3`
- mean validation Sharpe: `0.6084243033462294`
- mean rule-based Sharpe: `0.5807367743726034`
- mean Sharpe delta: `0.027687528973625853`
- positive Sharpe delta folds: `1`
- positive return delta folds: `1`
- robust-ready: `false`

Interpretation:

- The mean delta is slightly positive.
- The fold consistency is weak.
- This confirms that the current bottleneck is not only the SAC algorithm; data coverage and validation must improve.

Robustness matrix result:

- cases: `8`
- mean Sharpe delta: `-0.16407816642347073`
- median Sharpe delta: `-0.16017712382241217`
- worst Sharpe delta: `-0.4055403832446698`
- mean total return delta: `0.0012976537192070092`
- positive Sharpe case rate: `0.5`
- positive return case rate: `0.5`
- robust case rate: `0.0`
- robust-ready: `false`

Interpretation:

- The allocator is not robust across synthetic data seeds and cost assumptions.
- The current structure should be treated as a falsification harness, not a strategy.
- The next improvement should be reward ablation, candidate benchmark decomposition, and broader data generation before adding more RL algorithms.

Reward ablation result:

- cases: `10`
- ablations: `5`
- best ablation by Sharpe: `no_drawdown_penalty`
- worst ablation by Sharpe: `no_concentration_penalty`
- best mean Sharpe delta: `-0.1789425443042961`
- full reward mean Sharpe delta: `-0.1791830938487189`
- best minus full Sharpe delta: `0.0002405495444227912`
- mean Sharpe delta: `-0.17923910952356698`
- robust case rate: `0.0`
- robust-ready: `false`

Interpretation:

- Individual reward penalties are not driving the failure.
- The tiny difference between full reward and the best ablation suggests the current policy is not strongly using the risk-shaping terms.
- The bottleneck is more likely weak state/data coverage and environment realism.
- Next work should add multi-regime data generation and candidate-level benchmark decomposition before tuning coefficients.

Multi-regime candidate benchmark result:

- cases: `3`
- mean selected Sharpe: `-0.9164348741317294`
- mean selected minus best Sharpe: `-0.9164348741317294`
- worst selected minus best Sharpe: `-1.0888504377611108`
- selected matches best cases: `0`
- strongest candidate by mean Sharpe: `no_trade`
- weakest regime counts: `{'calm_mean_reverting': 2, 'slow_reversion': 1}`
- benchmark-ready: `false`

Interpretation:

- The candidate set itself is weak under multi-regime stress.
- RL allocation is not the primary issue if no-trade is the strongest candidate.
- The regime decomposition shows losses in both calm mean-reverting and slow-reversion segments.
- The next redesign should add better supervised/signal candidates, meta-label filters, and regime-conditioned candidate diagnostics before more SAC tuning.

## Recommended Roadmap

### Phase 1. Keep RL As A Risk Controller

Do not let RL directly invent alpha yet.

Add stronger baselines:

- volatility targeting
- CVaR sizing
- drawdown-based sizing
- no-trade/cash policy
- equal-risk budget policy

The RL policy must beat these, not only the naive baseline.

### Phase 2. Add A Strategy Selector

Replace the single multiplier policy with a meta-controller:

```text
state -> choose one:
  no_trade
  mean_reversion_low_risk
  mean_reversion_full
  volatility_defensive
  cvar_defensive
```

This is more realistic than asking SAC to directly produce all behavior from one continuous action.

### Phase 3. Add Offline RL Safety

Add gates inspired by offline RL concerns:

- action distribution coverage
- behavior policy comparison
- random policy baseline
- conservative policy penalty
- out-of-distribution state report

### Phase 4. Expand Data

The current single-pair synthetic setup is too small.

Needed:

- multi-pair synthetic generator
- multiple regime generators
- multiple cost/slippage assumptions
- real OHLCV adapter
- more walk-forward folds

### Phase 5. Paper Trading Only After Robust Gates

Paper trading should require:

- robust-ready walk-forward result
- positive mean Sharpe delta
- positive return delta in a majority of folds
- stable regime behavior across folds
- checkpoint reproducibility
- cost/slippage stress pass

## Practical Decision

Current decision:

```text
Do not scale this as a live strategy.
Do not claim stable regime learning yet.
Keep it as a research scaffold.
Refactor toward RL meta-control and robust validation.
```

The next useful implementation is a `StrategySelector` / meta-controller, not more SAC tuning on the same single multiplier environment.
