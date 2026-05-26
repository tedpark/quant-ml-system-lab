# RL Trading Failure Analysis

This note analyzes whether the current financial RL direction is failing because RL is the wrong tool, because the data is insufficient, or because the current environment and validation design are too weak.

## Executive Conclusion

The current structure is not strong enough to support a production trading claim.

The right diagnosis is not:

```text
RL cannot work for trading.
```

The better diagnosis is:

```text
The current data, environment, reward, and validation protocol are too weak
for a learned trading policy to be trusted.
```

RL may still be useful, but only in a narrower role:

```text
prediction alpha: supervised ML / signal models
regime detection: HMM or other state model
risk allocation: SAC or constrained policy optimization
execution rules: deterministic risk and compliance controls
```

The current repo should remain a learning and research lab. It should not be treated as a live strategy system.

## Evidence From The Current Repo

The current lab has three relevant RL branches:

- SAC position multiplier
- DQN discrete strategy selector
- SAC continuous strategy-family allocator

### SAC Position Multiplier

The single split looked acceptable, but walk-forward failed robustness:

- trade-ready folds: `1 / 3`
- trade-ready rate: `0.3333333333333333`
- mean total return delta: `0.03844859587141414`
- mean Sharpe delta: `-0.13335791895013735`
- robust-ready: `false`

Interpretation:

- It can reduce losses versus a weak baseline.
- It does not consistently improve risk-adjusted return.
- The regime behavior is not robust enough.

### DQN Strategy Selector

The DQN selector beat the rule-based selector on one validation split, but showed action concentration:

- DQN validation Sharpe: `-0.024163421476573885`
- rule-based validation Sharpe: `-0.6856862476669808`
- random validation Sharpe: `-1.5893581968590802`
- validation action concentration: `0.8275862068965517`

Interpretation:

- It is a useful baseline for learned selection.
- It is too close to one-action collapse.
- DQN is not the preferred architecture for continuous risk allocation.

### SAC Strategy Allocator

The SAC allocator is architecturally better for continuous risk budget control, but the first split did not beat the rule-based selector:

- SAC validation Sharpe: `-0.9180280393066089`
- rule-based validation Sharpe: `-0.6856862476669808`
- SAC validation weight concentration: `0.21261754789544707`

Interpretation:

- SAC avoided one-action collapse.
- It learned diversified weights.
- It did not learn a stronger validation policy.
- The likely bottleneck is not the SAC implementation alone; it is data, reward, and validation.

### SAC Allocator Walk-Forward

The walk-forward SAC allocator report now tests whether the SAC allocator survives multiple time windows:

- folds: `3`
- mean validation Sharpe: `0.6084243033462294`
- mean rule-based Sharpe: `0.5807367743726034`
- mean Sharpe delta: `0.027687528973625853`
- mean total return delta: `0.007387042262560224`
- positive Sharpe delta folds: `1`
- positive return delta folds: `1`
- robust-ready: `false`

Interpretation:

- The mean result is slightly better than the rule-based selector.
- The fold consistency is poor.
- SAC is not robust enough yet.
- This supports the diagnosis that validation/data coverage is the primary bottleneck.

### SAC Allocator Robustness Matrix

The robustness matrix now repeats the SAC allocator across synthetic data seeds, SAC random seeds, and transaction-cost assumptions:

- cases: `8`
- mean Sharpe delta: `-0.16407816642347073`
- median Sharpe delta: `-0.16017712382241217`
- worst Sharpe delta: `-0.4055403832446698`
- mean total return delta: `0.0012976537192070092`
- worst total return delta: `-0.006128065861054743`
- positive Sharpe case rate: `0.5`
- positive return case rate: `0.5`
- robust case rate: `0.0`
- robust-ready: `false`

Interpretation:

- The allocator sometimes improves total return, but the Sharpe result is not stable.
- One synthetic data seed is materially worse than the rule-based selector.
- No case passes the internal walk-forward robustness gate.
- The current failure mode is consistent with insufficient data diversity and reward/environment mismatch, not with a proven impossibility of RL.

### SAC Allocator Reward Ablation

The reward ablation report removes one public reward component at a time:

- cases: `10`
- ablations: `5`
- best ablation by Sharpe: `no_drawdown_penalty`
- worst ablation by Sharpe: `no_concentration_penalty`
- best mean Sharpe delta: `-0.1789425443042961`
- full reward mean Sharpe delta: `-0.1791830938487189`
- best minus full Sharpe delta: `0.0002405495444227912`
- mean Sharpe delta: `-0.17923910952356698`
- worst Sharpe delta: `-0.40148032323206045`
- mean total return delta: `0.000780051480848698`
- robust case rate: `0.0`
- robust-ready: `false`

Interpretation:

- Removing individual penalties barely changes the result.
- The reward is not the only bottleneck.
- The current state representation and synthetic data coverage are probably too weak to let SAC learn a stable allocator.
- The next redesign should prioritize broader regime/data generation and stronger baselines before tuning reward coefficients.

## What The Literature Suggests

### 1. Trading RL Is Mostly Offline RL

Trading agents train on fixed historical data. They cannot safely explore live markets.

Relevant papers:

- [Offline Reinforcement Learning: Tutorial, Review, and Perspectives on Open Problems](https://arxiv.org/abs/2005.01643)
- [A Survey on Offline Reinforcement Learning: Taxonomy, Review, and Open Problems](https://arxiv.org/abs/2203.01387)
- [Conservative Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2006.04779)
- [Implicit Q-Learning for Offline Reinforcement Learning](https://arxiv.org/abs/2110.06169)

Implication for this repo:

- A policy can overestimate actions that are poorly represented in historical data.
- A learned strategy can look good in-sample while failing out-of-sample.
- The current synthetic single-pair dataset is far too narrow for robust offline RL.

### 2. Backtest Overfitting Is A Central Risk

Relevant papers:

- [The Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
- [Deep Reinforcement Learning for Cryptocurrency Trading: Practical Approach to Address Backtest Overfitting](https://arxiv.org/abs/2209.05559)

Implication for this repo:

- A single validation split is not enough.
- Multiple seeds are not enough if the market window is still narrow.
- Walk-forward and parameter-stability checks are mandatory.

### 3. Trading RL Needs Realistic Frictions And Baselines

Relevant papers:

- [Deep Reinforcement Learning for Trading: A Critical Survey](https://www.mdpi.com/2306-5729/6/11/119)
- [Reinforcement Learning in Financial Markets](https://www.mdpi.com/2306-5729/4/3/110)
- [FinRL: A Deep Reinforcement Learning Library for Automated Stock Trading in Quantitative Finance](https://arxiv.org/abs/2011.09607)
- [FinRL: Deep Reinforcement Learning Framework to Automate Trading in Quantitative Finance](https://arxiv.org/abs/2111.09395)
- [DeepTrader: A Deep Reinforcement Learning Approach for Risk-Return Balanced Portfolio Management with Market Conditions Embedding](https://ojs.aaai.org/index.php/AAAI/article/view/16144)
- [An adaptive portfolio trading system: A risk-return portfolio optimization using recurrent reinforcement learning with expected maximum drawdown](https://www.sciencedirect.com/science/article/pii/S0957417417304402)
- [A Risk-Aware Reinforcement Learning Reward for Financial Trading](https://arxiv.org/abs/2506.04358)

Implication for this repo:

- RL must beat simple baselines: no-trade, volatility targeting, CVaR sizing, drawdown sizing, equal-risk weighting.
- Transaction costs, turnover, slippage, and liquidity constraints must be part of the environment.
- The public lab currently has only simplified costs and synthetic data.

### 4. Regime Structure Is Useful, But Not Sufficient

Relevant papers:

- [Regime-Based Portfolio Allocation Using Hidden Markov Models and Reinforcement Learning](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5785443)
- [Reinforcement learning for continuous-time mean-variance portfolio selection in a regime-switching market](https://www.sciencedirect.com/science/article/pii/S0165188923001938)
- [Dynamic Asset Allocation for Varied Financial Markets under Regime Switching Framework](https://www.sciencedirect.com/science/article/pii/S0377221713002658)
- [Dynamic Asset Allocation under Market Regime Uncertainty: A Markov Decision Process Approach](https://papers.ssrn.com/sol3/Delivery.cfm/6600759.pdf?abstractid=6600759&mirid=1)

Implication for this repo:

- Regime features can help, but they do not automatically create alpha.
- HMM state should probably control risk and allocation, not invent directional edge by itself.
- Regime transitions and uncertainty should be explicit state features.

## Is RL The Wrong Tool?

RL is probably the wrong tool for direct alpha discovery in the current setup.

Reasons:

- daily synthetic single-pair data is too small
- reward is noisy and weak
- action coverage is narrow
- future regimes can differ from training regimes
- single-split validation is fragile

RL can still be useful for:

- risk budget control
- position sizing
- exit/hold discipline
- allocation among already validated signals
- policy learning in a simulator with strong synthetic regime diversity

Better framing:

```text
Do not ask RL to discover alpha.
Ask RL to allocate risk across alpha candidates that already have evidence.
```

## Is The Data Insufficient?

Yes. Data is the biggest bottleneck.

The current data is enough for:

- proving code structure
- testing train/validation mechanics
- checkpointing
- diagnostics
- public portfolio artifact

The current data is not enough for:

- robust RL policy learning
- live trading claims
- high-confidence regime behavior
- estimating rare drawdown states
- stress-testing turnover and costs

Minimum data upgrades:

- many synthetic pairs, not one pair
- multiple regime generators
- multiple volatility and correlation regimes
- real OHLCV adapter
- transaction cost and slippage scenarios
- train/validation/test splits by time
- walk-forward folds across different market conditions

## Is The Environment Mis-Specified?

Partly, yes.

Current environment weaknesses:

- reward still mixes return, turnover, volatility exposure, and drawdown in a hand-tuned way
- the action affects next-step return only, which can encourage short-horizon behavior
- no explicit position holding cost
- no liquidity or market impact
- no uncertainty penalty for out-of-distribution states
- no conservative offline RL penalty
- no benchmark-relative reward term

Better environment target:

```text
state:
  regime probability
  regime uncertainty
  signal strength
  spread momentum
  realized volatility
  recent PnL
  drawdown
  cost stress level
  action coverage score

action:
  risk budget in [0, 1]
  allocation weights over candidate strategies

reward:
  net return
  - turnover cost
  - drawdown penalty
  - high-volatility exposure penalty
  - concentration penalty
  - benchmark underperformance penalty
  - out-of-distribution action penalty
```

## Recommended Fundamental Redesign

The current structure should move from:

```text
single pair
-> HMM features
-> SAC/DQN policy
-> single split report
```

To:

```text
multi-regime data generator / real data adapter
-> signal family
-> baseline risk policies
-> SAC risk allocator
-> walk-forward and seed-stability engine
-> overfit rejection gates
```

## Practical Next Steps

### Phase 1. Stop Adding Algorithms

Do not add PPO, TD3, QR-DQN, Decision Transformer, or more architectures yet.

The problem is not algorithm variety. The problem is:

- data coverage
- validation
- environment design
- overfit rejection

### Phase 2. Build Stronger Data And Validation

Required artifacts:

- multi-pair synthetic generator
- walk-forward SAC allocator report: done
- multi-seed SAC allocator report: first version done
- transaction-cost stress report: first version done
- reward ablation report: first version done
- individual-candidate benchmark report

### Phase 3. Add Offline RL Safety

Required gates:

- action distribution coverage
- out-of-distribution state score
- conservative action penalty
- random policy baseline
- no-trade baseline
- best static candidate baseline
- rule-based selector baseline

### Phase 4. Only Then Improve SAC

After the above, improve SAC with:

- stronger reward normalization
- target entropy tuning by action dimension
- action smoothing
- conservative allocation penalty
- clipped risk budget
- regime-uncertainty penalty

## Decision

Current decision:

```text
Do not treat the current RL strategy as tradable.
Do not keep adding RL algorithms.
Keep SAC as the preferred allocator architecture.
Invest next effort in data generation, validation, and offline-RL safety gates.
```

This keeps the project useful for learning, research, and portfolio signaling without pretending that a fragile backtest is a real trading edge.
