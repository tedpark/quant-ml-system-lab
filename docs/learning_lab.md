# ML / AI / RL Learning Lab

This repository is both a portfolio artifact and a long-term learning lab.

The goal is to keep learning ML, AI, and reinforcement learning through small, reproducible experiments that can later connect to financial ML systems.

## Principles

- Every study topic should leave code, tests, or a short write-up.
- Public examples must stay sanitized.
- Trading alpha, production parameters, and live performance remain private.
- Start from simple environments before financial RL.
- Prefer reproducible experiments over impressive claims.

## Learning Tracks

### Track 1. ML Fundamentals

Topics:

- train/validation/test split
- leakage
- metrics
- calibration
- feature importance
- error analysis

Artifacts:

- Kaggle notebooks
- validation notes
- benchmark reports

### Track 2. Production ML

Topics:

- FastAPI serving
- schema validation
- monitoring
- drift reports
- experiment tracking
- CI and Docker

Artifacts:

- API examples
- monitoring reports
- local registry manifests

### Track 3. Reinforcement Learning

Topics:

- MDP
- Bellman equation
- Q-learning
- DQN
- policy gradient
- PPO
- SAC
- distributional RL

Artifacts:

- `src/quant_ml_lab/rl.py`
- `src/quant_ml_lab/sac.py`
- `src/quant_ml_lab/torch_sac.py`
- `src/quant_ml_lab/hmm_rl.py`
- `src/quant_ml_lab/torch_sac_sizing.py`
- `src/quant_ml_lab/strategy_selector.py`
- `src/quant_ml_lab/strategy_selector_dqn.py`
- `src/quant_ml_lab/strategy_selector_sac.py`
- `src/quant_ml_lab/strategy_candidate_benchmark.py`
- `examples/run_q_learning_gridworld.py`
- `examples/run_sac_bandit.py`
- `examples/run_torch_sac_quadratic.py`
- `examples/run_hmm_sac_sizing.py`
- `examples/run_hmm_sac_training_validation.py`
- `examples/run_strategy_selector_demo.py`
- `examples/run_strategy_selector_dqn_demo.py`
- `examples/run_strategy_candidate_benchmark.py`
- `examples/run_strategy_allocator_sac_demo.py`
- `examples/run_strategy_allocator_sac_walk_forward.py`
- `examples/run_strategy_allocator_sac_robustness.py`
- `examples/run_strategy_allocator_sac_reward_ablation.py`
- future DQN/PPO examples

### Track 4. Financial ML / RL

Topics:

- walk-forward validation
- transaction costs
- slippage
- CVaR
- regime detection
- risk-aware sizing
- forward-only regime probabilities
- RL position multipliers

Artifacts:

- baseline vs regime reports
- CVaR sizing report
- RL sizing comparison report
- forward-HMM + SAC sizing report
- pair RL strategy candidate report
- strategy selector demo report
- strategy selector DQN demo report
- strategy candidate benchmark report
- strategy allocator SAC demo report
- strategy allocator SAC walk-forward report
- strategy allocator SAC robustness report
- strategy allocator SAC reward ablation report
- RL trading research notes

## Current RL Lab

The first RL lab is a deterministic GridWorld with tabular Q-learning.

Run:

```bash
python examples/run_q_learning_gridworld.py
```

Output:

```text
reports/q_learning_gridworld.json
```

Why this exists:

- verifies MDP concepts
- provides a tested Q-learning loop
- creates a bridge toward DQN and financial RL
- avoids pretending that trading RL is simple

## Current SAC Lab

The first SAC lab is a one-state continuous bandit with a small discrete action grid.

It demonstrates SAC concepts without adding a deep learning dependency yet:

- stochastic policy
- twin critics
- entropy term
- soft value target
- actor update toward high soft-Q actions

Run:

```bash
python examples/run_sac_bandit.py
```

Output:

```text
reports/sac_bandit.json
```

Why this exists:

- builds SAC intuition before neural networks
- keeps tests fast and deterministic
- avoids exposing trading logic
- creates a bridge toward PyTorch SAC and later financial position sizing

Next SAC steps:

1. Add automatic entropy temperature tuning.
2. Add a multi-step continuous-control toy environment.
3. Add DQN/PPO comparison labs.
4. Only then connect SAC concepts to sanitized financial sizing examples.

## Current PyTorch SAC Lab

The PyTorch SAC lab trains a tanh-squashed Gaussian actor and twin critics on a tiny quadratic action environment.

Run:

```bash
python examples/run_torch_sac_quadratic.py
```

Output:

```text
reports/torch_sac_quadratic.json
```

What it demonstrates:

- replay buffer
- Gaussian actor
- tanh action squashing
- log-prob correction
- twin critics
- target critics
- soft target update
- automatic entropy temperature tuning
- actor loss and critic loss
- alpha loss and alpha trace

This is still not a trading strategy. It is a controlled PyTorch SAC learning lab.

### SAC Implementation Boundary

The private stock-trading implementation is more production-oriented than this public lab. It contains additional operational pieces such as richer environment design, n-step replay, optional prioritized replay, checkpointing, experiment tracking, train/evaluation loops, and deployment-oriented model loading.

The public repository intentionally ports only the architecture-level learning pieces:

- tanh-squashed Gaussian policy
- twin Q critics
- target critics
- replay buffer
- entropy-regularized actor update
- automatic alpha tuning
- soft target updates
- checkpoint save and reload
- sanitized HMM regime features
- sanitized position-sizing environment

The public repository intentionally excludes:

- production universe construction
- production features
- production HMM parameters
- private reward shaping
- live trading thresholds
- broker/execution code
- raw live performance
- trained private checkpoints

That boundary is deliberate. The repo should prove that the ML/RL system can be understood, trained, evaluated, and connected to a financial workflow without exposing actual alpha.

## Current HMM + SAC Sizing Lab

The first HMM + SAC lab connects a forward-only 2-state Gaussian HMM regime filter to a PyTorch SAC position multiplier environment.

Run:

```bash
python examples/run_hmm_sac_sizing.py
```

Output:

```text
reports/hmm_sac_sizing.json
```

What it demonstrates:

- HMM emission parameters are fit on train data only.
- Test data uses recursive forward probabilities only.
- Baseline positions are generated by a simple public mean-reversion rule.
- SAC learns a continuous position multiplier in `[0, 1]`.
- Evaluation uses the learned state-dependent deterministic policy.
- Reward includes transaction cost, turnover penalty, and high-volatility exposure penalty.

Run the production-style validation loop:

```bash
python examples/run_hmm_sac_training_validation.py
```

Output:

```text
reports/hmm_sac_training_validation.json
artifacts/rl_checkpoints/hmm_sac_seed_*.pt
```

What the validation loop adds:

- train/validation split inside the public RL frame
- multi-seed SAC training
- checkpoint save and reload
- validation after checkpoint reload
- finite-metric acceptance gates
- train and validation metrics in one report

What it intentionally does not include:

- production universe
- production features
- production HMM parameters
- live trading rules
- broker integration
- private checkpoints

This is the public learning-lab version of the idea: HMM structures the market state, and RL is constrained to risk/sizing decisions.

Production-grade does not mean "profitable" in this public lab. It means the training loop has the control points required before any model could be trusted: deterministic data split, repeated seeds, held-out validation, checkpoint reproducibility, explicit metrics, and documented failure gates.

## Current Strategy Candidate

The first strategy-candidate pipeline is documented in `docs/strategy_candidate.md`.

Run:

```bash
make pair-rl-strategy
```

Output:

```text
reports/pair_rl_strategy.json
artifacts/strategy_checkpoints/pair_rl_seed_*.pt
```

This connects baseline signal generation, HMM regime features, SAC sizing, validation gates, checkpointing, and latest signal generation. The output includes `trade_ready`; if gates fail, the strategy can still be studied but should not be treated as paper-trading approved.

The strategy report also includes regime-conditioned policy behavior, so the lab can inspect whether SAC learned defensive, aggressive, or neutral sizing in high-volatility regimes. The current public state includes HMM probability, regime transition, spread momentum, spread volatility, recent baseline PnL, and baseline drawdown so the actor has more context than a raw signal-strength overlay.

Run the walk-forward robustness check:

```bash
make pair-rl-strategy-walk-forward
```

The walk-forward report is the stricter check. A single split can be useful for iteration, but paper-trading approval should eventually require robust walk-forward gates.

Research diagnosis:

```text
docs/rl_trading_research_notes.md
```

Current conclusion: the present SAC multiplier environment is useful as a scaffold, but the next architecture should move toward RL as a regime-aware meta-controller and risk allocator.

## Current Strategy Selector Scaffold

The strategy selector scaffold is the first module for that next architecture.

Run:

```bash
make strategy-selector-demo
```

Output:

```text
reports/strategy_selector_demo.json
docs/benchmark_reports/strategy_selector_demo.md
```

What it demonstrates:

- candidate strategy family construction
- no-trade policy
- low-risk mean-reversion policy
- volatility defensive policy
- CVaR/drawdown defensive policy
- regime-aware selection audit
- selected-position metrics

What it does not claim:

- live trading readiness
- private alpha
- learned RL meta-control
- robust profitability

This creates the module boundary for the future learned selector:

```text
state -> strategy choice + risk budget
```

## Current DQN Strategy Selector

The DQN strategy selector is the first learned policy for the meta-controller interface.

Run:

```bash
make strategy-selector-dqn-demo
```

Output:

```text
reports/strategy_selector_dqn_demo.json
docs/benchmark_reports/strategy_selector_dqn_demo.md
artifacts/strategy_checkpoints/strategy_selector_dqn.pt
```

What it demonstrates:

- discrete action space over strategy candidates
- DQN replay buffer
- target network
- epsilon-greedy exploration
- loss, reward, and Q-value traces
- action concentration diagnostics
- held-out validation split
- random selector baseline
- individual candidate baselines
- rule-based selector comparison
- checkpoint writing

What it does not prove yet:

- robust regime learning
- live strategy readiness
- stable performance across seeds
- stable performance across walk-forward folds

The next required artifact is a walk-forward DQN selector report.

## Current SAC Strategy Allocator

The SAC strategy allocator is the preferred path for continuous risk budgeting.

Run:

```bash
make strategy-allocator-sac-demo
```

Output:

```text
reports/strategy_allocator_sac_demo.json
docs/benchmark_reports/strategy_allocator_sac_demo.md
artifacts/strategy_checkpoints/strategy_allocator_sac.pt
```

Walk-forward:

```bash
make strategy-allocator-sac-walk-forward
```

Output:

```text
reports/strategy_allocator_sac_walk_forward.json
docs/benchmark_reports/strategy_allocator_sac_walk_forward.md
```

Robustness matrix:

```bash
make strategy-allocator-sac-robustness
```

Output:

```text
reports/strategy_allocator_sac_robustness.json
docs/benchmark_reports/strategy_allocator_sac_robustness.md
```

Reward ablation:

```bash
make strategy-allocator-sac-reward-ablation
```

Output:

```text
reports/strategy_allocator_sac_reward_ablation.json
docs/benchmark_reports/strategy_allocator_sac_reward_ablation.md
```

What it demonstrates:

- continuous SAC action over the strategy family
- softmax strategy weights
- weighted position construction
- actor, critic, alpha diagnostics
- rule-based selector comparison
- individual candidate baselines
- checkpoint writing

Current limitation:

- the first single-split SAC allocator does not beat the rule-based selector yet
- it should be improved through walk-forward validation, multi-seed checks, and reward ablations
- the first walk-forward report is also not robust-ready, with only `1 / 3` positive Sharpe-delta folds
- the robustness matrix is also not robust-ready: `8` cases, mean Sharpe delta `-0.16407816642347073`, robust case rate `0.0`
- the reward ablation is also not robust-ready: `10` cases, full reward mean Sharpe delta `-0.1791830938487189`, robust case rate `0.0`

Current interpretation:

- SAC remains the preferred allocator architecture.
- The current data, state, environment, and validation protocol are still too weak for strategy claims.
- The next learning step is broader data generation, candidate benchmark decomposition, and offline-RL safety gates.

## Current Candidate Benchmark

The candidate benchmark decomposes the public strategy family before asking SAC to allocate across it.

Run:

```bash
make strategy-candidate-benchmark
```

Output:

```text
reports/strategy_candidate_benchmark.json
docs/benchmark_reports/strategy_candidate_benchmark.md
```

Current result:

- cases: `3`
- mean selected Sharpe: `-0.9164348741317294`
- strongest candidate by mean Sharpe: `no_trade`
- weakest regime counts: `{'calm_mean_reverting': 2, 'slow_reversion': 1}`
- selected matches best cases: `0`
- benchmark-ready: `false`

Interpretation: the current candidate set is not strong enough under multi-regime stress. The regime decomposition points to calm mean-reverting and slow-reversion segments as the weakest cases. This is a more fundamental blocker than SAC tuning.

## Monthly Learning Outputs

Each month should produce at least one of:

- one Kaggle notebook
- one RL implementation
- one benchmark report
- one blog draft
- one model serving or monitoring improvement
- one paper implementation note
