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
- `examples/run_q_learning_gridworld.py`
- `examples/run_sac_bandit.py`
- `examples/run_torch_sac_quadratic.py`
- `examples/run_hmm_sac_sizing.py`
- `examples/run_hmm_sac_training_validation.py`
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

The strategy report also includes regime-conditioned policy behavior, so the lab can inspect whether SAC learned defensive, aggressive, or neutral sizing in high-volatility regimes.

## Monthly Learning Outputs

Each month should produce at least one of:

- one Kaggle notebook
- one RL implementation
- one benchmark report
- one blog draft
- one model serving or monitoring improvement
- one paper implementation note
