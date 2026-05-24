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
- `examples/run_q_learning_gridworld.py`
- future DQN/PPO/SAC examples

### Track 4. Financial ML / RL

Topics:

- walk-forward validation
- transaction costs
- slippage
- CVaR
- regime detection
- risk-aware sizing

Artifacts:

- baseline vs regime reports
- CVaR sizing report
- RL sizing comparison report

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

## Monthly Learning Outputs

Each month should produce at least one of:

- one Kaggle notebook
- one RL implementation
- one benchmark report
- one blog draft
- one model serving or monitoring improvement
- one paper implementation note

