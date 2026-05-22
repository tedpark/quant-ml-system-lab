# Portfolio Artifact Plan

This document turns the broader AI quant career strategy into five concrete artifacts inside this repository.

The goal is to make this project easy for a recruiter, hiring manager, or engineer interviewer to verify within 10-15 minutes.

## The Five Artifacts

```text
1. Executable GitHub repository
2. Benchmark reports
3. Model serving / demo layer
4. Technical writing series
5. Resume / portfolio / book landing-page material
```

These artifacts are deliberately connected. The repository proves execution, the benchmark reports prove evaluation judgment, the serving layer proves production ML ability, the writing proves technical communication, and the resume/landing material turns the work into a hiring signal.

## 1. Executable GitHub Repository

### Purpose

Show that this is not a notebook-only project. The public repo should demonstrate a reproducible financial ML engineering workflow with tests, examples, documentation, and a serving boundary.

### Current Evidence

Implemented:

- Python package under `src/quant_ml_lab`
- unit tests under `tests/`
- executable examples under `examples/`
- Makefile commands
- Dockerfile
- CI workflow badge
- sanitized public/private boundary

Primary files:

- `README.md`
- `Makefile`
- `pyproject.toml`
- `Dockerfile`
- `.github/workflows/ci.yml`
- `src/quant_ml_lab/`
- `tests/`
- `examples/`

### Verification Commands

```bash
make test
make sample-backtest
make walk-forward
make monitoring-report
make experiment-demo
make latency-benchmark
make serve
```

### Completion Criteria

- A reviewer can clone the repo and run `make test`.
- A reviewer can generate at least one backtest report.
- A reviewer can generate a walk-forward report.
- A reviewer can start the demo API.
- The README clearly explains what is included and what is intentionally excluded.
- The project does not expose private strategy logic, production thresholds, broker code, or live trading performance.

### Next Improvements

- Add a `make all-reports` command that runs every report generator.
- Add a short `docs/reproducibility.md`.
- Add static sample JSON reports under `docs/examples/` if the generated `reports/` directory remains uncommitted.
- Add a repo diagram image or Mermaid diagram to `docs/architecture.md`.

## 2. Benchmark Reports

### Purpose

Show that the system is evaluated honestly. In financial ML, validation quality is more important than model complexity.

### Current Evidence

Implemented or scaffolded:

- leakage-aware validation utilities
- time-ordered split logic
- walk-forward evaluation example
- transaction-cost-aware metrics
- local benchmark examples
- latency benchmark example

Primary files:

- `docs/benchmarks.md`
- `docs/validation.md`
- `src/quant_ml_lab/benchmarking.py`
- `src/quant_ml_lab/validation.py`
- `src/quant_ml_lab/walk_forward.py`
- `examples/run_sample_backtest.py`
- `examples/run_walk_forward.py`
- `examples/run_latency_benchmark.py`

### Required Benchmark Reports

#### Baseline vs Regime Filter

Target file:

```text
docs/benchmark_reports/baseline_vs_regime_filter.md
```

Must include:

- baseline definition
- regime filter definition
- train/test or walk-forward split
- transaction cost assumptions
- Sharpe / Sortino / MDD / win rate / turnover
- regime-specific performance
- failure cases

#### SAC vs PPO vs QR-DQN

Target file:

```text
docs/benchmark_reports/rl_sizing_comparison.md
```

Must include:

- shared environment assumptions
- identical split protocol
- identical cost model
- seed protocol
- reward definition
- sizing constraints
- comparison table
- instability or failure analysis

#### CVaR Sizing Before/After

Target file:

```text
docs/benchmark_reports/cvar_sizing.md
```

Must include:

- how return quantiles are estimated
- CVaR 5% / 10%
- position multiplier rule
- drawdown before/after
- turnover impact
- veto/floor behavior

#### Serving Latency

Target file:

```text
docs/benchmark_reports/serving_latency.md
```

Must include:

- model or dummy model shape
- p50 / p95 / p99 latency
- batch size
- local machine details if publishable
- warmup protocol
- failure thresholds

### Completion Criteria

- Each report explains the protocol before the result.
- Each report includes at least one failure or limitation.
- Metrics include costs, drawdown, and tail risk, not only return.
- Reports are reproducible from scripts or Makefile commands.

### Next Improvements

- Keep generated Markdown reports under `docs/benchmark_reports/`.
- Add JSON report snapshots generated from examples.
- Add a benchmark summary table to `README.md`.

## 3. Model Serving / Demo Layer

### Purpose

Show that models can be moved out of research code and exposed through a stable production-style contract.

### Current Evidence

Implemented:

- FastAPI demo API
- serving schemas
- prediction response metadata
- monitoring report skeleton
- local latency benchmark
- manifest-only model registry pattern

Primary files:

- `docs/serving.md`
- `docs/deployment.md`
- `docs/monitoring.md`
- `src/quant_ml_lab/api.py`
- `src/quant_ml_lab/serving.py`
- `src/quant_ml_lab/monitoring.py`
- `examples/run_monitoring_report.py`
- `examples/run_experiment_demo.py`
- `examples/run_latency_benchmark.py`
- `tests/test_api.py`
- `tests/test_serving.py`
- `tests/test_monitoring.py`

### Required API Surface

Minimum public demo endpoints:

```text
GET  /health
POST /predict
GET  /model/info
GET  /metrics or report-style monitoring output
```

If the repo later adds model reload behavior:

```text
POST /model/reload
```

### Required Operating Signals

The demo should expose or document:

- model version
- input schema validation
- prediction timestamp
- latency
- drift metric placeholder
- rolling quality placeholder
- error response shape

### Completion Criteria

- `make serve` starts the API.
- README includes a valid curl example.
- API tests pass.
- Serving docs explain the public/private boundary.
- The demo does not imply that public dummy predictions are live trading signals.

### Next Improvements

- Add OpenAPI screenshot or generated API contract.
- Add a static `docs/examples/predict_response.json`.
- Add a model reload pattern using a sanitized manifest, not a private checkpoint.
- Add a "how this maps to production" section in `docs/serving.md`.

## 4. Technical Writing Series

### Purpose

Turn the implementation into a readable hiring signal for overseas and remote roles. A hiring manager should understand the technical judgment without reading every file.

### Recommended Articles

#### 1. Why Financial ML Fails Without Walk-Forward Validation

Repo anchors:

- `docs/validation.md`
- `src/quant_ml_lab/validation.py`
- `src/quant_ml_lab/walk_forward.py`
- `examples/run_walk_forward.py`

Core message:

Financial ML fails when validation leaks future information or ignores transaction costs.

#### 2. Regime-Aware Pair Trading with HMM

Repo anchors:

- `docs/ai_quant_strategy.md`
- future `docs/benchmarks/baseline_vs_regime_filter.md`

Core message:

The model should answer when a strategy is valid, not just whether a single trade looks attractive.

#### 3. QR-DQN and CVaR for Tail-Risk-Aware Position Sizing

Repo anchors:

- `src/quant_ml_lab/risk.py`
- future `docs/benchmarks/cvar_sizing.md`

Core message:

Distributional outputs are useful because sizing should react to downside tail risk, not only expected return.

#### 4. SAC vs PPO vs QR-DQN for Trading Position Sizing

Repo anchors:

- future `docs/benchmarks/rl_sizing_comparison.md`

Core message:

RL should be constrained and benchmarked under the same environment, split, and cost assumptions.

#### 5. Serving RL Policies with FastAPI and MLflow-Style Registry Metadata

Repo anchors:

- `docs/serving.md`
- `docs/deployment.md`
- `docs/monitoring.md`
- `src/quant_ml_lab/api.py`
- `src/quant_ml_lab/serving.py`
- `src/quant_ml_lab/experiments.py`

Core message:

Production ML credibility comes from the lifecycle around the model: schema, versioning, latency, monitoring, and rollback patterns.

### Article Template

Each article should follow this structure:

```text
Problem
Why the naive approach fails
Method
Implementation
Experiment
Result
Failure cases
What I would improve next
```

### Completion Criteria

- Every article links back to this repo.
- Every article includes executable code references.
- Every article includes at least one limitation or failure mode.
- At least three articles are published in English before using this project aggressively for overseas applications.

### Next Improvements

- Add `docs/writing/` with article drafts.
- Add a blog index to the README after publication.
- Cross-link the English book version once the public landing page exists.

## 5. Resume / Portfolio / Book Landing-Page Material

### Purpose

Convert the repo into concise hiring language.

This matters because recruiters do not read full books or full repositories. They scan for credible signals:

- scope
- reproducibility
- metrics
- production relevance
- public/private boundary
- communication clarity

### Resume Bullet

Use this for ML Engineer / Financial ML Engineer roles:

```text
Built a sanitized public companion project for a production-style financial ML/RL system covering leakage-aware walk-forward validation, transaction-cost-aware benchmarking, CVaR risk-control utilities, experiment tracking patterns, FastAPI serving contracts, latency benchmarking, and monitoring report skeletons.
```

Stronger version when RL benchmark reports are published:

```text
Built a production-style AI quant system covering HMM-style regime-aware validation, pair-trading baselines, SAC/PPO/QR-DQN position-sizing comparisons, CVaR-aware risk control, MLflow-style registry metadata, FastAPI inference, and drift/latency monitoring.
```

### LinkedIn / Portfolio Summary

```text
Quant ML System Lab is a sanitized public companion project for financial ML engineering. It demonstrates how to structure a production-style quant ML workflow from validation and benchmark design to risk control, model serving, and monitoring without exposing proprietary trading rules or private data.
```

### Book Landing-Page Copy

```text
Companion lab for Agentic Quant Trading with Python: a hands-on financial ML engineering project covering walk-forward validation, regime-aware modeling patterns, distributional RL risk-control concepts, CVaR-aware sizing, experiment tracking, FastAPI-style serving, and monitoring.
```

### Interview Pitch

```text
In financial ML, the first problem is not model complexity but evaluation error.
This project starts with leakage-aware walk-forward validation and transaction-cost-aware baselines.
Then it adds the production ML layer around the research workflow: experiment tracking,
model metadata, serving contracts, latency benchmarks, and monitoring reports.
The public repository is intentionally sanitized, so it demonstrates the engineering architecture
without disclosing private strategy parameters or live trading logic.
```

### Completion Criteria

- README has a clear one-screen explanation.
- `docs/recruiting_notes.md` includes the target roles and interview pitch.
- Portfolio page links to this repo.
- Book landing page links to this repo.
- Resume points to this repo as public evidence.

### Next Improvements

- Add a public project page under the main portfolio.
- Add the English book landing page.
- Add a short demo video or GIF after the API and reports stabilize.
- Add published blog links once articles are live.

## 90-Day Execution Checklist

### Days 1-30: Make the Repo Self-Verifying

- [x] README explains the public/private boundary.
- [x] `make test` exists.
- [x] sample backtest example exists.
- [x] walk-forward example exists.
- [x] API serving example exists.
- [x] add `make all-reports`.
- [x] add generated Markdown benchmark snapshots.
- [x] add `docs/reproducibility.md`.

### Days 31-60: Publish Honest Benchmarks

- [x] publish baseline vs regime-filter report.
- [x] publish CVaR sizing report.
- [x] publish serving latency report.
- [x] publish RL sizing comparison design, even before all models are implemented.
- [ ] add benchmark summary to README.

### Days 61-90: Convert to Hiring Assets

- [ ] publish at least three English articles.
- [ ] add portfolio project page.
- [ ] add English book landing page link.
- [ ] update resume bullet with repo URL.
- [ ] update LinkedIn featured project.

## Final Positioning

This repository should support one clear claim:

```text
I build production-style AI quant systems that connect financial data pipelines,
model training, walk-forward validation, risk-aware decisioning, inference serving,
and operational monitoring.
```

The public repo does not need to reveal a profitable strategy. It needs to prove engineering judgment, validation discipline, and production ML execution.
