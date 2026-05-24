.PHONY: test lint sample-backtest walk-forward baseline-vs-regime cvar-sizing rl-sizing-comparison q-learning-gridworld sac-bandit monitoring-report experiment-demo latency-benchmark all-reports serve

test:
	python -m pytest -q

lint:
	ruff check src tests examples

sample-backtest:
	python examples/run_sample_backtest.py

walk-forward:
	python examples/run_walk_forward.py

baseline-vs-regime:
	python examples/run_baseline_vs_regime_filter.py

cvar-sizing:
	python examples/run_cvar_sizing_report.py

rl-sizing-comparison:
	python examples/run_rl_sizing_comparison.py

q-learning-gridworld:
	python examples/run_q_learning_gridworld.py

sac-bandit:
	python examples/run_sac_bandit.py

monitoring-report:
	python examples/run_monitoring_report.py

experiment-demo:
	python examples/run_experiment_demo.py

latency-benchmark:
	python examples/run_latency_benchmark.py

all-reports: sample-backtest walk-forward baseline-vs-regime cvar-sizing rl-sizing-comparison q-learning-gridworld sac-bandit monitoring-report experiment-demo latency-benchmark

serve:
	uvicorn quant_ml_lab.api:app --app-dir src --reload
