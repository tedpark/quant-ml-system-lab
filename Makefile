.PHONY: test lint sample-backtest walk-forward baseline-vs-regime cvar-sizing rl-sizing-comparison q-learning-gridworld sac-bandit torch-sac hmm-sac-sizing hmm-sac-training-validation pair-rl-strategy pair-rl-strategy-walk-forward strategy-selector-demo strategy-selector-dqn-demo strategy-allocator-sac-demo strategy-allocator-sac-walk-forward strategy-allocator-sac-robustness strategy-allocator-sac-reward-ablation monitoring-report experiment-demo latency-benchmark all-reports serve

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

torch-sac:
	python examples/run_torch_sac_quadratic.py

hmm-sac-sizing:
	python examples/run_hmm_sac_sizing.py

hmm-sac-training-validation:
	python examples/run_hmm_sac_training_validation.py

pair-rl-strategy:
	python examples/run_pair_rl_strategy.py

pair-rl-strategy-walk-forward:
	python examples/run_pair_rl_strategy_walk_forward.py

strategy-selector-demo:
	python examples/run_strategy_selector_demo.py

strategy-selector-dqn-demo:
	python examples/run_strategy_selector_dqn_demo.py

strategy-allocator-sac-demo:
	python examples/run_strategy_allocator_sac_demo.py

strategy-allocator-sac-walk-forward:
	python examples/run_strategy_allocator_sac_walk_forward.py

strategy-allocator-sac-robustness:
	python examples/run_strategy_allocator_sac_robustness.py

strategy-allocator-sac-reward-ablation:
	python examples/run_strategy_allocator_sac_reward_ablation.py

monitoring-report:
	python examples/run_monitoring_report.py

experiment-demo:
	python examples/run_experiment_demo.py

latency-benchmark:
	python examples/run_latency_benchmark.py

all-reports: sample-backtest walk-forward baseline-vs-regime cvar-sizing rl-sizing-comparison q-learning-gridworld sac-bandit torch-sac hmm-sac-sizing hmm-sac-training-validation pair-rl-strategy pair-rl-strategy-walk-forward strategy-selector-demo strategy-selector-dqn-demo strategy-allocator-sac-demo strategy-allocator-sac-walk-forward strategy-allocator-sac-robustness strategy-allocator-sac-reward-ablation monitoring-report experiment-demo latency-benchmark

serve:
	uvicorn quant_ml_lab.api:app --app-dir src --reload
