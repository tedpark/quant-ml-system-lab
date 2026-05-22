.PHONY: test lint sample-backtest walk-forward monitoring-report experiment-demo latency-benchmark serve

test:
	python -m pytest -q

lint:
	ruff check src tests

sample-backtest:
	python examples/run_sample_backtest.py

walk-forward:
	python examples/run_walk_forward.py

monitoring-report:
	python examples/run_monitoring_report.py

experiment-demo:
	python examples/run_experiment_demo.py

latency-benchmark:
	python examples/run_latency_benchmark.py

serve:
	uvicorn quant_ml_lab.api:app --app-dir src --reload
