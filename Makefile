.PHONY: test lint sample-backtest walk-forward monitoring-report

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
