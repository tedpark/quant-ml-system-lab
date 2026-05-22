.PHONY: test lint sample-backtest

test:
	python -m pytest -q

lint:
	ruff check src tests

sample-backtest:
	python examples/run_sample_backtest.py
