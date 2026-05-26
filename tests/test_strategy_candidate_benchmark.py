from quant_ml_lab.data import SyntheticRegimePairConfig, make_synthetic_regime_pair
from quant_ml_lab.strategy_candidate_benchmark import (
    run_strategy_candidate_benchmark_matrix,
)
from quant_ml_lab.strategy_selector import StrategySelectorConfig


def test_run_strategy_candidate_benchmark_matrix_returns_summary():
    price_sets = {
        "regime_seed_41": make_synthetic_regime_pair(
            SyntheticRegimePairConfig(periods=360, seed=41)
        ),
        "regime_seed_42": make_synthetic_regime_pair(
            SyntheticRegimePairConfig(periods=360, seed=42)
        ),
    }

    report = run_strategy_candidate_benchmark_matrix(
        price_sets,
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        train_fraction=0.65,
    )

    assert len(report.cases) == 2
    assert report.summary["cases"] == 2
    assert "mean_candidate_sharpe" in report.summary
    assert "best_candidate_counts" in report.summary
    assert "weakest_regime_counts" in report.summary
    assert "rl_allocation_ready" in report.summary
    assert "redesign_reasons" in report.summary
    assert "benchmark_ready" in report.summary
    assert report.cases[0].regime_candidate_metrics
