import pandas as pd
import pytest

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair
from quant_ml_lab.monitoring import (
    drift_status,
    feature_drift_report,
    ks_distance,
    population_stability_index,
    rolling_performance,
)
from quant_ml_lab.validation import backtest_pair_baseline


def test_population_stability_index_detects_shift():
    expected = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4] * 20)
    actual_same = expected.copy()
    actual_shifted = pd.Series([1.0, 1.1, 1.2, 1.3, 1.4] * 20)

    assert population_stability_index(expected, actual_same) < 0.01
    assert population_stability_index(expected, actual_shifted) > 1.0


def test_ks_distance_is_bounded():
    expected = pd.Series([0, 1, 2, 3, 4])
    actual = pd.Series([0, 1, 2, 3, 4])

    assert ks_distance(expected, actual) == 0.0
    assert 0.0 <= ks_distance(expected, pd.Series([4, 5, 6])) <= 1.0


def test_feature_drift_report_returns_statuses():
    expected = make_synthetic_pair(SyntheticPairConfig(seed=1, periods=120))
    actual = make_synthetic_pair(SyntheticPairConfig(seed=2, periods=120))

    report = feature_drift_report(expected, actual, ["asset_a", "asset_b"])

    assert [metric.feature for metric in report] == ["asset_a", "asset_b"]
    assert set(metric.status for metric in report).issubset({"ok", "warn", "alert"})


def test_drift_status_threshold_validation():
    with pytest.raises(ValueError):
        drift_status(0.1, warn=0.2, alert=0.1)


def test_rolling_performance_uses_latest_window():
    df = make_synthetic_pair(SyntheticPairConfig(periods=180))
    result, _ = backtest_pair_baseline(df)

    perf = rolling_performance(result["net_return"], window=40)

    assert perf.latest_rolling_return == perf.latest_rolling_return
    assert perf.latest_rolling_max_drawdown <= 0.0
