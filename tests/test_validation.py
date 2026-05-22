from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair
from quant_ml_lab.validation import (
    BacktestConfig,
    backtest_pair_baseline,
    generate_mean_reversion_positions,
)


def test_positions_are_bounded():
    df = make_synthetic_pair(SyntheticPairConfig(periods=160))
    result, _ = backtest_pair_baseline(df, BacktestConfig())

    assert set(result["position"].unique()).issubset({-1, 0, 1})
    assert result["equity"].iloc[-1] > 0


def test_backtest_metrics_are_finite():
    df = make_synthetic_pair(SyntheticPairConfig(periods=180))
    _, metrics = backtest_pair_baseline(df)

    values = metrics.as_dict()
    assert values["trades"] >= 0
    assert values["turnover"] >= 0
    for key in ["total_return", "annualized_return", "annualized_volatility", "sharpe", "max_drawdown"]:
        assert values[key] == values[key]


def test_entry_must_exceed_exit():
    try:
        generate_mean_reversion_positions(make_synthetic_pair()["asset_a"], entry_z=0.2, exit_z=0.5)
    except ValueError as exc:
        assert "entry_z" in str(exc)
    else:
        raise AssertionError("expected ValueError")
