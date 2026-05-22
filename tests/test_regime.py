from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.regime import (
    RegimeFilterConfig,
    backtest_regime_filter,
    estimate_high_vol_threshold,
    rolling_spread_volatility,
)
from quant_ml_lab.validation import backtest_pair_baseline


def test_rolling_spread_volatility_is_non_negative():
    df = make_synthetic_pair(SyntheticPairConfig(periods=120))
    baseline, _ = backtest_pair_baseline(df)

    vol = rolling_spread_volatility(baseline["spread"], lookback=20)

    assert (vol >= 0).all()
    assert len(vol) == len(df)


def test_regime_filter_uses_train_threshold_and_returns_metrics():
    df = make_synthetic_pair(SyntheticPairConfig(periods=260))
    train, test = train_test_split_time(df)
    train_result, _ = backtest_pair_baseline(train)

    threshold = estimate_high_vol_threshold(train_result["spread"], RegimeFilterConfig())
    result, report = backtest_regime_filter(train, test)

    assert threshold >= 0
    assert report.threshold == threshold
    assert 0.0 <= report.high_vol_share <= 1.0
    assert "regime_multiplier" in result.columns
    assert report.filtered.trades >= 0
