from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.hmm_rl import ForwardGaussianHMM2State, build_hmm_rl_dataset
from quant_ml_lab.regime import rolling_spread_volatility
from quant_ml_lab.validation import backtest_pair_baseline


def test_forward_hmm_probabilities_sum_to_one():
    df = make_synthetic_pair(SyntheticPairConfig(periods=220))
    baseline, _ = backtest_pair_baseline(df)
    vol = rolling_spread_volatility(baseline["spread"], lookback=20)
    hmm = ForwardGaussianHMM2State().fit(vol.iloc[:150])

    probs = hmm.predict_proba_forward(vol.iloc[150:])

    assert list(probs.columns) == ["normal_prob", "high_vol_prob"]
    assert ((probs.sum(axis=1) - 1.0).abs() < 1e-8).all()


def test_hmm_rl_dataset_contains_features():
    df = make_synthetic_pair(SyntheticPairConfig(periods=260))
    train, test = train_test_split_time(df)

    dataset = build_hmm_rl_dataset(train, test)

    assert dataset.feature_columns == (
        "feature_zscore",
        "feature_abs_zscore",
        "feature_high_vol_prob",
        "feature_position",
    )
    assert set(dataset.feature_columns).issubset(dataset.frame.columns)
    assert dataset.frame["high_vol_prob"].between(0.0, 1.0).all()
