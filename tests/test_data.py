from quant_ml_lab.data import (
    SyntheticPairConfig,
    SyntheticRegimePairConfig,
    make_synthetic_pair,
    make_synthetic_regime_pair,
    train_test_split_time,
)


def test_make_synthetic_pair_is_reproducible():
    a = make_synthetic_pair(SyntheticPairConfig(seed=123, periods=120))
    b = make_synthetic_pair(SyntheticPairConfig(seed=123, periods=120))

    assert a.equals(b)
    assert list(a.columns) == ["asset_a", "asset_b"]
    assert len(a) == 120
    assert a.index.is_monotonic_increasing


def test_train_test_split_time_preserves_order():
    df = make_synthetic_pair(SyntheticPairConfig(periods=100))
    train, test = train_test_split_time(df, train_fraction=0.75)

    assert len(train) == 75
    assert len(test) == 25
    assert train.index.max() < test.index.min()


def test_make_synthetic_regime_pair_has_regime_metadata():
    df = make_synthetic_regime_pair(SyntheticRegimePairConfig(periods=260, seed=31))

    assert len(df) == 260
    assert {"asset_a", "asset_b", "synthetic_regime"}.issubset(df.columns)
    assert df["synthetic_regime"].nunique() >= 2
    assert df[["asset_a", "asset_b"]].gt(0.0).all().all()
