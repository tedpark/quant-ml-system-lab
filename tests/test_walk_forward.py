from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair
from quant_ml_lab.walk_forward import (
    WalkForwardConfig,
    iter_walk_forward_splits,
    run_walk_forward,
    summarize_walk_forward,
)


def test_walk_forward_splits_preserve_time_order():
    df = make_synthetic_pair(SyntheticPairConfig(periods=420))
    splits = iter_walk_forward_splits(df, WalkForwardConfig(train_size=120, test_size=40, step_size=40))

    assert len(splits) > 1
    for train, test in splits:
        assert train.index.max() < test.index.min()
        assert len(train) == 120
        assert len(test) == 40


def test_run_walk_forward_returns_fold_metrics():
    df = make_synthetic_pair(SyntheticPairConfig(periods=420))
    folds = run_walk_forward(df, WalkForwardConfig(train_size=120, test_size=40, step_size=80))

    assert len(folds) >= 3
    assert folds[0].fold == 1
    assert folds[0].train_end < folds[0].test_start
    summary = summarize_walk_forward(folds)
    assert summary["folds"] == len(folds)
    assert "mean_sharpe" in summary


def test_walk_forward_rejects_short_data():
    df = make_synthetic_pair(SyntheticPairConfig(periods=50))

    try:
        iter_walk_forward_splits(df, WalkForwardConfig(train_size=40, test_size=20))
    except ValueError as exc:
        assert "not enough rows" in str(exc)
    else:
        raise AssertionError("expected ValueError")
