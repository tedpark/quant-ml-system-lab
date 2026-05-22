from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import pandas as pd

from quant_ml_lab.validation import BacktestConfig, BacktestMetrics, backtest_pair_baseline


@dataclass(frozen=True)
class WalkForwardConfig:
    train_size: int = 252
    test_size: int = 63
    step_size: int = 63


@dataclass(frozen=True)
class WalkForwardFold:
    fold: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    metrics: BacktestMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "fold": self.fold,
            "train_start": self.train_start,
            "train_end": self.train_end,
            "test_start": self.test_start,
            "test_end": self.test_end,
            "metrics": self.metrics.as_dict(),
        }


BacktestFn = Callable[[pd.DataFrame, BacktestConfig], tuple[pd.DataFrame, BacktestMetrics]]


def iter_walk_forward_splits(
    df: pd.DataFrame, config: WalkForwardConfig
) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    """Return expanding-free, time-ordered walk-forward splits."""
    _validate_walk_forward_config(df, config)
    splits: list[tuple[pd.DataFrame, pd.DataFrame]] = []
    start = 0
    while start + config.train_size + config.test_size <= len(df):
        train = df.iloc[start : start + config.train_size].copy()
        test_start = start + config.train_size
        test = df.iloc[test_start : test_start + config.test_size].copy()
        splits.append((train, test))
        start += config.step_size
    return splits


def run_walk_forward(
    df: pd.DataFrame,
    wf_config: WalkForwardConfig | None = None,
    bt_config: BacktestConfig | None = None,
    backtest_fn: BacktestFn = backtest_pair_baseline,
) -> list[WalkForwardFold]:
    """Run a public-demo walk-forward evaluation on synthetic/sample data.

    The train window is returned for auditability, but this baseline does not
    tune parameters on train. Production parameter search is intentionally out
    of scope for the public repository.
    """
    wf_cfg = wf_config or WalkForwardConfig()
    bt_cfg = bt_config or BacktestConfig()
    folds: list[WalkForwardFold] = []
    for fold_idx, (train, test) in enumerate(iter_walk_forward_splits(df, wf_cfg), start=1):
        _, metrics = backtest_fn(test, bt_cfg)
        folds.append(
            WalkForwardFold(
                fold=fold_idx,
                train_start=_index_label(train.index[0]),
                train_end=_index_label(train.index[-1]),
                test_start=_index_label(test.index[0]),
                test_end=_index_label(test.index[-1]),
                metrics=metrics,
            )
        )
    return folds


def summarize_walk_forward(folds: Sequence[WalkForwardFold]) -> dict[str, float | int]:
    if not folds:
        raise ValueError("folds must not be empty")
    total_returns = [fold.metrics.total_return for fold in folds]
    sharpes = [fold.metrics.sharpe for fold in folds]
    max_drawdowns = [fold.metrics.max_drawdown for fold in folds]
    return {
        "folds": len(folds),
        "mean_total_return": float(pd.Series(total_returns).mean()),
        "mean_sharpe": float(pd.Series(sharpes).mean()),
        "worst_max_drawdown": float(pd.Series(max_drawdowns).min()),
    }


def _validate_walk_forward_config(df: pd.DataFrame, config: WalkForwardConfig) -> None:
    if config.train_size <= 0 or config.test_size <= 0 or config.step_size <= 0:
        raise ValueError("train_size, test_size, and step_size must be positive")
    if config.train_size + config.test_size > len(df):
        raise ValueError("not enough rows for one walk-forward fold")
    if not df.index.is_monotonic_increasing:
        raise ValueError("df index must be sorted in time order")


def _index_label(value: object) -> str:
    if hasattr(value, "date"):
        return str(value.date())
    return str(value)
