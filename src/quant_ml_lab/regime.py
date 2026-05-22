from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_ml_lab.validation import (
    BacktestConfig,
    BacktestMetrics,
    backtest_pair_baseline,
    compute_metrics,
)


@dataclass(frozen=True)
class RegimeFilterConfig:
    vol_lookback: int = 20
    high_vol_quantile: float = 0.75
    high_vol_multiplier: float = 0.35


@dataclass(frozen=True)
class RegimeFilterReport:
    threshold: float
    high_vol_share: float
    baseline: BacktestMetrics
    filtered: BacktestMetrics

    def as_dict(self) -> dict[str, object]:
        return {
            "threshold": self.threshold,
            "high_vol_share": self.high_vol_share,
            "baseline": self.baseline.as_dict(),
            "filtered": self.filtered.as_dict(),
        }


def rolling_spread_volatility(spread: pd.Series, lookback: int) -> pd.Series:
    if lookback < 5:
        raise ValueError("lookback must be at least 5")
    return spread.diff().fillna(0.0).rolling(lookback).std(ddof=0).fillna(0.0)


def estimate_high_vol_threshold(train_spread: pd.Series, config: RegimeFilterConfig) -> float:
    if not 0.0 < config.high_vol_quantile < 1.0:
        raise ValueError("high_vol_quantile must be between 0 and 1")
    vol = rolling_spread_volatility(train_spread, config.vol_lookback)
    return float(vol.quantile(config.high_vol_quantile))


def apply_regime_position_filter(
    baseline_result: pd.DataFrame,
    threshold: float,
    config: RegimeFilterConfig,
) -> pd.DataFrame:
    if not 0.0 <= config.high_vol_multiplier <= 1.0:
        raise ValueError("high_vol_multiplier must be between 0 and 1")
    result = baseline_result.copy()
    spread_vol = rolling_spread_volatility(result["spread"], config.vol_lookback)
    is_high_vol = spread_vol > threshold
    multiplier = pd.Series(1.0, index=result.index, name="regime_multiplier")
    multiplier.loc[is_high_vol] = config.high_vol_multiplier

    filtered_position = result["position"] * multiplier
    spread_return = result["spread"].diff().fillna(0.0)
    gross_return = filtered_position.shift(1).fillna(0.0) * -spread_return
    turnover = filtered_position.diff().abs().fillna(filtered_position.abs())
    inferred_cost_bps = _infer_cost_bps(result)
    cost = turnover * (inferred_cost_bps / 10_000.0)
    net_return = gross_return - cost

    result["spread_volatility"] = spread_vol
    result["regime"] = np.where(is_high_vol, "high_vol", "normal")
    result["regime_multiplier"] = multiplier
    result["filtered_position"] = filtered_position
    result["filtered_gross_return"] = gross_return
    result["filtered_cost"] = cost
    result["filtered_net_return"] = net_return
    result["filtered_equity"] = (1.0 + net_return).cumprod()
    return result


def backtest_regime_filter(
    train: pd.DataFrame,
    test: pd.DataFrame,
    bt_config: BacktestConfig | None = None,
    regime_config: RegimeFilterConfig | None = None,
) -> tuple[pd.DataFrame, RegimeFilterReport]:
    bt_cfg = bt_config or BacktestConfig()
    reg_cfg = regime_config or RegimeFilterConfig()
    train_result, _ = backtest_pair_baseline(train, bt_cfg)
    test_result, baseline_metrics = backtest_pair_baseline(test, bt_cfg)
    threshold = estimate_high_vol_threshold(train_result["spread"], reg_cfg)
    filtered_result = apply_regime_position_filter(test_result, threshold, reg_cfg)
    filtered_metrics = compute_metrics(
        filtered_result["filtered_net_return"],
        filtered_result["filtered_position"].diff().abs().fillna(
            filtered_result["filtered_position"].abs()
        ),
    )
    high_vol_share = float((filtered_result["regime"] == "high_vol").mean())
    return filtered_result, RegimeFilterReport(
        threshold=threshold,
        high_vol_share=high_vol_share,
        baseline=baseline_metrics,
        filtered=filtered_metrics,
    )


def _infer_cost_bps(result: pd.DataFrame) -> float:
    turnover = result["position"].diff().abs().fillna(result["position"].abs())
    active = turnover > 0
    if not active.any():
        return 0.0
    inferred = result.loc[active, "cost"] / turnover.loc[active]
    return float(inferred.median() * 10_000.0)
