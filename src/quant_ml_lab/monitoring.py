from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DriftMetric:
    feature: str
    psi: float
    ks_distance: float
    status: str

    def as_dict(self) -> dict[str, float | str]:
        return {
            "feature": self.feature,
            "psi": self.psi,
            "ks_distance": self.ks_distance,
            "status": self.status,
        }


@dataclass(frozen=True)
class RollingPerformance:
    latest_rolling_return: float
    latest_rolling_sharpe: float
    latest_rolling_max_drawdown: float

    def as_dict(self) -> dict[str, float]:
        return {
            "latest_rolling_return": self.latest_rolling_return,
            "latest_rolling_sharpe": self.latest_rolling_sharpe,
            "latest_rolling_max_drawdown": self.latest_rolling_max_drawdown,
        }


def population_stability_index(
    expected: pd.Series,
    actual: pd.Series,
    bins: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """Compute PSI for a single feature.

    This is a public-demo monitoring primitive. Production thresholds and alert
    routing are intentionally out of scope.
    """
    expected_values = _clean_numeric(expected)
    actual_values = _clean_numeric(actual)
    if expected_values.empty or actual_values.empty:
        raise ValueError("expected and actual must contain numeric values")
    if bins < 2:
        raise ValueError("bins must be at least 2")

    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.quantile(expected_values, quantiles)
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0

    expected_counts, _ = np.histogram(expected_values, bins=edges)
    actual_counts, _ = np.histogram(actual_values, bins=edges)
    expected_pct = expected_counts / max(expected_counts.sum(), 1)
    actual_pct = actual_counts / max(actual_counts.sum(), 1)
    expected_pct = np.clip(expected_pct, epsilon, None)
    actual_pct = np.clip(actual_pct, epsilon, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def ks_distance(expected: pd.Series, actual: pd.Series) -> float:
    expected_values = np.sort(_clean_numeric(expected).to_numpy())
    actual_values = np.sort(_clean_numeric(actual).to_numpy())
    if expected_values.size == 0 or actual_values.size == 0:
        raise ValueError("expected and actual must contain numeric values")

    combined = np.sort(np.concatenate([expected_values, actual_values]))
    expected_cdf = np.searchsorted(expected_values, combined, side="right") / expected_values.size
    actual_cdf = np.searchsorted(actual_values, combined, side="right") / actual_values.size
    return float(np.max(np.abs(expected_cdf - actual_cdf)))


def drift_status(psi: float, warn: float = 0.1, alert: float = 0.25) -> str:
    if warn <= 0 or alert <= warn:
        raise ValueError("thresholds must satisfy 0 < warn < alert")
    if psi >= alert:
        return "alert"
    if psi >= warn:
        return "warn"
    return "ok"


def feature_drift_report(
    expected: pd.DataFrame,
    actual: pd.DataFrame,
    features: list[str],
    bins: int = 10,
) -> list[DriftMetric]:
    missing = [feature for feature in features if feature not in expected or feature not in actual]
    if missing:
        raise ValueError(f"missing features: {missing}")
    metrics: list[DriftMetric] = []
    for feature in features:
        psi = population_stability_index(expected[feature], actual[feature], bins=bins)
        ks = ks_distance(expected[feature], actual[feature])
        metrics.append(
            DriftMetric(feature=feature, psi=psi, ks_distance=ks, status=drift_status(psi))
        )
    return metrics


def rolling_performance(returns: pd.Series, window: int = 63) -> RollingPerformance:
    values = returns.fillna(0.0)
    if window <= 1:
        raise ValueError("window must be greater than 1")
    if len(values) < window:
        raise ValueError("not enough returns for rolling window")

    window_returns = values.iloc[-window:]
    equity = (1.0 + window_returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0)
    volatility = float(window_returns.std(ddof=0) * np.sqrt(252.0))
    annualized_return = float((1.0 + total_return) ** (252.0 / window) - 1.0)
    sharpe = annualized_return / volatility if volatility > 0 else 0.0
    drawdown = equity / equity.cummax() - 1.0
    return RollingPerformance(
        latest_rolling_return=total_return,
        latest_rolling_sharpe=float(sharpe),
        latest_rolling_max_drawdown=float(drawdown.min()),
    )


def _clean_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").dropna()
