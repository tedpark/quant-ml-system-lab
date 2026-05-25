from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_ml_lab.regime import rolling_spread_volatility
from quant_ml_lab.validation import BacktestConfig, backtest_pair_baseline


@dataclass(frozen=True)
class GaussianHMMConfig:
    stay_prob: float = 0.96
    high_state_quantile: float = 0.70
    min_std: float = 1e-6


class ForwardGaussianHMM2State:
    """Small forward-only 2-state Gaussian HMM for public regime demos.

    This is intentionally minimal and transparent. It fits emission parameters
    on train data, then uses only recursive forward probabilities on test data.
    """

    def __init__(self, config: GaussianHMMConfig | None = None) -> None:
        self.config = config or GaussianHMMConfig()
        self.means: np.ndarray | None = None
        self.stds: np.ndarray | None = None
        self.transition = np.array(
            [
                [self.config.stay_prob, 1.0 - self.config.stay_prob],
                [1.0 - self.config.stay_prob, self.config.stay_prob],
            ],
            dtype=float,
        )

    def fit(self, observations: pd.Series) -> ForwardGaussianHMM2State:
        values = observations.dropna().to_numpy(dtype=float)
        if values.size < 20:
            raise ValueError("at least 20 observations are required")
        if not 0.0 < self.config.high_state_quantile < 1.0:
            raise ValueError("high_state_quantile must be between 0 and 1")

        threshold = float(np.quantile(values, self.config.high_state_quantile))
        low = values[values <= threshold]
        high = values[values > threshold]
        if low.size == 0 or high.size == 0:
            raise ValueError("observations must produce both states")

        self.means = np.array([low.mean(), high.mean()], dtype=float)
        self.stds = np.maximum(
            np.array([low.std(ddof=0), high.std(ddof=0)], dtype=float),
            self.config.min_std,
        )
        return self

    def predict_proba_forward(self, observations: pd.Series) -> pd.DataFrame:
        if self.means is None or self.stds is None:
            raise RuntimeError("fit must be called first")

        alpha = np.array([0.5, 0.5], dtype=float)
        rows: list[np.ndarray] = []
        for value in observations.fillna(0.0).to_numpy(dtype=float):
            predicted = alpha @ self.transition
            likelihood = self._emission_likelihood(value)
            alpha = predicted * likelihood
            alpha = alpha / alpha.sum()
            rows.append(alpha.copy())
        return pd.DataFrame(rows, index=observations.index, columns=["normal_prob", "high_vol_prob"])

    def _emission_likelihood(self, value: float) -> np.ndarray:
        assert self.means is not None
        assert self.stds is not None
        z = (value - self.means) / self.stds
        return np.exp(-0.5 * z * z) / (self.stds * np.sqrt(2.0 * np.pi))


@dataclass(frozen=True)
class HMMRLDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    disclosure: str = "Synthetic forward-HMM + RL sizing demo. No production strategy logic."


def build_hmm_rl_dataset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    bt_config: BacktestConfig | None = None,
    hmm_config: GaussianHMMConfig | None = None,
    vol_lookback: int = 20,
) -> HMMRLDataset:
    bt_cfg = bt_config or BacktestConfig()
    train_result, _ = backtest_pair_baseline(train, bt_cfg)
    test_result, _ = backtest_pair_baseline(test, bt_cfg)

    train_vol = rolling_spread_volatility(train_result["spread"], vol_lookback)
    test_vol = rolling_spread_volatility(test_result["spread"], vol_lookback)
    hmm = ForwardGaussianHMM2State(hmm_config).fit(train_vol)
    probs = hmm.predict_proba_forward(test_vol)

    frame = test_result.join(probs)
    frame["abs_zscore"] = frame["zscore"].abs().clip(0.0, 4.0)
    frame["spread_return"] = frame["spread"].diff().fillna(0.0)
    frame["spread_return_next"] = frame["spread"].diff().shift(-1).fillna(0.0)
    frame["baseline_position"] = frame["position"].astype(float)
    frame["baseline_turnover"] = frame["position"].diff().abs().fillna(frame["position"].abs())
    frame["baseline_net_return"] = frame["position"].shift(1).fillna(0.0) * -frame[
        "spread_return"
    ] - frame["baseline_turnover"] * (bt_cfg.transaction_cost_bps / 10_000.0)
    baseline_equity = (1.0 + frame["baseline_net_return"]).cumprod()
    baseline_drawdown = (baseline_equity / baseline_equity.cummax() - 1.0).fillna(0.0)
    train_vol_median = float(train_vol.replace(0.0, np.nan).median())
    if not np.isfinite(train_vol_median) or train_vol_median <= 0.0:
        train_vol_median = 1.0

    frame["feature_zscore"] = (frame["zscore"].clip(-4.0, 4.0) / 4.0).fillna(0.0)
    frame["feature_abs_zscore"] = (frame["abs_zscore"] / 4.0).fillna(0.0)
    frame["feature_high_vol_prob"] = frame["high_vol_prob"].fillna(0.5)
    frame["feature_regime_transition"] = frame["high_vol_prob"].diff().abs().clip(0.0, 1.0).fillna(0.0)
    frame["feature_position"] = frame["baseline_position"].fillna(0.0)
    frame["feature_spread_momentum"] = (
        frame["spread_return"].rolling(5).mean().fillna(0.0).clip(-0.03, 0.03) / 0.03
    )
    frame["feature_spread_volatility"] = (
        (test_vol / train_vol_median).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(0.0, 3.0)
        / 3.0
    )
    frame["feature_recent_pnl"] = (
        frame["baseline_net_return"].rolling(5).mean().fillna(0.0).clip(-0.02, 0.02) / 0.02
    )
    frame["feature_baseline_drawdown"] = baseline_drawdown.clip(-0.25, 0.0).abs() / 0.25

    feature_columns = (
        "feature_zscore",
        "feature_abs_zscore",
        "feature_high_vol_prob",
        "feature_regime_transition",
        "feature_position",
        "feature_spread_momentum",
        "feature_spread_volatility",
        "feature_recent_pnl",
        "feature_baseline_drawdown",
    )
    return HMMRLDataset(frame=frame, feature_columns=feature_columns)
