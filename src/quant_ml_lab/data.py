from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticPairConfig:
    periods: int = 520
    seed: int = 7
    start: str = "2020-01-01"
    beta: float = 1.15
    spread_half_life: float = 18.0
    noise_scale: float = 0.015


def make_synthetic_pair(config: SyntheticPairConfig | None = None) -> pd.DataFrame:
    """Create a synthetic, non-proprietary pair dataset for public examples.

    The generated data intentionally does not encode any production universe,
    private feature recipe, or trading parameter.
    """
    cfg = config or SyntheticPairConfig()
    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range(cfg.start, periods=cfg.periods)

    market_returns = rng.normal(0.0003, 0.012, cfg.periods)
    asset_a = 100.0 * np.exp(np.cumsum(market_returns))

    mean_reversion = np.exp(np.log(0.5) / cfg.spread_half_life)
    spread = np.zeros(cfg.periods)
    for i in range(1, cfg.periods):
        spread[i] = mean_reversion * spread[i - 1] + rng.normal(0.0, cfg.noise_scale)

    asset_b = (asset_a / cfg.beta) * np.exp(spread)
    return pd.DataFrame({"date": dates, "asset_a": asset_a, "asset_b": asset_b}).set_index("date")


def train_test_split_time(df: pd.DataFrame, train_fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.1 < train_fraction < 0.95:
        raise ValueError("train_fraction must be between 0.1 and 0.95")
    split = int(len(df) * train_fraction)
    return df.iloc[:split].copy(), df.iloc[split:].copy()
