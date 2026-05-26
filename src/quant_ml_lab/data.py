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


@dataclass(frozen=True)
class SyntheticRegimeSegment:
    length: int
    market_drift: float
    market_vol: float
    spread_half_life: float
    spread_noise_scale: float
    beta: float = 1.15
    jump_prob: float = 0.0
    jump_scale: float = 0.0
    name: str = "regime"


@dataclass(frozen=True)
class SyntheticRegimePairConfig:
    periods: int = 780
    seed: int = 17
    start: str = "2020-01-01"
    initial_price: float = 100.0
    segments: tuple[SyntheticRegimeSegment, ...] = (
        SyntheticRegimeSegment(
            length=180,
            market_drift=0.0004,
            market_vol=0.009,
            spread_half_life=12.0,
            spread_noise_scale=0.010,
            name="calm_mean_reverting",
        ),
        SyntheticRegimeSegment(
            length=140,
            market_drift=-0.0002,
            market_vol=0.024,
            spread_half_life=35.0,
            spread_noise_scale=0.026,
            jump_prob=0.035,
            jump_scale=0.045,
            name="volatile_dislocated",
        ),
        SyntheticRegimeSegment(
            length=180,
            market_drift=0.0007,
            market_vol=0.013,
            spread_half_life=22.0,
            spread_noise_scale=0.016,
            name="trend_moderate",
        ),
        SyntheticRegimeSegment(
            length=140,
            market_drift=0.0,
            market_vol=0.018,
            spread_half_life=60.0,
            spread_noise_scale=0.022,
            jump_prob=0.02,
            jump_scale=0.030,
            name="slow_reversion",
        ),
    )


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


def make_synthetic_regime_pair(
    config: SyntheticRegimePairConfig | None = None,
) -> pd.DataFrame:
    """Create a synthetic pair with explicit regime shifts for robustness tests.

    This generator is deliberately stylized. It is for falsifying fragile public
    strategy scaffolds, not for claiming realism or live-trading edge.
    """
    cfg = config or SyntheticRegimePairConfig()
    if cfg.periods <= 0:
        raise ValueError("periods must be positive")
    if not cfg.segments:
        raise ValueError("at least one regime segment is required")

    rng = np.random.default_rng(cfg.seed)
    dates = pd.bdate_range(cfg.start, periods=cfg.periods)
    expanded = _expand_regime_segments(cfg.segments, cfg.periods)

    market_returns = np.zeros(cfg.periods, dtype=float)
    spread = np.zeros(cfg.periods, dtype=float)
    beta = np.zeros(cfg.periods, dtype=float)
    regime_names: list[str] = []
    market_vol = np.zeros(cfg.periods, dtype=float)
    spread_noise = np.zeros(cfg.periods, dtype=float)

    for i, segment in enumerate(expanded):
        if segment.length <= 0:
            raise ValueError("segment length must be positive")
        beta[i] = segment.beta
        regime_names.append(segment.name)
        market_vol[i] = segment.market_vol
        spread_noise[i] = segment.spread_noise_scale
        jump = (
            rng.normal(0.0, segment.jump_scale)
            if segment.jump_prob > 0.0 and rng.random() < segment.jump_prob
            else 0.0
        )
        market_returns[i] = rng.normal(segment.market_drift, segment.market_vol) + jump
        if i == 0:
            continue
        mean_reversion = np.exp(np.log(0.5) / max(segment.spread_half_life, 1e-6))
        spread_jump = (
            rng.normal(0.0, segment.jump_scale)
            if segment.jump_prob > 0.0 and rng.random() < segment.jump_prob
            else 0.0
        )
        spread[i] = (
            mean_reversion * spread[i - 1]
            + rng.normal(0.0, segment.spread_noise_scale)
            + spread_jump
        )

    asset_a = cfg.initial_price * np.exp(np.cumsum(market_returns))
    asset_b = (asset_a / beta) * np.exp(spread)
    return pd.DataFrame(
        {
            "date": dates,
            "asset_a": asset_a,
            "asset_b": asset_b,
            "synthetic_regime": regime_names,
            "synthetic_market_vol": market_vol,
            "synthetic_spread_noise": spread_noise,
        }
    ).set_index("date")


def train_test_split_time(df: pd.DataFrame, train_fraction: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.1 < train_fraction < 0.95:
        raise ValueError("train_fraction must be between 0.1 and 0.95")
    split = int(len(df) * train_fraction)
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def _expand_regime_segments(
    segments: tuple[SyntheticRegimeSegment, ...],
    periods: int,
) -> list[SyntheticRegimeSegment]:
    expanded: list[SyntheticRegimeSegment] = []
    while len(expanded) < periods:
        for segment in segments:
            expanded.extend([segment] * segment.length)
            if len(expanded) >= periods:
                break
    return expanded[:periods]
