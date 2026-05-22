from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    entry_z: float = 1.25
    exit_z: float = 0.25
    lookback: int = 40
    transaction_cost_bps: float = 2.0


@dataclass(frozen=True)
class BacktestMetrics:
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    turnover: float
    trades: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "total_return": self.total_return,
            "annualized_return": self.annualized_return,
            "annualized_volatility": self.annualized_volatility,
            "sharpe": self.sharpe,
            "sortino": self.sortino,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "turnover": self.turnover,
            "trades": self.trades,
        }


def rolling_zscore(series: pd.Series, lookback: int) -> pd.Series:
    if lookback < 5:
        raise ValueError("lookback must be at least 5")
    mean = series.rolling(lookback).mean()
    std = series.rolling(lookback).std(ddof=0)
    return ((series - mean) / std.replace(0, np.nan)).fillna(0.0)


def pair_spread(df: pd.DataFrame) -> pd.Series:
    required = {"asset_a", "asset_b"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return np.log(df["asset_a"]) - np.log(df["asset_b"])


def generate_mean_reversion_positions(zscore: pd.Series, entry_z: float, exit_z: float) -> pd.Series:
    if entry_z <= exit_z:
        raise ValueError("entry_z must be greater than exit_z")

    positions: list[int] = []
    position = 0
    for z in zscore:
        if position == 0:
            if z > entry_z:
                position = -1
            elif z < -entry_z:
                position = 1
        elif abs(z) < exit_z:
            position = 0
        positions.append(position)
    return pd.Series(positions, index=zscore.index, name="position")


def backtest_pair_baseline(df: pd.DataFrame, config: BacktestConfig | None = None) -> tuple[pd.DataFrame, BacktestMetrics]:
    cfg = config or BacktestConfig()
    spread = pair_spread(df)
    zscore = rolling_zscore(spread, cfg.lookback)
    position = generate_mean_reversion_positions(zscore, cfg.entry_z, cfg.exit_z)

    spread_return = spread.diff().fillna(0.0)
    gross_return = position.shift(1).fillna(0.0) * -spread_return
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (cfg.transaction_cost_bps / 10_000.0)
    net_return = gross_return - cost

    equity = (1.0 + net_return).cumprod()
    result = pd.DataFrame(
        {
            "spread": spread,
            "zscore": zscore,
            "position": position,
            "gross_return": gross_return,
            "cost": cost,
            "net_return": net_return,
            "equity": equity,
        },
        index=df.index,
    )
    return result, compute_metrics(net_return, turnover)


def compute_metrics(returns: pd.Series, turnover: pd.Series) -> BacktestMetrics:
    returns = returns.fillna(0.0)
    equity = (1.0 + returns).cumprod()
    total_return = float(equity.iloc[-1] - 1.0) if len(equity) else 0.0
    ann_return = float((1.0 + total_return) ** (252.0 / max(len(returns), 1)) - 1.0)
    ann_vol = float(returns.std(ddof=0) * np.sqrt(252.0))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else 0.0
    downside = returns[returns < 0.0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(252.0)) if len(downside) else 0.0
    sortino = float(ann_return / downside_vol) if downside_vol > 0 else 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    max_drawdown = float(drawdown.min()) if len(drawdown) else 0.0
    active_returns = returns[returns != 0.0]
    win_rate = float((active_returns > 0.0).mean()) if len(active_returns) else 0.0
    total_turnover = float(turnover.sum())
    trades = int((turnover > 0).sum())
    return BacktestMetrics(
        total_return=total_return,
        annualized_return=ann_return,
        annualized_volatility=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        turnover=total_turnover,
        trades=trades,
    )
