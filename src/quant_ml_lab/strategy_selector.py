from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from quant_ml_lab.sizing import rolling_cvar_multiplier
from quant_ml_lab.validation import BacktestMetrics, compute_metrics

StrategyName = Literal[
    "no_trade",
    "mean_reversion_full",
    "mean_reversion_low_risk",
    "volatility_defensive",
    "cvar_defensive",
]


@dataclass(frozen=True)
class StrategyCandidate:
    name: StrategyName
    description: str
    position: pd.Series
    multiplier: pd.Series


@dataclass(frozen=True)
class StrategySelectorConfig:
    high_vol_threshold: float = 0.5
    defensive_multiplier: float = 0.35
    low_risk_multiplier: float = 0.5
    cvar_window: int = 63
    cvar_alpha: float = 0.1
    cvar_target_loss: float = -0.01
    cvar_floor: float = 0.2
    transaction_cost_bps: float = 2.0


@dataclass(frozen=True)
class StrategySelectionReport:
    selected_counts: dict[str, int]
    selected_metrics: BacktestMetrics
    candidate_metrics: dict[str, dict[str, float | int]]
    mean_risk_budget: float
    min_risk_budget: float
    max_risk_budget: float
    disclosure: str = (
        "Rule-based public meta-controller. This is a scaffold for future RL strategy selection."
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "selected_counts": self.selected_counts,
            "selected_metrics": self.selected_metrics.as_dict(),
            "candidate_metrics": self.candidate_metrics,
            "mean_risk_budget": self.mean_risk_budget,
            "min_risk_budget": self.min_risk_budget,
            "max_risk_budget": self.max_risk_budget,
            "disclosure": self.disclosure,
        }


def build_strategy_candidates(
    frame: pd.DataFrame,
    config: StrategySelectorConfig | None = None,
) -> dict[StrategyName, StrategyCandidate]:
    cfg = config or StrategySelectorConfig()
    _validate_selector_frame(frame)
    baseline = frame["baseline_position"].astype(float)
    zero = pd.Series(0.0, index=frame.index)
    one = pd.Series(1.0, index=frame.index)
    low_risk = pd.Series(cfg.low_risk_multiplier, index=frame.index)
    high_vol_multiplier = pd.Series(1.0, index=frame.index)
    high_vol_multiplier.loc[frame["high_vol_prob"] >= cfg.high_vol_threshold] = (
        cfg.defensive_multiplier
    )
    cvar_multiplier = rolling_cvar_multiplier(
        _baseline_returns(frame, cfg.transaction_cost_bps),
        window=cfg.cvar_window,
        alpha=cfg.cvar_alpha,
        target_loss=cfg.cvar_target_loss,
        floor=cfg.cvar_floor,
    )

    return {
        "no_trade": StrategyCandidate(
            name="no_trade",
            description="Cash policy. No position.",
            position=zero,
            multiplier=zero,
        ),
        "mean_reversion_full": StrategyCandidate(
            name="mean_reversion_full",
            description="Full baseline mean-reversion position.",
            position=baseline,
            multiplier=one,
        ),
        "mean_reversion_low_risk": StrategyCandidate(
            name="mean_reversion_low_risk",
            description="Baseline mean-reversion position with fixed low-risk multiplier.",
            position=baseline * low_risk,
            multiplier=low_risk,
        ),
        "volatility_defensive": StrategyCandidate(
            name="volatility_defensive",
            description="Baseline position reduced in high-volatility regimes.",
            position=baseline * high_vol_multiplier,
            multiplier=high_vol_multiplier,
        ),
        "cvar_defensive": StrategyCandidate(
            name="cvar_defensive",
            description="Baseline position scaled by rolling empirical CVaR.",
            position=baseline * cvar_multiplier,
            multiplier=cvar_multiplier,
        ),
    }


def select_strategy_by_regime(
    frame: pd.DataFrame,
    candidates: dict[StrategyName, StrategyCandidate],
    config: StrategySelectorConfig | None = None,
) -> tuple[pd.Series, pd.Series]:
    cfg = config or StrategySelectorConfig()
    _validate_selector_frame(frame)
    selected = pd.Series("mean_reversion_low_risk", index=frame.index, dtype=object)
    selected.loc[frame["baseline_position"].abs() == 0.0] = "no_trade"
    selected.loc[frame["feature_baseline_drawdown"] >= 0.25] = "cvar_defensive"
    selected.loc[frame["high_vol_prob"] >= cfg.high_vol_threshold] = "volatility_defensive"

    risk_budget = pd.Series(0.0, index=frame.index, dtype=float)
    for name, candidate in candidates.items():
        mask = selected == name
        risk_budget.loc[mask] = candidate.multiplier.loc[mask].abs()
    return selected.astype(str), risk_budget.clip(0.0, 1.0)


def apply_strategy_selection(
    frame: pd.DataFrame,
    candidates: dict[StrategyName, StrategyCandidate],
    selected: pd.Series,
    config: StrategySelectorConfig | None = None,
) -> tuple[pd.DataFrame, StrategySelectionReport]:
    cfg = config or StrategySelectorConfig()
    _validate_selector_frame(frame)
    selected_position = pd.Series(0.0, index=frame.index)
    selected_multiplier = pd.Series(0.0, index=frame.index)
    for name, candidate in candidates.items():
        mask = selected == name
        selected_position.loc[mask] = candidate.position.loc[mask]
        selected_multiplier.loc[mask] = candidate.multiplier.loc[mask]

    result = frame.copy()
    result["selected_strategy"] = selected
    result["selected_multiplier"] = selected_multiplier
    result["selected_position"] = selected_position
    result["selected_net_return"] = _returns_from_position(
        frame,
        selected_position,
        cfg.transaction_cost_bps,
    )
    result["selected_equity"] = (1.0 + result["selected_net_return"]).cumprod()
    selected_turnover = selected_position.diff().abs().fillna(selected_position.abs())
    candidate_metrics = {
        name: compute_metrics(
            _returns_from_position(frame, candidate.position, cfg.transaction_cost_bps),
            candidate.position.diff().abs().fillna(candidate.position.abs()),
        ).as_dict()
        for name, candidate in candidates.items()
    }
    report = StrategySelectionReport(
        selected_counts={str(key): int(value) for key, value in selected.value_counts().items()},
        selected_metrics=compute_metrics(result["selected_net_return"], selected_turnover),
        candidate_metrics=candidate_metrics,
        mean_risk_budget=float(selected_multiplier.abs().mean()),
        min_risk_budget=float(selected_multiplier.abs().min()),
        max_risk_budget=float(selected_multiplier.abs().max()),
    )
    return result, report


def run_strategy_selector(
    frame: pd.DataFrame,
    config: StrategySelectorConfig | None = None,
) -> tuple[pd.DataFrame, StrategySelectionReport]:
    cfg = config or StrategySelectorConfig()
    candidates = build_strategy_candidates(frame, cfg)
    selected, _ = select_strategy_by_regime(frame, candidates, cfg)
    return apply_strategy_selection(frame, candidates, selected, cfg)


def _validate_selector_frame(frame: pd.DataFrame) -> None:
    required = {
        "baseline_position",
        "spread_return_next",
        "high_vol_prob",
        "feature_baseline_drawdown",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")


def _baseline_returns(frame: pd.DataFrame, transaction_cost_bps: float) -> pd.Series:
    return _returns_from_position(frame, frame["baseline_position"], transaction_cost_bps)


def _returns_from_position(
    frame: pd.DataFrame,
    position: pd.Series,
    transaction_cost_bps: float,
) -> pd.Series:
    gross_return = position * -frame["spread_return_next"]
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (transaction_cost_bps / 10_000.0)
    return gross_return - cost
