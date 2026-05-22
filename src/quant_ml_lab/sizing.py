from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from quant_ml_lab.risk import cvar_position_multiplier, empirical_cvar
from quant_ml_lab.validation import BacktestMetrics, compute_metrics

SizingPolicy = Literal["base", "sac_proxy", "ppo_proxy", "qrdqn_cvar_proxy"]


@dataclass(frozen=True)
class SizingComparison:
    policy: SizingPolicy
    description: str
    metrics: BacktestMetrics
    mean_multiplier: float
    min_multiplier: float
    max_multiplier: float

    def as_dict(self) -> dict[str, object]:
        return {
            "policy": self.policy,
            "description": self.description,
            "metrics": self.metrics.as_dict(),
            "mean_multiplier": self.mean_multiplier,
            "min_multiplier": self.min_multiplier,
            "max_multiplier": self.max_multiplier,
        }


def rolling_cvar_multiplier(
    returns: pd.Series,
    window: int = 63,
    alpha: float = 0.1,
    target_loss: float = -0.02,
    floor: float = 0.0,
) -> pd.Series:
    if window < 5:
        raise ValueError("window must be at least 5")
    multipliers: list[float] = []
    values = returns.fillna(0.0)
    for idx in range(len(values)):
        start = max(0, idx - window + 1)
        sample = values.iloc[start : idx + 1].to_numpy(dtype=float)
        if sample.size < 5:
            multipliers.append(1.0)
            continue
        cvar = empirical_cvar(sample, alpha=alpha)
        multipliers.append(cvar_position_multiplier(cvar, target_loss=target_loss, floor=floor))
    return pd.Series(multipliers, index=returns.index, name="qrdqn_cvar_multiplier")


def policy_multiplier(
    baseline_result: pd.DataFrame,
    policy: SizingPolicy,
) -> pd.Series:
    index = baseline_result.index
    if policy == "base":
        return pd.Series(1.0, index=index, name="base_multiplier")

    zscore = baseline_result["zscore"].abs().clip(0.0, 3.0)
    if policy == "sac_proxy":
        multiplier = 0.25 + 0.75 * np.tanh(zscore / 2.0)
        return pd.Series(multiplier, index=index, name="sac_proxy_multiplier")

    if policy == "ppo_proxy":
        multiplier = (zscore / 2.0).clip(0.25, 1.0)
        return pd.Series(multiplier, index=index, name="ppo_proxy_multiplier")

    if policy == "qrdqn_cvar_proxy":
        return rolling_cvar_multiplier(
            baseline_result["net_return"],
            window=63,
            alpha=0.1,
            target_loss=-0.01,
            floor=0.2,
        )

    raise ValueError(f"unknown policy: {policy}")


def apply_sizing_policy(
    baseline_result: pd.DataFrame,
    policy: SizingPolicy,
) -> tuple[pd.DataFrame, SizingComparison]:
    multiplier = policy_multiplier(baseline_result, policy)
    result = baseline_result.copy()
    sized_position = result["position"] * multiplier
    spread_return = result["spread"].diff().fillna(0.0)
    gross_return = sized_position.shift(1).fillna(0.0) * -spread_return
    turnover = sized_position.diff().abs().fillna(sized_position.abs())
    cost_bps = _infer_cost_bps(result)
    cost = turnover * (cost_bps / 10_000.0)
    net_return = gross_return - cost

    result[f"{policy}_multiplier"] = multiplier
    result[f"{policy}_position"] = sized_position
    result[f"{policy}_net_return"] = net_return
    result[f"{policy}_equity"] = (1.0 + net_return).cumprod()
    comparison = SizingComparison(
        policy=policy,
        description=_policy_description(policy),
        metrics=compute_metrics(net_return, turnover),
        mean_multiplier=float(multiplier.mean()),
        min_multiplier=float(multiplier.min()),
        max_multiplier=float(multiplier.max()),
    )
    return result, comparison


def compare_sizing_policies(
    baseline_result: pd.DataFrame,
    policies: tuple[SizingPolicy, ...] = ("base", "sac_proxy", "ppo_proxy", "qrdqn_cvar_proxy"),
) -> list[SizingComparison]:
    return [apply_sizing_policy(baseline_result, policy)[1] for policy in policies]


def _policy_description(policy: SizingPolicy) -> str:
    descriptions = {
        "base": "Unscaled baseline position.",
        "sac_proxy": "Continuous tanh-style sizing proxy based on signal strength.",
        "ppo_proxy": "Clipped sizing proxy based on signal strength.",
        "qrdqn_cvar_proxy": "Distributional-RL-style CVaR proxy using rolling lower-tail returns.",
    }
    return descriptions[policy]


def _infer_cost_bps(result: pd.DataFrame) -> float:
    turnover = result["position"].diff().abs().fillna(result["position"].abs())
    active = turnover > 0
    if not active.any():
        return 0.0
    inferred = result.loc[active, "cost"] / turnover.loc[active]
    return float(inferred.median() * 10_000.0)
