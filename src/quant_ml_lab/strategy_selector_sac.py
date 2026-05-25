from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant_ml_lab.strategy_selector import (
    StrategyCandidate,
    StrategyName,
    StrategySelectorConfig,
    build_strategy_candidates,
    run_strategy_selector,
)
from quant_ml_lab.torch_sac import (
    GaussianActor,
    TorchSACConfig,
    save_sac_actor_checkpoint,
    train_torch_sac,
)
from quant_ml_lab.validation import BacktestMetrics, compute_metrics

SAC_STRATEGY_ACTIONS: tuple[StrategyName, ...] = (
    "no_trade",
    "mean_reversion_full",
    "mean_reversion_low_risk",
    "volatility_defensive",
    "cvar_defensive",
)


@dataclass(frozen=True)
class SACStrategyAllocatorEnvConfig:
    transaction_cost_bps: float = 2.0
    turnover_penalty: float = 0.0005
    high_vol_penalty: float = 0.001
    drawdown_penalty: float = 0.002
    concentration_penalty: float = 0.0003
    softmax_temperature: float = 0.75


@dataclass(frozen=True)
class SACStrategyAllocatorReport:
    train_metrics: BacktestMetrics
    validation_metrics: BacktestMetrics
    rule_based_validation_metrics: BacktestMetrics
    candidate_validation_metrics: dict[str, dict[str, float | int]]
    validation_mean_weights: dict[str, float]
    validation_weight_concentration: float
    deterministic_action: float | list[float]
    reward_tail_mean: float
    actor_loss_tail_mean: float
    critic_loss_tail_mean: float
    alpha_final: float
    alpha_loss_tail_mean: float
    diagnostics: dict[str, bool | float]
    checkpoint_path: str | None
    disclosure: str = (
        "Public SAC strategy allocator scaffold. Synthetic data only. Not a live strategy."
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "train_metrics": self.train_metrics.as_dict(),
            "validation_metrics": self.validation_metrics.as_dict(),
            "rule_based_validation_metrics": self.rule_based_validation_metrics.as_dict(),
            "candidate_validation_metrics": self.candidate_validation_metrics,
            "validation_mean_weights": self.validation_mean_weights,
            "validation_weight_concentration": self.validation_weight_concentration,
            "deterministic_action": self.deterministic_action,
            "reward_tail_mean": self.reward_tail_mean,
            "actor_loss_tail_mean": self.actor_loss_tail_mean,
            "critic_loss_tail_mean": self.critic_loss_tail_mean,
            "alpha_final": self.alpha_final,
            "alpha_loss_tail_mean": self.alpha_loss_tail_mean,
            "diagnostics": self.diagnostics,
            "checkpoint_path": self.checkpoint_path,
            "disclosure": self.disclosure,
        }


class SACStrategyAllocatorEnv:
    """SAC environment that allocates continuous weights across strategy candidates."""

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
        candidates: dict[StrategyName, StrategyCandidate],
        selector_config: StrategySelectorConfig | None = None,
        env_config: SACStrategyAllocatorEnvConfig | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.candidates = {
            name: StrategyCandidate(
                name=candidate.name,
                description=candidate.description,
                position=candidate.position.reset_index(drop=True),
                multiplier=candidate.multiplier.reset_index(drop=True),
            )
            for name, candidate in candidates.items()
        }
        self.selector_config = selector_config or StrategySelectorConfig()
        self.env_config = env_config or SACStrategyAllocatorEnvConfig()
        self.cursor = 0
        self.prev_position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.net_returns: list[float] = []
        self.positions: list[float] = []
        self.weights: list[np.ndarray] = []

    @property
    def state_dim(self) -> int:
        return len(self.feature_columns)

    @property
    def action_dim(self) -> int:
        return len(SAC_STRATEGY_ACTIONS)

    def reset(self) -> np.ndarray:
        self.cursor = 0
        self.prev_position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        self.net_returns = []
        self.positions = []
        self.weights = []
        return self._state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        weights = _action_to_weights(action, self.env_config.softmax_temperature)
        row = self.frame.iloc[self.cursor]
        candidate_positions = np.asarray(
            [
                float(self.candidates[name].position.iloc[self.cursor])
                for name in SAC_STRATEGY_ACTIONS
            ],
            dtype=np.float64,
        )
        position = float(weights @ candidate_positions)
        gross_return = position * -float(row["spread_return_next"])
        turnover = abs(position - self.prev_position)
        cost = turnover * (self.selector_config.transaction_cost_bps / 10_000.0)
        net_return = gross_return - cost
        next_equity = self.equity * (1.0 + net_return)
        next_peak = max(self.peak_equity, next_equity)
        drawdown = max(0.0, 1.0 - next_equity / next_peak)
        concentration = float(np.square(weights).sum())
        reward = (
            net_return
            - turnover * self.env_config.turnover_penalty
            - float(row["high_vol_prob"]) * abs(position) * self.env_config.high_vol_penalty
            - drawdown * self.env_config.drawdown_penalty
            - concentration * self.env_config.concentration_penalty
        )

        self.prev_position = position
        self.equity = next_equity
        self.peak_equity = next_peak
        self.net_returns.append(net_return)
        self.positions.append(position)
        self.weights.append(weights)
        self.cursor += 1
        done = self.cursor >= len(self.frame)
        next_state = self._state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        return next_state, float(reward), done

    def _state(self) -> np.ndarray:
        return self.frame.loc[self.cursor, list(self.feature_columns)].to_numpy(dtype=np.float32)


def train_validate_strategy_allocator_sac(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    selector_config: StrategySelectorConfig | None = None,
    env_config: SACStrategyAllocatorEnvConfig | None = None,
    sac_config: TorchSACConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[SACStrategyAllocatorReport, pd.DataFrame, pd.DataFrame]:
    selector_cfg = selector_config or StrategySelectorConfig()
    env_cfg = env_config or SACStrategyAllocatorEnvConfig()
    train_candidates = build_strategy_candidates(train_frame, selector_cfg)
    validation_candidates = build_strategy_candidates(validation_frame, selector_cfg)
    env = SACStrategyAllocatorEnv(train_frame, feature_columns, train_candidates, selector_cfg, env_cfg)
    result = train_torch_sac(
        env,
        sac_config
        or TorchSACConfig(
            steps=700,
            warmup_steps=96,
            batch_size=48,
            gamma=0.95,
            hidden_dim=64,
            target_entropy=-float(len(SAC_STRATEGY_ACTIONS)),
        ),
    )
    train_output, train_metrics, _ = evaluate_strategy_allocator_sac(
        train_frame,
        feature_columns,
        train_candidates,
        result.actor,
        selector_cfg,
        env_cfg,
    )
    validation_output, validation_metrics, validation_mean_weights = evaluate_strategy_allocator_sac(
        validation_frame,
        feature_columns,
        validation_candidates,
        result.actor,
        selector_cfg,
        env_cfg,
    )
    _, rule_report = run_strategy_selector(validation_frame, selector_cfg)

    saved_path: str | None = None
    if checkpoint_path is not None:
        save_sac_actor_checkpoint(result, checkpoint_path)
        saved_path = str(checkpoint_path)

    validation_weight_concentration = float(sum(value**2 for value in validation_mean_weights.values()))
    alpha_final = float(result.alpha_trace[-1]) if result.alpha_trace else float(result.config.alpha)
    actor_loss_tail_mean = (
        float(np.mean(result.actor_loss_trace[-20:])) if result.actor_loss_trace else 0.0
    )
    critic_loss_tail_mean = (
        float(np.mean(result.critic_loss_trace[-20:])) if result.critic_loss_trace else 0.0
    )
    alpha_loss_tail_mean = (
        float(np.mean(result.alpha_loss_trace[-20:])) if result.alpha_loss_trace else 0.0
    )
    diagnostics = {
        "actor_loss_is_finite": bool(np.isfinite(actor_loss_tail_mean)),
        "critic_loss_is_finite": bool(np.isfinite(critic_loss_tail_mean)),
        "alpha_is_positive": alpha_final > 0.0,
        "beats_rule_based_sharpe": validation_metrics.sharpe >= rule_report.selected_metrics.sharpe,
        "weight_concentration": validation_weight_concentration,
    }
    report = SACStrategyAllocatorReport(
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        rule_based_validation_metrics=rule_report.selected_metrics,
        candidate_validation_metrics=rule_report.candidate_metrics,
        validation_mean_weights=validation_mean_weights,
        validation_weight_concentration=validation_weight_concentration,
        deterministic_action=result.deterministic_action,
        reward_tail_mean=float(np.mean(result.reward_trace[-50:])) if result.reward_trace else 0.0,
        actor_loss_tail_mean=actor_loss_tail_mean,
        critic_loss_tail_mean=critic_loss_tail_mean,
        alpha_final=alpha_final,
        alpha_loss_tail_mean=alpha_loss_tail_mean,
        diagnostics=diagnostics,
        checkpoint_path=saved_path,
    )
    return report, train_output, validation_output


def evaluate_strategy_allocator_sac(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    candidates: dict[StrategyName, StrategyCandidate],
    actor: GaussianActor,
    selector_config: StrategySelectorConfig | None = None,
    env_config: SACStrategyAllocatorEnvConfig | None = None,
) -> tuple[pd.DataFrame, BacktestMetrics, dict[str, float]]:
    selector_cfg = selector_config or StrategySelectorConfig()
    env_cfg = env_config or SACStrategyAllocatorEnvConfig()
    actor.eval()
    positions: list[float] = []
    weights_by_row: list[np.ndarray] = []
    device = next(actor.parameters()).device
    for _, row in frame.iterrows():
        state = row.loc[list(feature_columns)].to_numpy(dtype=np.float32)
        with torch.no_grad():
            action = (
                actor.deterministic(torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0))
                .squeeze(0)
                .cpu()
                .numpy()
            )
        weights = _action_to_weights(action, env_cfg.softmax_temperature)
        candidate_positions = np.asarray(
            [float(candidates[name].position.loc[row.name]) for name in SAC_STRATEGY_ACTIONS],
            dtype=np.float64,
        )
        positions.append(float(weights @ candidate_positions))
        weights_by_row.append(weights)

    output = frame.copy()
    output["sac_allocator_position"] = positions
    output["sac_allocator_net_return"] = _returns_from_position(
        output,
        pd.Series(positions, index=output.index),
        selector_cfg.transaction_cost_bps,
    )
    output["sac_allocator_equity"] = (1.0 + output["sac_allocator_net_return"]).cumprod()
    weight_matrix = np.vstack(weights_by_row) if weights_by_row else np.zeros((0, len(SAC_STRATEGY_ACTIONS)))
    for index, name in enumerate(SAC_STRATEGY_ACTIONS):
        output[f"weight_{name}"] = weight_matrix[:, index] if len(weight_matrix) else []
    turnover = output["sac_allocator_position"].diff().abs().fillna(
        output["sac_allocator_position"].abs()
    )
    mean_weights = {
        name: float(weight_matrix[:, index].mean()) if len(weight_matrix) else 0.0
        for index, name in enumerate(SAC_STRATEGY_ACTIONS)
    }
    return output, compute_metrics(output["sac_allocator_net_return"], turnover), mean_weights


def _action_to_weights(action: np.ndarray, temperature: float) -> np.ndarray:
    safe_temperature = max(float(temperature), 1e-6)
    logits = np.asarray(action, dtype=np.float64) / safe_temperature
    logits = logits - logits.max()
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


def _returns_from_position(
    frame: pd.DataFrame,
    position: pd.Series,
    transaction_cost_bps: float,
) -> pd.Series:
    gross_return = position * -frame["spread_return_next"]
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (transaction_cost_bps / 10_000.0)
    return gross_return - cost
