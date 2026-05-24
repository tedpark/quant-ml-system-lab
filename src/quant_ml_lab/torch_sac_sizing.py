from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from quant_ml_lab.torch_sac import (
    GaussianActor,
    TorchSACConfig,
    save_sac_actor_checkpoint,
    train_torch_sac,
)
from quant_ml_lab.validation import BacktestMetrics, compute_metrics


@dataclass(frozen=True)
class HMMSizingEnvConfig:
    transaction_cost_bps: float = 2.0
    turnover_penalty: float = 0.0005
    high_vol_penalty: float = 0.001
    max_steps: int | None = None


class HMMSACPositionSizingEnv:
    """Sequential RL environment for HMM-aware position multiplier learning.

    Action is a continuous value in [-1, 1]. It is mapped to multiplier [0, 1].
    The multiplier scales the baseline position. The environment is synthetic
    and sanitized; it is not a production trading strategy.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
        config: HMMSizingEnvConfig | None = None,
    ) -> None:
        self.frame = frame.reset_index(drop=True).copy()
        self.feature_columns = feature_columns
        self.config = config or HMMSizingEnvConfig()
        self.cursor = 0
        self.prev_position = 0.0
        self.net_returns: list[float] = []
        self.positions: list[float] = []

    @property
    def state_dim(self) -> int:
        return len(self.feature_columns)

    @property
    def action_dim(self) -> int:
        return 1

    def reset(self) -> np.ndarray:
        self.cursor = 0
        self.prev_position = 0.0
        self.net_returns = []
        self.positions = []
        return self._state()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        raw_action = float(np.clip(action[0], -1.0, 1.0))
        multiplier = 0.5 * (raw_action + 1.0)
        row = self.frame.iloc[self.cursor]
        baseline_position = float(row["baseline_position"])
        sized_position = baseline_position * multiplier
        spread_return_next = float(row["spread_return_next"])
        gross_return = sized_position * -spread_return_next
        turnover = abs(sized_position - self.prev_position)
        cost = turnover * (self.config.transaction_cost_bps / 10_000.0)
        risk_penalty = float(row["high_vol_prob"]) * abs(sized_position) * self.config.high_vol_penalty
        reward = gross_return - cost - turnover * self.config.turnover_penalty - risk_penalty

        self.prev_position = sized_position
        self.net_returns.append(gross_return - cost)
        self.positions.append(sized_position)
        self.cursor += 1

        limit = self.config.max_steps or len(self.frame)
        done = self.cursor >= min(limit, len(self.frame))
        next_state = self._state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        return next_state, float(reward), done

    def _state(self) -> np.ndarray:
        return self.frame.loc[self.cursor, list(self.feature_columns)].to_numpy(dtype=np.float32)


@dataclass(frozen=True)
class HMMSACTrainingReport:
    deterministic_action: float
    mean_multiplier: float
    metrics: BacktestMetrics
    reward_tail_mean: float
    alpha_final: float
    actor_loss_tail_mean: float
    critic_loss_tail_mean: float

    def as_dict(self) -> dict[str, object]:
        return {
            "deterministic_action": self.deterministic_action,
            "mean_multiplier": self.mean_multiplier,
            "metrics": self.metrics.as_dict(),
            "reward_tail_mean": self.reward_tail_mean,
            "alpha_final": self.alpha_final,
            "actor_loss_tail_mean": self.actor_loss_tail_mean,
            "critic_loss_tail_mean": self.critic_loss_tail_mean,
        }


@dataclass(frozen=True)
class HMMSACValidationReport:
    train: HMMSACTrainingReport
    validation: HMMSACTrainingReport
    checkpoint_path: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "train": self.train.as_dict(),
            "validation": self.validation.as_dict(),
            "checkpoint_path": self.checkpoint_path,
        }


def train_hmm_sac_sizer(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    sac_config: TorchSACConfig | None = None,
    env_config: HMMSizingEnvConfig | None = None,
) -> tuple[HMMSACTrainingReport, pd.DataFrame]:
    env = HMMSACPositionSizingEnv(frame, feature_columns, env_config)
    result = train_torch_sac(env, sac_config or TorchSACConfig(steps=260, gamma=0.95))
    report, output = evaluate_hmm_sac_sizer(
        frame=frame,
        feature_columns=feature_columns,
        actor=result.actor,
        deterministic_action=result.deterministic_action,
        reward_tail_mean=float(np.mean(result.reward_trace[-20:])),
        alpha_final=float(result.alpha_trace[-1]) if result.alpha_trace else float(result.config.alpha),
        actor_loss_tail_mean=float(np.mean(result.actor_loss_trace[-20:]))
        if result.actor_loss_trace
        else 0.0,
        critic_loss_tail_mean=float(np.mean(result.critic_loss_trace[-20:]))
        if result.critic_loss_trace
        else 0.0,
        env_config=env_config,
    )
    return report, output


def train_validate_hmm_sac_sizer(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    sac_config: TorchSACConfig | None = None,
    env_config: HMMSizingEnvConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[HMMSACValidationReport, pd.DataFrame, pd.DataFrame]:
    env = HMMSACPositionSizingEnv(train_frame, feature_columns, env_config)
    result = train_torch_sac(env, sac_config or TorchSACConfig(steps=500, gamma=0.95))

    alpha_final = float(result.alpha_trace[-1]) if result.alpha_trace else float(result.config.alpha)
    actor_loss_tail_mean = (
        float(np.mean(result.actor_loss_trace[-20:])) if result.actor_loss_trace else 0.0
    )
    critic_loss_tail_mean = (
        float(np.mean(result.critic_loss_trace[-20:])) if result.critic_loss_trace else 0.0
    )
    train_report, train_output = evaluate_hmm_sac_sizer(
        frame=train_frame,
        feature_columns=feature_columns,
        actor=result.actor,
        deterministic_action=result.deterministic_action,
        reward_tail_mean=float(np.mean(result.reward_trace[-20:])),
        alpha_final=alpha_final,
        actor_loss_tail_mean=actor_loss_tail_mean,
        critic_loss_tail_mean=critic_loss_tail_mean,
        env_config=env_config,
    )
    validation_report, validation_output = evaluate_hmm_sac_sizer(
        frame=validation_frame,
        feature_columns=feature_columns,
        actor=result.actor,
        deterministic_action=result.deterministic_action,
        reward_tail_mean=float(np.mean(result.reward_trace[-20:])),
        alpha_final=alpha_final,
        actor_loss_tail_mean=actor_loss_tail_mean,
        critic_loss_tail_mean=critic_loss_tail_mean,
        env_config=env_config,
    )
    saved_path: str | None = None
    if checkpoint_path is not None:
        save_sac_actor_checkpoint(result, checkpoint_path)
        saved_path = str(checkpoint_path)
    report = HMMSACValidationReport(
        train=train_report,
        validation=validation_report,
        checkpoint_path=saved_path,
    )
    return report, train_output, validation_output


def evaluate_hmm_sac_sizer(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    actor: GaussianActor,
    deterministic_action: float,
    reward_tail_mean: float,
    alpha_final: float,
    actor_loss_tail_mean: float,
    critic_loss_tail_mean: float,
    env_config: HMMSizingEnvConfig | None = None,
) -> tuple[HMMSACTrainingReport, pd.DataFrame]:
    eval_env = HMMSACPositionSizingEnv(frame, feature_columns, env_config)
    state = eval_env.reset()
    done = False
    actor.eval()
    while not done:
        state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action = actor.deterministic(state_tensor).squeeze(0).numpy().astype(np.float32)
        state, _, done = eval_env.step(action)

    output = frame.iloc[: len(eval_env.net_returns)].copy()
    output["sac_sized_position"] = eval_env.positions
    output["sac_net_return"] = eval_env.net_returns
    output["sac_equity"] = (1.0 + output["sac_net_return"]).cumprod()
    turnover = output["sac_sized_position"].diff().abs().fillna(output["sac_sized_position"].abs())
    multiplier = (
        output["sac_sized_position"].abs() / output["baseline_position"].abs().replace(0.0, np.nan)
    ).fillna(0.0)
    report = HMMSACTrainingReport(
        deterministic_action=deterministic_action,
        mean_multiplier=float(multiplier.mean()),
        metrics=compute_metrics(output["sac_net_return"], turnover),
        reward_tail_mean=reward_tail_mean,
        alpha_final=alpha_final,
        actor_loss_tail_mean=actor_loss_tail_mean,
        critic_loss_tail_mean=critic_loss_tail_mean,
    )
    return report, output
