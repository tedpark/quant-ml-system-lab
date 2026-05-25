from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from quant_ml_lab.strategy_selector import (
    StrategyCandidate,
    StrategyName,
    StrategySelectionReport,
    StrategySelectorConfig,
    apply_strategy_selection,
    build_strategy_candidates,
    run_strategy_selector,
)

DEFAULT_STRATEGY_ACTIONS: tuple[StrategyName, ...] = (
    "no_trade",
    "mean_reversion_full",
    "mean_reversion_low_risk",
    "volatility_defensive",
    "cvar_defensive",
)


@dataclass(frozen=True)
class DQNSelectorConfig:
    episodes: int = 12
    batch_size: int = 32
    replay_capacity: int = 8_000
    gamma: float = 0.95
    lr: float = 1e-3
    hidden_dim: int = 64
    epsilon_start: float = 0.65
    epsilon_end: float = 0.05
    target_update_interval: int = 64
    turnover_penalty: float = 0.0005
    high_vol_penalty: float = 0.001
    drawdown_penalty: float = 0.002
    max_action_concentration: float = 0.85
    max_q_abs_mean: float = 2.0
    seed: int = 17
    device: str = "cpu"


@dataclass(frozen=True)
class DQNSelectorTrainingReport:
    train_metrics: dict[str, float | int]
    validation_metrics: dict[str, float | int]
    rule_based_validation_metrics: dict[str, float | int]
    random_validation_metrics: dict[str, float | int]
    candidate_validation_metrics: dict[str, dict[str, float | int]]
    validation_selection_counts: dict[str, int]
    rule_based_selection_counts: dict[str, int]
    random_selection_counts: dict[str, int]
    train_action_counts: dict[str, int]
    mean_train_reward_tail: float
    final_epsilon: float
    reward_trace_head: list[float]
    reward_trace_tail: list[float]
    loss_tail_mean: float
    loss_trace_head: list[float]
    loss_trace_tail: list[float]
    q_value_tail_mean: float
    q_value_tail_abs_mean: float
    q_value_trace_tail: list[float]
    validation_action_concentration: float
    diagnostics: dict[str, bool | float]
    checkpoint_path: str | None
    disclosure: str = (
        "Public DQN strategy-selector scaffold. Synthetic data only. Not a live strategy."
    )

    def as_dict(self) -> dict[str, object]:
        return {
            "train_metrics": self.train_metrics,
            "validation_metrics": self.validation_metrics,
            "rule_based_validation_metrics": self.rule_based_validation_metrics,
            "random_validation_metrics": self.random_validation_metrics,
            "candidate_validation_metrics": self.candidate_validation_metrics,
            "validation_selection_counts": self.validation_selection_counts,
            "rule_based_selection_counts": self.rule_based_selection_counts,
            "random_selection_counts": self.random_selection_counts,
            "train_action_counts": self.train_action_counts,
            "mean_train_reward_tail": self.mean_train_reward_tail,
            "final_epsilon": self.final_epsilon,
            "reward_trace_head": self.reward_trace_head,
            "reward_trace_tail": self.reward_trace_tail,
            "loss_tail_mean": self.loss_tail_mean,
            "loss_trace_head": self.loss_trace_head,
            "loss_trace_tail": self.loss_trace_tail,
            "q_value_tail_mean": self.q_value_tail_mean,
            "q_value_tail_abs_mean": self.q_value_tail_abs_mean,
            "q_value_trace_tail": self.q_value_trace_tail,
            "validation_action_concentration": self.validation_action_concentration,
            "diagnostics": self.diagnostics,
            "checkpoint_path": self.checkpoint_path,
            "disclosure": self.disclosure,
        }


class StrategySelectorEnv:
    def __init__(
        self,
        frame: pd.DataFrame,
        feature_columns: tuple[str, ...],
        candidates: dict[StrategyName, StrategyCandidate],
        strategy_actions: tuple[StrategyName, ...] = DEFAULT_STRATEGY_ACTIONS,
        selector_config: StrategySelectorConfig | None = None,
        dqn_config: DQNSelectorConfig | None = None,
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
        self.strategy_actions = strategy_actions
        self.selector_config = selector_config or StrategySelectorConfig()
        self.dqn_config = dqn_config or DQNSelectorConfig()
        self.cursor = 0
        self.prev_position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0

    @property
    def state_dim(self) -> int:
        return len(self.feature_columns)

    @property
    def action_dim(self) -> int:
        return len(self.strategy_actions)

    def reset(self) -> np.ndarray:
        self.cursor = 0
        self.prev_position = 0.0
        self.equity = 1.0
        self.peak_equity = 1.0
        return self._state()

    def step(self, action_index: int) -> tuple[np.ndarray, float, bool]:
        strategy_name = self.strategy_actions[int(action_index)]
        row = self.frame.iloc[self.cursor]
        candidate = self.candidates[strategy_name]
        position = float(candidate.position.iloc[self.cursor])
        spread_return_next = float(row["spread_return_next"])
        gross_return = position * -spread_return_next
        turnover = abs(position - self.prev_position)
        cost = turnover * (self.selector_config.transaction_cost_bps / 10_000.0)
        net_return = gross_return - cost
        next_equity = self.equity * (1.0 + net_return)
        next_peak = max(self.peak_equity, next_equity)
        drawdown = max(0.0, 1.0 - next_equity / next_peak)
        high_vol_prob = float(row["high_vol_prob"])
        reward = (
            net_return
            - turnover * self.dqn_config.turnover_penalty
            - high_vol_prob * abs(position) * self.dqn_config.high_vol_penalty
            - drawdown * self.dqn_config.drawdown_penalty
        )

        self.prev_position = position
        self.equity = next_equity
        self.peak_equity = next_peak
        self.cursor += 1
        done = self.cursor >= len(self.frame)
        next_state = self._state() if not done else np.zeros(self.state_dim, dtype=np.float32)
        return next_state, float(reward), done

    def _state(self) -> np.ndarray:
        return self.frame.loc[self.cursor, list(self.feature_columns)].to_numpy(dtype=np.float32)


class DQNReplayBuffer:
    def __init__(self, capacity: int, seed: int) -> None:
        self.items: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.items.append((state, action, reward, next_state, done))

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, batch_size: int, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if batch_size > len(self.items):
            raise ValueError("batch_size exceeds replay buffer length")
        indices = self.rng.choice(len(self.items), size=batch_size, replace=False)
        batch = [self.items[int(index)] for index in indices]
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(actions), dtype=torch.long, device=device).unsqueeze(-1),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device).unsqueeze(-1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=device),
            torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=device).unsqueeze(-1),
        )


class DQNSelectorNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, state: Tensor) -> Tensor:
        return self.net(state)


def train_validate_strategy_selector_dqn(
    train_frame: pd.DataFrame,
    validation_frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    selector_config: StrategySelectorConfig | None = None,
    dqn_config: DQNSelectorConfig | None = None,
    checkpoint_path: str | Path | None = None,
) -> tuple[DQNSelectorTrainingReport, pd.DataFrame, pd.DataFrame]:
    selector_cfg = selector_config or StrategySelectorConfig()
    dqn_cfg = dqn_config or DQNSelectorConfig()
    device = _resolve_device(dqn_cfg.device)
    torch.manual_seed(dqn_cfg.seed)
    rng = np.random.default_rng(dqn_cfg.seed)
    train_candidates = build_strategy_candidates(train_frame, selector_cfg)
    validation_candidates = build_strategy_candidates(validation_frame, selector_cfg)
    env = StrategySelectorEnv(
        train_frame,
        feature_columns,
        train_candidates,
        selector_config=selector_cfg,
        dqn_config=dqn_cfg,
    )
    q_net = DQNSelectorNetwork(env.state_dim, env.action_dim, dqn_cfg.hidden_dim).to(device)
    target_net = DQNSelectorNetwork(env.state_dim, env.action_dim, dqn_cfg.hidden_dim).to(device)
    target_net.load_state_dict(q_net.state_dict())
    optimizer = torch.optim.Adam(q_net.parameters(), lr=dqn_cfg.lr)
    buffer = DQNReplayBuffer(dqn_cfg.replay_capacity, dqn_cfg.seed)
    reward_trace: list[float] = []
    loss_trace: list[float] = []
    q_value_trace: list[float] = []
    train_action_counts = {name: 0 for name in DEFAULT_STRATEGY_ACTIONS}
    update_count = 0
    total_steps = max(1, dqn_cfg.episodes * len(train_frame))

    state = env.reset()
    for step in range(total_steps):
        epsilon = _epsilon_at_step(step, total_steps, dqn_cfg)
        if rng.random() < epsilon:
            action = int(rng.integers(0, env.action_dim))
        else:
            action = _greedy_action(q_net, state, device)
        train_action_counts[DEFAULT_STRATEGY_ACTIONS[action]] += 1
        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        reward_trace.append(reward)
        q_value_trace.append(_state_q_mean(q_net, state, device))
        state = env.reset() if done else next_state

        if len(buffer) >= dqn_cfg.batch_size:
            loss = _update_dqn(q_net, target_net, optimizer, buffer, dqn_cfg, device)
            loss_trace.append(loss)
            update_count += 1
            if update_count % dqn_cfg.target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

    q_net.cpu()
    train_output, train_report = evaluate_strategy_selector_dqn(
        train_frame,
        feature_columns,
        train_candidates,
        q_net,
        selector_cfg,
    )
    validation_output, validation_report = evaluate_strategy_selector_dqn(
        validation_frame,
        feature_columns,
        validation_candidates,
        q_net,
        selector_cfg,
    )
    _, rule_validation_report = run_strategy_selector(validation_frame, selector_cfg)
    _, random_validation_report = evaluate_random_strategy_selector(
        validation_frame,
        validation_candidates,
        selector_cfg,
        seed=dqn_cfg.seed,
    )
    saved_path: str | None = None
    if checkpoint_path is not None:
        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": q_net.state_dict(),
                "feature_columns": feature_columns,
                "strategy_actions": DEFAULT_STRATEGY_ACTIONS,
                "config": dqn_cfg,
            },
            path,
        )
        saved_path = str(path)

    validation_action_concentration = _action_concentration(validation_report.selected_counts)
    q_value_tail_abs_mean = float(np.mean(np.abs(q_value_trace[-50:]))) if q_value_trace else 0.0
    loss_tail_mean = float(np.mean(loss_trace[-50:])) if loss_trace else 0.0
    diagnostics = {
        "loss_is_finite": bool(np.isfinite(loss_tail_mean)),
        "q_values_are_bounded": q_value_tail_abs_mean <= dqn_cfg.max_q_abs_mean,
        "validation_not_single_action": (
            validation_action_concentration <= dqn_cfg.max_action_concentration
        ),
        "beats_random_sharpe": (
            validation_report.selected_metrics.sharpe
            >= random_validation_report.selected_metrics.sharpe
        ),
        "beats_rule_based_sharpe": (
            validation_report.selected_metrics.sharpe
            >= rule_validation_report.selected_metrics.sharpe
        ),
        "validation_action_concentration": validation_action_concentration,
        "q_value_tail_abs_mean": q_value_tail_abs_mean,
    }
    report = DQNSelectorTrainingReport(
        train_metrics=train_report.selected_metrics.as_dict(),
        validation_metrics=validation_report.selected_metrics.as_dict(),
        rule_based_validation_metrics=rule_validation_report.selected_metrics.as_dict(),
        random_validation_metrics=random_validation_report.selected_metrics.as_dict(),
        candidate_validation_metrics=validation_report.candidate_metrics,
        validation_selection_counts=validation_report.selected_counts,
        rule_based_selection_counts=rule_validation_report.selected_counts,
        random_selection_counts=random_validation_report.selected_counts,
        train_action_counts={str(key): int(value) for key, value in train_action_counts.items()},
        mean_train_reward_tail=float(np.mean(reward_trace[-50:])) if reward_trace else 0.0,
        final_epsilon=_epsilon_at_step(total_steps - 1, total_steps, dqn_cfg),
        reward_trace_head=[float(value) for value in reward_trace[:10]],
        reward_trace_tail=[float(value) for value in reward_trace[-10:]],
        loss_tail_mean=loss_tail_mean,
        loss_trace_head=[float(value) for value in loss_trace[:10]],
        loss_trace_tail=[float(value) for value in loss_trace[-10:]],
        q_value_tail_mean=float(np.mean(q_value_trace[-50:])) if q_value_trace else 0.0,
        q_value_tail_abs_mean=q_value_tail_abs_mean,
        q_value_trace_tail=[float(value) for value in q_value_trace[-10:]],
        validation_action_concentration=validation_action_concentration,
        diagnostics=diagnostics,
        checkpoint_path=saved_path,
    )
    return report, train_output, validation_output


def evaluate_strategy_selector_dqn(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    candidates: dict[StrategyName, StrategyCandidate],
    q_net: DQNSelectorNetwork,
    selector_config: StrategySelectorConfig | None = None,
) -> tuple[pd.DataFrame, StrategySelectionReport]:
    selector_cfg = selector_config or StrategySelectorConfig()
    device = next(q_net.parameters()).device
    q_net.eval()
    selected: list[StrategyName] = []
    for _, row in frame.iterrows():
        state = row.loc[list(feature_columns)].to_numpy(dtype=np.float32)
        action = _greedy_action(q_net, state, device)
        selected.append(DEFAULT_STRATEGY_ACTIONS[action])
    selected_series = pd.Series(selected, index=frame.index, dtype=object)
    return apply_strategy_selection(frame, candidates, selected_series, selector_cfg)


def evaluate_random_strategy_selector(
    frame: pd.DataFrame,
    candidates: dict[StrategyName, StrategyCandidate],
    selector_config: StrategySelectorConfig | None = None,
    seed: int = 17,
) -> tuple[pd.DataFrame, StrategySelectionReport]:
    selector_cfg = selector_config or StrategySelectorConfig()
    rng = np.random.default_rng(seed)
    actions = rng.integers(0, len(DEFAULT_STRATEGY_ACTIONS), size=len(frame))
    selected = pd.Series(
        [DEFAULT_STRATEGY_ACTIONS[int(action)] for action in actions],
        index=frame.index,
        dtype=object,
    )
    return apply_strategy_selection(frame, candidates, selected, selector_cfg)


def _update_dqn(
    q_net: DQNSelectorNetwork,
    target_net: DQNSelectorNetwork,
    optimizer: torch.optim.Optimizer,
    buffer: DQNReplayBuffer,
    config: DQNSelectorConfig,
    device: torch.device,
) -> float:
    states, actions, rewards, next_states, dones = buffer.sample(config.batch_size, device)
    q_values = q_net(states).gather(1, actions)
    with torch.no_grad():
        next_actions = q_net(next_states).argmax(dim=1, keepdim=True)
        next_q = target_net(next_states).gather(1, next_actions)
        target = rewards + (1.0 - dones) * config.gamma * next_q
    loss = F.smooth_l1_loss(q_values, target)
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(q_net.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.detach().cpu().item())


def _greedy_action(q_net: DQNSelectorNetwork, state: np.ndarray, device: torch.device) -> int:
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return int(q_net(state_tensor).argmax(dim=1).item())


def _state_q_mean(q_net: DQNSelectorNetwork, state: np.ndarray, device: torch.device) -> float:
    with torch.no_grad():
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return float(q_net(state_tensor).mean().detach().cpu().item())


def _action_concentration(counts: dict[str, int]) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return max(counts.values()) / total


def _epsilon_at_step(step: int, total_steps: int, config: DQNSelectorConfig) -> float:
    progress = min(1.0, max(0.0, step / max(1, total_steps - 1)))
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
