from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Normal
from torch.nn import functional as F

LOG_STD_MIN: Final[float] = -5.0
LOG_STD_MAX: Final[float] = 2.0
EPS: Final[float] = 1e-6


@dataclass(frozen=True)
class QuadraticEnvConfig:
    target_action: float = 0.5
    max_steps: int = 1


class QuadraticActionEnv:
    """Tiny continuous-control env for neural SAC tests.

    State is a constant scalar. Reward is maximized when action is close to
    `target_action`. This is intentionally not a trading environment.
    """

    def __init__(self, config: QuadraticEnvConfig | None = None) -> None:
        self.config = config or QuadraticEnvConfig()
        self.steps = 0

    @property
    def state_dim(self) -> int:
        return 1

    @property
    def action_dim(self) -> int:
        return 1

    def reset(self) -> np.ndarray:
        self.steps = 0
        return np.array([1.0], dtype=np.float32)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool]:
        self.steps += 1
        value = float(np.clip(action[0], -1.0, 1.0))
        reward = -((value - self.config.target_action) ** 2)
        done = self.steps >= self.config.max_steps
        return np.array([1.0], dtype=np.float32), reward, done


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000, seed: int = 7) -> None:
        self.items: deque[tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]] = deque(
            maxlen=capacity
        )
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.items.append((state, action, reward, next_state, done))

    def __len__(self) -> int:
        return len(self.items)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if batch_size > len(self.items):
            raise ValueError("batch_size exceeds replay buffer length")
        indices = self.rng.choice(len(self.items), size=batch_size, replace=False)
        batch = [self.items[int(index)] for index in indices]
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32),
            torch.as_tensor(np.asarray(actions), dtype=torch.float32),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32).unsqueeze(-1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32),
            torch.as_tensor(np.asarray(dones), dtype=torch.float32).unsqueeze(-1),
        )


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class GaussianActor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.backbone = MLP(state_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state: Tensor) -> tuple[Tensor, Tensor]:
        features = self.backbone(state)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state: Tensor) -> tuple[Tensor, Tensor]:
        mean, log_std = self(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        raw_action = normal.rsample()
        action = torch.tanh(raw_action)
        log_prob = normal.log_prob(raw_action) - torch.log(1 - action.pow(2) + EPS)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def deterministic(self, state: Tensor) -> Tensor:
        mean, _ = self(state)
        return torch.tanh(mean)


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.q = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state: Tensor, action: Tensor) -> Tensor:
        return self.q(torch.cat([state, action], dim=-1))


@dataclass(frozen=True)
class TorchSACConfig:
    steps: int = 220
    warmup_steps: int = 32
    batch_size: int = 32
    gamma: float = 0.0
    tau: float = 0.05
    alpha: float = 0.05
    actor_lr: float = 3e-3
    critic_lr: float = 3e-3
    hidden_dim: int = 32
    seed: int = 123


@dataclass(frozen=True)
class TorchSACResult:
    deterministic_action: float
    reward_trace: list[float]
    actor_loss_trace: list[float]
    critic_loss_trace: list[float]

    def as_dict(self) -> dict[str, object]:
        return {
            "deterministic_action": self.deterministic_action,
            "reward_trace": self.reward_trace,
            "actor_loss_trace": self.actor_loss_trace,
            "critic_loss_trace": self.critic_loss_trace,
        }


def train_torch_sac(
    env: QuadraticActionEnv,
    config: TorchSACConfig | None = None,
) -> TorchSACResult:
    cfg = config or TorchSACConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    actor = GaussianActor(env.state_dim, env.action_dim, cfg.hidden_dim)
    q1 = Critic(env.state_dim, env.action_dim, cfg.hidden_dim)
    q2 = Critic(env.state_dim, env.action_dim, cfg.hidden_dim)
    target_q1 = Critic(env.state_dim, env.action_dim, cfg.hidden_dim)
    target_q2 = Critic(env.state_dim, env.action_dim, cfg.hidden_dim)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=cfg.critic_lr)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=cfg.critic_lr)
    buffer = ReplayBuffer(seed=cfg.seed)

    reward_trace: list[float] = []
    actor_loss_trace: list[float] = []
    critic_loss_trace: list[float] = []
    state = env.reset()

    for step in range(cfg.steps):
        if step < cfg.warmup_steps:
            action = rng.uniform(-1.0, 1.0, size=(env.action_dim,)).astype(np.float32)
        else:
            with torch.no_grad():
                action_tensor, _ = actor.sample(torch.as_tensor(state).float().unsqueeze(0))
            action = action_tensor.squeeze(0).numpy().astype(np.float32)

        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        reward_trace.append(reward)
        state = env.reset() if done else next_state

        if len(buffer) < cfg.batch_size:
            continue

        states, actions, rewards, next_states, dones = buffer.sample(cfg.batch_size)
        critic_loss = _update_critics(
            actor,
            q1,
            q2,
            target_q1,
            target_q2,
            q1_opt,
            q2_opt,
            states,
            actions,
            rewards,
            next_states,
            dones,
            cfg,
        )
        actor_loss = _update_actor(actor, q1, q2, actor_opt, states, cfg.alpha)
        _soft_update(target_q1, q1, cfg.tau)
        _soft_update(target_q2, q2, cfg.tau)
        critic_loss_trace.append(critic_loss)
        actor_loss_trace.append(actor_loss)

    eval_state = torch.as_tensor(env.reset(), dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        deterministic_action = actor.deterministic(eval_state).item()
    return TorchSACResult(
        deterministic_action=deterministic_action,
        reward_trace=reward_trace,
        actor_loss_trace=actor_loss_trace,
        critic_loss_trace=critic_loss_trace,
    )


def _update_critics(
    actor: GaussianActor,
    q1: Critic,
    q2: Critic,
    target_q1: Critic,
    target_q2: Critic,
    q1_opt: torch.optim.Optimizer,
    q2_opt: torch.optim.Optimizer,
    states: Tensor,
    actions: Tensor,
    rewards: Tensor,
    next_states: Tensor,
    dones: Tensor,
    cfg: TorchSACConfig,
) -> float:
    with torch.no_grad():
        next_actions, next_log_prob = actor.sample(next_states)
        target_q = torch.min(target_q1(next_states, next_actions), target_q2(next_states, next_actions))
        target = rewards + cfg.gamma * (1.0 - dones) * (target_q - cfg.alpha * next_log_prob)

    q1_loss = F.mse_loss(q1(states, actions), target)
    q2_loss = F.mse_loss(q2(states, actions), target)
    q1_opt.zero_grad()
    q1_loss.backward()
    q1_opt.step()
    q2_opt.zero_grad()
    q2_loss.backward()
    q2_opt.step()
    return float((q1_loss + q2_loss).detach().item())


def _update_actor(
    actor: GaussianActor,
    q1: Critic,
    q2: Critic,
    actor_opt: torch.optim.Optimizer,
    states: Tensor,
    alpha: float,
) -> float:
    sampled_actions, log_prob = actor.sample(states)
    min_q = torch.min(q1(states, sampled_actions), q2(states, sampled_actions))
    actor_loss = (alpha * log_prob - min_q).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    return float(actor_loss.detach().item())


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)
