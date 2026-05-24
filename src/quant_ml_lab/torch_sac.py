from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
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

    def sample(
        self,
        batch_size: int,
        device: torch.device | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        if batch_size > len(self.items):
            raise ValueError("batch_size exceeds replay buffer length")
        indices = self.rng.choice(len(self.items), size=batch_size, replace=False)
        batch = [self.items[int(index)] for index in indices]
        states, actions, rewards, next_states, dones = zip(*batch, strict=True)
        target_device = device or torch.device("cpu")
        return (
            torch.as_tensor(np.asarray(states), dtype=torch.float32, device=target_device),
            torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=target_device),
            torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=target_device).unsqueeze(-1),
            torch.as_tensor(np.asarray(next_states), dtype=torch.float32, device=target_device),
            torch.as_tensor(np.asarray(dones), dtype=torch.float32, device=target_device).unsqueeze(-1),
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
    alpha_init: float = 0.05
    alpha_lr: float = 3e-3
    alpha_floor: float = 1e-4
    target_entropy: float = -1.0
    automatic_entropy_tuning: bool = True
    actor_lr: float = 3e-3
    critic_lr: float = 3e-3
    hidden_dim: int = 32
    seed: int = 123
    replay_capacity: int = 10_000
    device: str = "cpu"


@dataclass(frozen=True)
class TorchSACResult:
    deterministic_action: float
    reward_trace: list[float]
    actor_loss_trace: list[float]
    critic_loss_trace: list[float]
    alpha_trace: list[float]
    alpha_loss_trace: list[float]
    actor: GaussianActor
    config: TorchSACConfig
    state_dim: int
    action_dim: int

    def as_dict(self) -> dict[str, object]:
        return {
            "deterministic_action": self.deterministic_action,
            "reward_trace": self.reward_trace,
            "actor_loss_trace": self.actor_loss_trace,
            "critic_loss_trace": self.critic_loss_trace,
            "alpha_trace": self.alpha_trace,
            "alpha_loss_trace": self.alpha_loss_trace,
        }


def train_torch_sac(
    env: QuadraticActionEnv,
    config: TorchSACConfig | None = None,
) -> TorchSACResult:
    cfg = config or TorchSACConfig()
    _validate_config(cfg)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    device = _resolve_device(cfg.device)

    actor = GaussianActor(env.state_dim, env.action_dim, cfg.hidden_dim).to(device)
    q1 = Critic(env.state_dim, env.action_dim, cfg.hidden_dim).to(device)
    q2 = Critic(env.state_dim, env.action_dim, cfg.hidden_dim).to(device)
    target_q1 = Critic(env.state_dim, env.action_dim, cfg.hidden_dim).to(device)
    target_q2 = Critic(env.state_dim, env.action_dim, cfg.hidden_dim).to(device)
    target_q1.load_state_dict(q1.state_dict())
    target_q2.load_state_dict(q2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr)
    q1_opt = torch.optim.Adam(q1.parameters(), lr=cfg.critic_lr)
    q2_opt = torch.optim.Adam(q2.parameters(), lr=cfg.critic_lr)
    log_alpha = torch.tensor(
        np.log(cfg.alpha_init),
        dtype=torch.float32,
        device=device,
        requires_grad=cfg.automatic_entropy_tuning,
    )
    alpha_opt = torch.optim.Adam([log_alpha], lr=cfg.alpha_lr)
    buffer = ReplayBuffer(capacity=cfg.replay_capacity, seed=cfg.seed)

    reward_trace: list[float] = []
    actor_loss_trace: list[float] = []
    critic_loss_trace: list[float] = []
    alpha_trace: list[float] = []
    alpha_loss_trace: list[float] = []
    state = env.reset()

    for step in range(cfg.steps):
        if step < cfg.warmup_steps:
            action = rng.uniform(-1.0, 1.0, size=(env.action_dim,)).astype(np.float32)
        else:
            with torch.no_grad():
                action_tensor, _ = actor.sample(
                    torch.as_tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                )
            action = action_tensor.squeeze(0).cpu().numpy().astype(np.float32)

        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        reward_trace.append(reward)
        state = env.reset() if done else next_state

        if len(buffer) < cfg.batch_size:
            continue

        states, actions, rewards, next_states, dones = buffer.sample(cfg.batch_size, device)
        alpha = _current_alpha(log_alpha, cfg)
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
            alpha,
        )
        actor_loss, log_prob = _update_actor(actor, q1, q2, actor_opt, states, alpha)
        alpha_loss = _update_alpha(alpha_opt, log_alpha, log_prob, cfg)
        _soft_update(target_q1, q1, cfg.tau)
        _soft_update(target_q2, q2, cfg.tau)
        critic_loss_trace.append(critic_loss)
        actor_loss_trace.append(actor_loss)
        alpha_trace.append(float(_current_alpha(log_alpha, cfg).detach().item()))
        alpha_loss_trace.append(alpha_loss)

    eval_state = torch.as_tensor(env.reset(), dtype=torch.float32, device=device).unsqueeze(0)
    actor.eval()
    with torch.no_grad():
        deterministic_action = actor.deterministic(eval_state).item()
    actor.cpu()
    return TorchSACResult(
        deterministic_action=deterministic_action,
        reward_trace=reward_trace,
        actor_loss_trace=actor_loss_trace,
        critic_loss_trace=critic_loss_trace,
        alpha_trace=alpha_trace,
        alpha_loss_trace=alpha_loss_trace,
        actor=actor,
        config=cfg,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
    )


def _validate_config(cfg: TorchSACConfig) -> None:
    if cfg.steps <= 0:
        raise ValueError("steps must be positive")
    if cfg.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if cfg.warmup_steps < 0:
        raise ValueError("warmup_steps must be non-negative")
    if not 0.0 <= cfg.gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1]")
    if not 0.0 < cfg.tau <= 1.0:
        raise ValueError("tau must be in (0, 1]")
    if cfg.alpha <= 0.0 or cfg.alpha_init <= 0.0 or cfg.alpha_floor <= 0.0:
        raise ValueError("alpha values must be positive")
    if cfg.replay_capacity < cfg.batch_size:
        raise ValueError("replay_capacity must be at least batch_size")


def _resolve_device(device: str) -> torch.device:
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def _current_alpha(log_alpha: Tensor, cfg: TorchSACConfig) -> Tensor:
    if cfg.automatic_entropy_tuning:
        return log_alpha.exp().clamp_min(cfg.alpha_floor).detach()
    return torch.tensor(cfg.alpha, dtype=torch.float32, device=log_alpha.device)


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
    alpha: Tensor,
) -> float:
    with torch.no_grad():
        next_actions, next_log_prob = actor.sample(next_states)
        target_q = torch.min(target_q1(next_states, next_actions), target_q2(next_states, next_actions))
        target = rewards + cfg.gamma * (1.0 - dones) * (target_q - alpha * next_log_prob)

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
    alpha: Tensor,
) -> tuple[float, Tensor]:
    sampled_actions, log_prob = actor.sample(states)
    min_q = torch.min(q1(states, sampled_actions), q2(states, sampled_actions))
    actor_loss = (alpha * log_prob - min_q).mean()
    actor_opt.zero_grad()
    actor_loss.backward()
    actor_opt.step()
    return float(actor_loss.detach().item()), log_prob.detach()


def _update_alpha(
    alpha_opt: torch.optim.Optimizer,
    log_alpha: Tensor,
    log_prob: Tensor,
    cfg: TorchSACConfig,
) -> float:
    if not cfg.automatic_entropy_tuning:
        return 0.0
    alpha_loss = -(log_alpha * (log_prob + cfg.target_entropy).detach()).mean()
    alpha_opt.zero_grad()
    alpha_loss.backward()
    alpha_opt.step()
    with torch.no_grad():
        log_alpha.clamp_(min=float(np.log(cfg.alpha_floor)))
    return float(alpha_loss.detach().item())


def _soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, source_param in zip(target.parameters(), source.parameters(), strict=True):
            target_param.data.mul_(1.0 - tau).add_(source_param.data, alpha=tau)


def save_sac_actor_checkpoint(result: TorchSACResult, path: str | Path) -> None:
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "actor_state_dict": result.actor.state_dict(),
            "config": asdict(result.config),
            "state_dim": result.state_dim,
            "action_dim": result.action_dim,
            "hidden_dim": result.config.hidden_dim,
            "deterministic_action": result.deterministic_action,
        },
        checkpoint_path,
    )


def load_sac_actor_checkpoint(path: str | Path, map_location: str = "cpu") -> GaussianActor:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    actor = GaussianActor(
        state_dim=int(payload["state_dim"]),
        action_dim=int(payload["action_dim"]),
        hidden_dim=int(payload["hidden_dim"]),
    )
    actor.load_state_dict(payload["actor_state_dict"])
    actor.eval()
    return actor
