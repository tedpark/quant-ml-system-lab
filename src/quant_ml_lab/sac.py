from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ContinuousBanditConfig:
    target_action: float = 0.65
    action_low: float = -1.0
    action_high: float = 1.0
    noise_scale: float = 0.01


class ContinuousBandit:
    """One-state continuous-control toy environment for SAC concept study.

    Reward is highest when the action is close to `target_action`.
    This is not a trading environment.
    """

    def __init__(self, config: ContinuousBanditConfig | None = None, seed: int = 7) -> None:
        self.config = config or ContinuousBanditConfig()
        self.rng = np.random.default_rng(seed)

    def reward(self, action: float) -> float:
        clipped = float(np.clip(action, self.config.action_low, self.config.action_high))
        noise = float(self.rng.normal(0.0, self.config.noise_scale))
        return -((clipped - self.config.target_action) ** 2) + noise


@dataclass(frozen=True)
class SoftActorCriticBanditConfig:
    iterations: int = 300
    candidate_actions: int = 41
    samples_per_iteration: int = 16
    actor_lr: float = 0.08
    critic_lr: float = 0.12
    alpha: float = 0.08
    gamma: float = 0.0
    seed: int = 42


@dataclass(frozen=True)
class SoftActorCriticBanditResult:
    actions: np.ndarray
    policy: np.ndarray
    q1: np.ndarray
    q2: np.ndarray
    expected_action: float
    greedy_action: float
    entropy: float
    mean_reward_trace: list[float]

    def as_dict(self) -> dict[str, object]:
        return {
            "expected_action": self.expected_action,
            "greedy_action": self.greedy_action,
            "entropy": self.entropy,
            "mean_reward_trace": self.mean_reward_trace,
            "candidate_actions": self.actions.tolist(),
            "policy": self.policy.tolist(),
        }


def train_soft_actor_critic_bandit(
    env: ContinuousBandit,
    config: SoftActorCriticBanditConfig | None = None,
) -> SoftActorCriticBanditResult:
    """Train a small discrete-action SAC analogue over continuous action candidates.

    This is a learning bridge: it keeps SAC concepts visible without pulling in
    a deep learning dependency. Full neural SAC can be added later.
    """
    cfg = config or SoftActorCriticBanditConfig()
    rng = np.random.default_rng(cfg.seed)
    actions = np.linspace(env.config.action_low, env.config.action_high, cfg.candidate_actions)
    logits = np.zeros(cfg.candidate_actions, dtype=float)
    q1 = np.zeros(cfg.candidate_actions, dtype=float)
    q2 = np.zeros(cfg.candidate_actions, dtype=float)
    mean_reward_trace: list[float] = []

    for _ in range(cfg.iterations):
        policy = softmax(logits)
        sampled_indices = rng.choice(
            np.arange(cfg.candidate_actions),
            size=cfg.samples_per_iteration,
            replace=True,
            p=policy,
        )
        rewards: list[float] = []
        for idx in sampled_indices:
            reward = env.reward(float(actions[idx]))
            rewards.append(reward)
            target = reward + cfg.gamma * soft_value(q1, q2, policy, cfg.alpha)
            q1[idx] += cfg.critic_lr * (target - q1[idx])
            q2[idx] += cfg.critic_lr * (target - q2[idx])

        soft_q = np.minimum(q1, q2)
        actor_objective = soft_q - cfg.alpha * safe_log(policy)
        logits += cfg.actor_lr * (actor_objective - actor_objective.mean())
        mean_reward_trace.append(float(np.mean(rewards)))

    policy = softmax(logits)
    soft_q = np.minimum(q1, q2)
    return SoftActorCriticBanditResult(
        actions=actions,
        policy=policy,
        q1=q1,
        q2=q2,
        expected_action=float(np.sum(actions * policy)),
        greedy_action=float(actions[int(np.argmax(soft_q))]),
        entropy=entropy(policy),
        mean_reward_trace=mean_reward_trace,
    )


def soft_value(q1: np.ndarray, q2: np.ndarray, policy: np.ndarray, alpha: float) -> float:
    soft_q = np.minimum(q1, q2)
    return float(np.sum(policy * (soft_q - alpha * safe_log(policy))))


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def safe_log(values: np.ndarray, epsilon: float = 1e-12) -> np.ndarray:
    return np.log(np.clip(values, epsilon, None))


def entropy(policy: np.ndarray) -> float:
    return float(-np.sum(policy * safe_log(policy)))
