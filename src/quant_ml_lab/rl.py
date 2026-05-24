from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Action = int
State = int


@dataclass(frozen=True)
class GridWorldConfig:
    rows: int = 4
    cols: int = 4
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (3, 3)
    step_reward: float = -0.01
    goal_reward: float = 1.0
    max_steps: int = 64


class GridWorld:
    """Small deterministic environment for learning RL basics.

    This is intentionally unrelated to production trading. It exists to keep
    RL study reproducible before adding more complex financial RL examples.
    """

    ACTIONS: tuple[tuple[int, int], ...] = (
        (-1, 0),  # up
        (0, 1),  # right
        (1, 0),  # down
        (0, -1),  # left
    )

    def __init__(self, config: GridWorldConfig | None = None) -> None:
        self.config = config or GridWorldConfig()
        self.position = self.config.start
        self.steps = 0

    @property
    def n_states(self) -> int:
        return self.config.rows * self.config.cols

    @property
    def n_actions(self) -> int:
        return len(self.ACTIONS)

    def reset(self) -> State:
        self.position = self.config.start
        self.steps = 0
        return self.state_index(self.position)

    def step(self, action: Action) -> tuple[State, float, bool]:
        if not 0 <= action < self.n_actions:
            raise ValueError("invalid action")
        self.steps += 1
        dr, dc = self.ACTIONS[action]
        row = min(max(self.position[0] + dr, 0), self.config.rows - 1)
        col = min(max(self.position[1] + dc, 0), self.config.cols - 1)
        self.position = (row, col)
        done = self.position == self.config.goal or self.steps >= self.config.max_steps
        reward = self.config.goal_reward if self.position == self.config.goal else self.config.step_reward
        return self.state_index(self.position), reward, done

    def state_index(self, position: tuple[int, int]) -> State:
        row, col = position
        return row * self.config.cols + col

    def position_for_state(self, state: State) -> tuple[int, int]:
        return divmod(state, self.config.cols)


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 300
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon_start: float = 0.8
    epsilon_end: float = 0.05
    seed: int = 42


@dataclass(frozen=True)
class QLearningResult:
    q_table: np.ndarray
    episode_returns: list[float]
    policy: list[int]

    def as_dict(self) -> dict[str, object]:
        return {
            "episode_returns": self.episode_returns,
            "policy": self.policy,
            "q_table_shape": list(self.q_table.shape),
        }


def train_q_learning(
    env: GridWorld,
    config: QLearningConfig | None = None,
) -> QLearningResult:
    cfg = config or QLearningConfig()
    rng = np.random.default_rng(cfg.seed)
    q_table = np.zeros((env.n_states, env.n_actions), dtype=float)
    episode_returns: list[float] = []

    for episode in range(cfg.episodes):
        epsilon = _linear_decay(
            cfg.epsilon_start,
            cfg.epsilon_end,
            episode,
            max(cfg.episodes - 1, 1),
        )
        state = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(env.n_actions))
            else:
                action = int(np.argmax(q_table[state]))
            next_state, reward, done = env.step(action)
            td_target = reward + cfg.gamma * float(np.max(q_table[next_state])) * (not done)
            td_error = td_target - q_table[state, action]
            q_table[state, action] += cfg.alpha * td_error
            total_reward += reward
            state = next_state

        episode_returns.append(total_reward)

    policy = [int(np.argmax(q_table[state])) for state in range(env.n_states)]
    return QLearningResult(q_table=q_table, episode_returns=episode_returns, policy=policy)


def evaluate_policy(env: GridWorld, policy: list[int]) -> tuple[float, list[tuple[int, int]]]:
    state = env.reset()
    done = False
    total_reward = 0.0
    path = [env.position_for_state(state)]
    while not done:
        action = policy[state]
        state, reward, done = env.step(action)
        total_reward += reward
        path.append(env.position_for_state(state))
    return total_reward, path


def _linear_decay(start: float, end: float, step: int, total_steps: int) -> float:
    fraction = min(max(step / total_steps, 0.0), 1.0)
    return start + fraction * (end - start)
