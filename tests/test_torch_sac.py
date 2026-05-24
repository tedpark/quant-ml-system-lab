import numpy as np
import torch

from quant_ml_lab.torch_sac import (
    GaussianActor,
    QuadraticActionEnv,
    QuadraticEnvConfig,
    ReplayBuffer,
    TorchSACConfig,
    train_torch_sac,
)


def test_actor_sample_shapes_and_bounds():
    actor = GaussianActor(state_dim=1, action_dim=1)
    state = torch.ones(4, 1)

    action, log_prob = actor.sample(state)

    assert action.shape == (4, 1)
    assert log_prob.shape == (4, 1)
    assert torch.all(action <= 1.0)
    assert torch.all(action >= -1.0)


def test_replay_buffer_sample_shapes():
    buffer = ReplayBuffer(seed=1)
    for _ in range(5):
        buffer.push(
            np.array([1.0], dtype=np.float32),
            np.array([0.2], dtype=np.float32),
            1.0,
            np.array([1.0], dtype=np.float32),
            True,
        )

    states, actions, rewards, next_states, dones = buffer.sample(4)

    assert states.shape == (4, 1)
    assert actions.shape == (4, 1)
    assert rewards.shape == (4, 1)
    assert next_states.shape == (4, 1)
    assert dones.shape == (4, 1)


def test_torch_sac_trains_toward_target_action():
    env = QuadraticActionEnv(QuadraticEnvConfig(target_action=0.5))
    result = train_torch_sac(env, TorchSACConfig(steps=180, warmup_steps=24, seed=11))

    assert abs(result.deterministic_action - 0.5) < 0.6
    assert len(result.reward_trace) == 180
    assert result.actor_loss_trace
    assert result.critic_loss_trace
    assert result.alpha_trace
    assert result.alpha_loss_trace
    assert all(np.isfinite(value) for value in result.alpha_trace[-5:])
    assert min(result.alpha_trace) > 0.0


def test_torch_sac_can_use_fixed_alpha():
    env = QuadraticActionEnv(QuadraticEnvConfig(target_action=0.2))
    result = train_torch_sac(
        env,
        TorchSACConfig(
            steps=80,
            warmup_steps=16,
            batch_size=16,
            automatic_entropy_tuning=False,
            alpha=0.03,
            seed=21,
        ),
    )

    assert result.alpha_trace
    assert set(round(value, 6) for value in result.alpha_trace) == {0.03}
    assert set(result.alpha_loss_trace) == {0.0}
