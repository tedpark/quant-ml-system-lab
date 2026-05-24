import numpy as np

from quant_ml_lab.sac import (
    ContinuousBandit,
    ContinuousBanditConfig,
    SoftActorCriticBanditConfig,
    entropy,
    soft_value,
    softmax,
    train_soft_actor_critic_bandit,
)


def test_softmax_returns_distribution():
    probs = softmax(np.array([1.0, 2.0, 3.0]))

    assert probs.shape == (3,)
    assert np.isclose(probs.sum(), 1.0)
    assert probs[-1] > probs[0]


def test_soft_value_uses_min_twin_critic_and_entropy():
    q1 = np.array([1.0, 2.0])
    q2 = np.array([0.5, 3.0])
    policy = np.array([0.5, 0.5])

    value = soft_value(q1, q2, policy, alpha=0.1)

    assert value > 1.2
    assert value < 1.4


def test_entropy_is_positive_for_stochastic_policy():
    policy = np.array([0.5, 0.5])

    assert entropy(policy) > 0.0


def test_sac_bandit_learns_near_target_action():
    env = ContinuousBandit(ContinuousBanditConfig(target_action=0.65, noise_scale=0.0))
    result = train_soft_actor_critic_bandit(
        env,
        SoftActorCriticBanditConfig(iterations=260, seed=123, alpha=0.05),
    )

    assert abs(result.expected_action - env.config.target_action) < 0.35
    assert abs(result.greedy_action - env.config.target_action) < 0.20
    assert result.entropy > 0.0
