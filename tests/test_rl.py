from quant_ml_lab.rl import (
    GridWorld,
    GridWorldConfig,
    QLearningConfig,
    evaluate_policy,
    train_q_learning,
)


def test_grid_world_reaches_goal_with_manual_policy():
    env = GridWorld(GridWorldConfig())
    state = env.reset()
    assert state == 0

    for action in [1, 1, 1, 2, 2, 2]:
        state, _, done = env.step(action)

    assert done
    assert env.position == env.config.goal


def test_q_learning_trains_policy_with_expected_shape():
    env = GridWorld(GridWorldConfig(max_steps=32))
    result = train_q_learning(env, QLearningConfig(episodes=120, seed=123))

    assert result.q_table.shape == (env.n_states, env.n_actions)
    assert len(result.episode_returns) == 120
    assert len(result.policy) == env.n_states


def test_learned_policy_gets_positive_return():
    env = GridWorld(GridWorldConfig(max_steps=32))
    result = train_q_learning(env, QLearningConfig(episodes=300, seed=7))
    total_reward, path = evaluate_policy(env, result.policy)

    assert total_reward > 0.0
    assert path[-1] == env.config.goal
