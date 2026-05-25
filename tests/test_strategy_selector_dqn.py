from pathlib import Path

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.strategy_selector import StrategySelectorConfig, build_strategy_candidates
from quant_ml_lab.strategy_selector_dqn import (
    DQNSelectorConfig,
    StrategySelectorEnv,
    train_validate_strategy_selector_dqn,
)


def _small_dataset(periods: int = 360, seed: int = 31):
    prices = make_synthetic_pair(SyntheticPairConfig(periods=periods, seed=seed))
    train, test = train_test_split_time(prices)
    return build_hmm_rl_dataset(train, test)


def test_strategy_selector_env_steps_with_discrete_action():
    dataset = _small_dataset()
    candidates = build_strategy_candidates(dataset.frame)
    env = StrategySelectorEnv(dataset.frame, dataset.feature_columns, candidates)

    state = env.reset()
    next_state, reward, done = env.step(2)

    assert state.shape == next_state.shape
    assert isinstance(reward, float)
    assert done is False


def test_train_validate_strategy_selector_dqn_returns_reports(tmp_path: Path):
    dataset = _small_dataset(periods=420, seed=32)
    split = int(len(dataset.frame) * 0.65)
    train_frame = dataset.frame.iloc[:split].copy()
    validation_frame = dataset.frame.iloc[split:].copy()
    checkpoint = tmp_path / "selector_dqn.pt"

    report, train_output, validation_output = train_validate_strategy_selector_dqn(
        train_frame=train_frame,
        validation_frame=validation_frame,
        feature_columns=dataset.feature_columns,
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        dqn_config=DQNSelectorConfig(episodes=2, batch_size=16, seed=19),
        checkpoint_path=checkpoint,
    )

    assert checkpoint.exists()
    assert report.validation_selection_counts
    assert report.rule_based_selection_counts
    assert report.random_selection_counts
    assert report.candidate_validation_metrics
    assert "loss_is_finite" in report.diagnostics
    assert report.loss_trace_tail
    assert report.q_value_trace_tail
    assert report.validation_action_concentration <= 1.0
    assert "selected_strategy" in train_output.columns
    assert "selected_net_return" in validation_output.columns
    assert report.validation_metrics["trades"] >= 0
