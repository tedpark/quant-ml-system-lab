from pathlib import Path

import numpy as np

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.strategy_selector import StrategySelectorConfig, build_strategy_candidates
from quant_ml_lab.strategy_selector_sac import (
    SACStrategyAllocatorEnv,
    SACStrategyAllocatorEnvConfig,
    run_strategy_allocator_sac_walk_forward,
    train_validate_strategy_allocator_sac,
)
from quant_ml_lab.torch_sac import TorchSACConfig
from quant_ml_lab.walk_forward import WalkForwardConfig


def _small_dataset(periods: int = 420, seed: int = 51):
    prices = make_synthetic_pair(SyntheticPairConfig(periods=periods, seed=seed))
    train, test = train_test_split_time(prices)
    return build_hmm_rl_dataset(train, test)


def test_sac_strategy_allocator_env_maps_continuous_action_to_step():
    dataset = _small_dataset()
    candidates = build_strategy_candidates(dataset.frame)
    env = SACStrategyAllocatorEnv(dataset.frame, dataset.feature_columns, candidates)

    state = env.reset()
    next_state, reward, done = env.step(np.zeros(env.action_dim, dtype=np.float32))

    assert state.shape == next_state.shape
    assert isinstance(reward, float)
    assert done is False
    assert len(env.weights) == 1
    assert abs(float(env.weights[0].sum()) - 1.0) < 1e-9


def test_train_validate_strategy_allocator_sac_returns_report(tmp_path: Path):
    dataset = _small_dataset(periods=430, seed=52)
    split = int(len(dataset.frame) * 0.65)
    train_frame = dataset.frame.iloc[:split].copy()
    validation_frame = dataset.frame.iloc[split:].copy()
    checkpoint = tmp_path / "allocator_sac.pt"

    report, train_output, validation_output = train_validate_strategy_allocator_sac(
        train_frame=train_frame,
        validation_frame=validation_frame,
        feature_columns=dataset.feature_columns,
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        env_config=SACStrategyAllocatorEnvConfig(transaction_cost_bps=2.0),
        sac_config=TorchSACConfig(
            steps=96,
            warmup_steps=16,
            batch_size=16,
            gamma=0.9,
            hidden_dim=16,
            target_entropy=-5.0,
            seed=53,
        ),
        checkpoint_path=checkpoint,
    )

    assert checkpoint.exists()
    assert report.validation_mean_weights
    assert abs(sum(report.validation_mean_weights.values()) - 1.0) < 1e-6
    assert "sac_allocator_position" in train_output.columns
    assert "sac_allocator_net_return" in validation_output.columns
    assert report.diagnostics["alpha_is_positive"] is True


def test_run_strategy_allocator_sac_walk_forward_returns_summary():
    prices = make_synthetic_pair(SyntheticPairConfig(periods=500, seed=54))

    report = run_strategy_allocator_sac_walk_forward(
        prices,
        wf_config=WalkForwardConfig(train_size=300, test_size=80, step_size=80),
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        env_config=SACStrategyAllocatorEnvConfig(transaction_cost_bps=2.0),
        sac_config=TorchSACConfig(
            steps=64,
            warmup_steps=16,
            batch_size=16,
            gamma=0.9,
            hidden_dim=16,
            target_entropy=-5.0,
            seed=55,
        ),
    )

    assert report.folds
    assert report.summary["folds"] == len(report.folds)
    assert "mean_sharpe_delta" in report.summary
    assert "robust_ready" in report.summary
