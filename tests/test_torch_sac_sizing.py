import numpy as np

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.torch_sac import TorchSACConfig
from quant_ml_lab.torch_sac_sizing import HMMSACPositionSizingEnv, train_hmm_sac_sizer


def test_hmm_sac_sizing_env_step_shapes():
    df = make_synthetic_pair(SyntheticPairConfig(periods=260))
    train, test = train_test_split_time(df)
    dataset = build_hmm_rl_dataset(train, test)
    env = HMMSACPositionSizingEnv(dataset.frame, dataset.feature_columns)

    state = env.reset()
    next_state, reward, done = env.step(np.array([0.0], dtype=np.float32))

    assert state.shape == (4,)
    assert next_state.shape == (4,)
    assert isinstance(reward, float)
    assert done is False


def test_train_hmm_sac_sizer_returns_metrics():
    df = make_synthetic_pair(SyntheticPairConfig(periods=260))
    train, test = train_test_split_time(df)
    dataset = build_hmm_rl_dataset(train, test)

    report, result = train_hmm_sac_sizer(
        dataset.frame,
        dataset.feature_columns,
        TorchSACConfig(steps=80, warmup_steps=16, batch_size=16, hidden_dim=16, seed=3),
    )

    assert "sac_net_return" in result.columns
    assert 0.0 <= report.mean_multiplier <= 1.0
    assert report.metrics.trades >= 0
