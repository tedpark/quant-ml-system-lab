from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair
from quant_ml_lab.strategy import PairRLStrategyConfig, build_pair_rl_strategy
from quant_ml_lab.torch_sac import TorchSACConfig


def test_build_pair_rl_strategy_produces_signal_and_gates(tmp_path):
    prices = make_synthetic_pair(SyntheticPairConfig(periods=420, seed=12))

    report, validation_output = build_pair_rl_strategy(
        prices,
        config=PairRLStrategyConfig(
            seeds=(3,),
            min_validation_rows=30,
            checkpoint_dir=str(tmp_path),
            require_baseline_outperformance=False,
        ),
        sac_config=TorchSACConfig(
            steps=90,
            warmup_steps=16,
            batch_size=16,
            hidden_dim=16,
            seed=3,
        ),
    )

    assert report.strategy_name == "pair_mean_reversion_hmm_sac_sizer"
    assert report.best_seed == 3
    assert report.infrastructure_gates["best_checkpoint_exists"] is True
    assert abs(report.latest_signal.sized_position) <= 1.0
    assert report.latest_signal.leg_a_target == -report.latest_signal.sized_position
    assert report.latest_signal.leg_b_target == report.latest_signal.sized_position
    assert "sac_sized_position" in validation_output.columns


def test_build_pair_rl_strategy_requires_price_columns():
    prices = make_synthetic_pair(SyntheticPairConfig(periods=300)).drop(columns=["asset_b"])

    try:
        build_pair_rl_strategy(prices, config=PairRLStrategyConfig(seeds=(1,)))
    except ValueError as exc:
        assert "missing columns" in str(exc)
    else:
        raise AssertionError("expected ValueError")
