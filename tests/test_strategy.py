from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair
from quant_ml_lab.strategy import (
    PairRLStrategyConfig,
    analyze_regime_behavior,
    build_pair_rl_strategy,
)
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
    assert report.regime_behavior.normal.rows + report.regime_behavior.high_vol.rows == len(
        validation_output
    )
    assert report.regime_behavior.learned_regime_response
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


def test_analyze_regime_behavior_detects_defensive_high_vol_response():
    prices = make_synthetic_pair(SyntheticPairConfig(periods=420, seed=15))
    _, validation_output = build_pair_rl_strategy(
        prices,
        config=PairRLStrategyConfig(
            seeds=(3,),
            min_validation_rows=30,
            require_baseline_outperformance=False,
        ),
        sac_config=TorchSACConfig(
            steps=80,
            warmup_steps=16,
            batch_size=16,
            hidden_dim=16,
            seed=3,
        ),
    )
    validation_output = validation_output.copy()
    split = len(validation_output) // 2
    validation_output["high_vol_prob"] = 0.1
    validation_output.iloc[split:, validation_output.columns.get_loc("high_vol_prob")] = 0.9
    validation_output.iloc[
        split:, validation_output.columns.get_loc("sac_sized_position")
    ] *= 0.25

    report = analyze_regime_behavior(validation_output)

    assert report.high_vol.rows > 0
    assert report.normal.rows > 0
    assert report.learned_regime_response == "defensive_sizing_in_high_vol"
