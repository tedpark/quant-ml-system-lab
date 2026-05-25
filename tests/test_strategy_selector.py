from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.strategy_selector import (
    StrategySelectorConfig,
    build_strategy_candidates,
    run_strategy_selector,
    select_strategy_by_regime,
)


def test_strategy_candidates_include_meta_control_family():
    prices = make_synthetic_pair(SyntheticPairConfig(periods=320, seed=5))
    train, test = train_test_split_time(prices)
    dataset = build_hmm_rl_dataset(train, test)

    candidates = build_strategy_candidates(dataset.frame)

    assert set(candidates) == {
        "no_trade",
        "mean_reversion_full",
        "mean_reversion_low_risk",
        "volatility_defensive",
        "cvar_defensive",
    }
    for candidate in candidates.values():
        assert len(candidate.position) == len(dataset.frame)
        assert candidate.multiplier.abs().max() <= 1.0


def test_strategy_selector_prefers_defensive_policy_in_high_vol():
    prices = make_synthetic_pair(SyntheticPairConfig(periods=320, seed=6))
    train, test = train_test_split_time(prices)
    dataset = build_hmm_rl_dataset(train, test)
    frame = dataset.frame.copy()
    frame["baseline_position"] = 1.0
    frame["high_vol_prob"] = 0.9
    candidates = build_strategy_candidates(frame)

    selected, risk_budget = select_strategy_by_regime(frame, candidates)

    assert set(selected.unique()) == {"volatility_defensive"}
    assert risk_budget.max() <= 1.0


def test_run_strategy_selector_returns_metrics_and_counts():
    prices = make_synthetic_pair(SyntheticPairConfig(periods=360, seed=7))
    train, test = train_test_split_time(prices)
    dataset = build_hmm_rl_dataset(train, test)

    result, report = run_strategy_selector(
        dataset.frame,
        StrategySelectorConfig(high_vol_threshold=0.5),
    )

    assert "selected_strategy" in result.columns
    assert "selected_net_return" in result.columns
    assert report.selected_counts
    assert report.selected_metrics.trades >= 0
    assert 0.0 <= report.min_risk_budget <= report.max_risk_budget <= 1.0
