from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.sizing import (
    apply_sizing_policy,
    compare_sizing_policies,
    rolling_cvar_multiplier,
)
from quant_ml_lab.validation import backtest_pair_baseline


def test_rolling_cvar_multiplier_is_bounded():
    df = make_synthetic_pair(SyntheticPairConfig(periods=220))
    _, test = train_test_split_time(df)
    baseline, _ = backtest_pair_baseline(test)

    multiplier = rolling_cvar_multiplier(baseline["net_return"], floor=0.2)

    assert len(multiplier) == len(baseline)
    assert multiplier.min() >= 0.2
    assert multiplier.max() <= 1.0


def test_apply_sizing_policy_returns_metrics():
    df = make_synthetic_pair(SyntheticPairConfig(periods=220))
    _, test = train_test_split_time(df)
    baseline, _ = backtest_pair_baseline(test)

    result, comparison = apply_sizing_policy(baseline, "qrdqn_cvar_proxy")

    assert "qrdqn_cvar_proxy_net_return" in result.columns
    assert comparison.policy == "qrdqn_cvar_proxy"
    assert 0.0 <= comparison.min_multiplier <= comparison.max_multiplier <= 1.0


def test_compare_sizing_policies_includes_all_public_proxies():
    df = make_synthetic_pair(SyntheticPairConfig(periods=220))
    _, test = train_test_split_time(df)
    baseline, _ = backtest_pair_baseline(test)

    comparisons = compare_sizing_policies(baseline)

    assert [comparison.policy for comparison in comparisons] == [
        "base",
        "sac_proxy",
        "ppo_proxy",
        "qrdqn_cvar_proxy",
    ]
