from quant_ml_lab.data import SyntheticRegimePairConfig, make_synthetic_regime_pair
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.meta_labeling import (
    MetaLabelConfig,
    build_meta_label_frame,
    run_meta_label_readiness_matrix,
)
from quant_ml_lab.strategy_selector import StrategySelectorConfig


def test_build_meta_label_frame_has_trade_skip_labels():
    prices = make_synthetic_regime_pair(SyntheticRegimePairConfig(periods=360, seed=71))
    train = prices.iloc[:240]
    validation = prices.iloc[240:]
    dataset = build_hmm_rl_dataset(train, validation)

    label_frame = build_meta_label_frame(
        dataset.frame,
        StrategySelectorConfig(transaction_cost_bps=2.0),
        MetaLabelConfig(horizon=3, transaction_cost_bps=2.0),
    )

    assert not label_frame.empty
    assert {"candidate", "forward_return", "meta_label"}.issubset(label_frame.columns)
    assert set(label_frame["meta_label"].unique()).issubset({0, 1})


def test_run_meta_label_readiness_matrix_returns_summary():
    price_sets = {
        "regime_seed_81": make_synthetic_regime_pair(
            SyntheticRegimePairConfig(periods=360, seed=81)
        ),
        "regime_seed_82": make_synthetic_regime_pair(
            SyntheticRegimePairConfig(periods=360, seed=82)
        ),
    }

    report = run_meta_label_readiness_matrix(
        price_sets,
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        meta_config=MetaLabelConfig(horizon=3, transaction_cost_bps=2.0),
        train_fraction=0.65,
    )

    assert len(report.cases) == 2
    assert report.summary["cases"] == 2
    assert "meta_label_ready" in report.summary
    assert "research_decision" in report.summary
    assert report.cases[0].diagnostics
