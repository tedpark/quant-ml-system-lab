from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report_utils import markdown_table, write_markdown

from quant_ml_lab.data import SyntheticRegimePairConfig, make_synthetic_regime_pair
from quant_ml_lab.meta_labeling import MetaLabelConfig, run_meta_label_readiness_matrix
from quant_ml_lab.strategy_selector import StrategySelectorConfig


def main() -> None:
    price_sets = {
        "regime_seed_501": make_synthetic_regime_pair(
            SyntheticRegimePairConfig(periods=760, seed=501)
        ),
        "regime_seed_502": make_synthetic_regime_pair(
            SyntheticRegimePairConfig(periods=760, seed=502)
        ),
        "regime_seed_503": make_synthetic_regime_pair(
            SyntheticRegimePairConfig(periods=760, seed=503)
        ),
    }
    report = run_meta_label_readiness_matrix(
        price_sets=price_sets,
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        meta_config=MetaLabelConfig(horizon=5, transaction_cost_bps=2.0),
        train_fraction=0.65,
    )
    payload = {
        "experiment": "meta_label_readiness",
        "purpose": "Public meta-label readiness diagnostics before RL allocation.",
        **report.as_dict(),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/meta_label_readiness.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_markdown(
        "docs/benchmark_reports/meta_label_readiness.md",
        _markdown_report(payload),
    )
    print(json.dumps(payload, indent=2))


def _markdown_report(payload: dict[str, object]) -> str:
    summary = payload["summary"]
    assert isinstance(summary, dict)
    cases = payload["cases"]
    assert isinstance(cases, list)
    rows = []
    for case in cases:
        assert isinstance(case, dict)
        diagnostics = case["diagnostics"]
        assert isinstance(diagnostics, dict)
        for candidate, metrics in diagnostics.items():
            assert isinstance(metrics, dict)
            rows.append(
                {
                    "dataset": case["dataset_id"],
                    "candidate": candidate,
                    "events": metrics["events"],
                    "validation_events": metrics["validation_events"],
                    "positive_rate": metrics["positive_rate"],
                    "validation_positive_rate": metrics["validation_positive_rate"],
                    "mean_forward_return": metrics["mean_forward_return"],
                    "validation_mean_forward_return": metrics[
                        "validation_mean_forward_return"
                    ],
                    "best_feature": metrics["best_feature"],
                    "best_feature_bin": metrics["best_feature_bin"],
                    "best_bin_mean_forward_return": metrics["best_bin_mean_forward_return"],
                    "best_bin_lift": metrics["best_bin_lift"],
                    "ready": metrics["meta_label_ready"],
                }
            )

    return f"""# Meta-Label Readiness

This report checks whether the public candidate signals have enough label quality
for a supervised trade/skip filter before SAC allocation.

## Summary

- cases: `{summary["cases"]}`
- candidate diagnostics: `{summary["candidate_diagnostics"]}`
- ready candidate diagnostics: `{summary["ready_candidate_diagnostics"]}`
- ready candidate rate: `{summary["ready_candidate_rate"]}`
- ready counts: `{summary["ready_counts"]}`
- mean best-bin lift: `{summary["mean_best_bin_lift"]}`
- best candidate by lift: `{summary["best_candidate_by_lift"]}`
- best feature by lift: `{summary["best_feature_by_lift"]}`
- best bin lift: `{summary["best_bin_lift"]}`
- meta-label ready: `{summary["meta_label_ready"]}`
- research decision: `{summary["research_decision"]}`

## Candidate Diagnostics

{markdown_table(rows, ["dataset", "candidate", "events", "validation_events", "positive_rate", "validation_positive_rate", "mean_forward_return", "validation_mean_forward_return", "best_feature", "best_feature_bin", "best_bin_mean_forward_return", "best_bin_lift", "ready"])}

## Interpretation

The best feature bucket is selected on the earlier slice of each candidate's
label history and evaluated on the later slice. Meta-labeling should be used as a
filter only when the validation bucket keeps positive lift and positive mean
forward return. If this gate fails, the next work is better labels, features, or
candidate signals, not more SAC tuning.
"""


if __name__ == "__main__":
    main()
