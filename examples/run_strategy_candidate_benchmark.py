from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report_utils import markdown_table, write_markdown

from quant_ml_lab.data import SyntheticRegimePairConfig, make_synthetic_regime_pair
from quant_ml_lab.strategy_candidate_benchmark import run_strategy_candidate_benchmark_matrix
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
    report = run_strategy_candidate_benchmark_matrix(
        price_sets=price_sets,
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        train_fraction=0.65,
    )
    payload = {
        "experiment": "strategy_candidate_benchmark",
        "purpose": "Candidate-level benchmark decomposition on multi-regime synthetic data.",
        **report.as_dict(),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/strategy_candidate_benchmark.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_markdown(
        "docs/benchmark_reports/strategy_candidate_benchmark.md",
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
        selected_metrics = case["selected_metrics"]
        assert isinstance(selected_metrics, dict)
        rows.append(
            {
                "dataset": case["dataset_id"],
                "selected_sharpe": selected_metrics["sharpe"],
                "best_candidate": case["best_candidate_by_sharpe"],
                "best_candidate_sharpe": case["best_candidate_sharpe"],
                "selected_minus_best_sharpe": case["selected_minus_best_sharpe"],
            }
        )
    candidate_rows = [
        {
            "candidate": name,
            "mean_sharpe": value,
            "mean_total_return": summary["mean_candidate_total_return"][name],
            "best_count": summary["best_candidate_counts"][name],
        }
        for name, value in summary["mean_candidate_sharpe"].items()
    ]
    return f"""# Strategy Candidate Benchmark

This report decomposes the public strategy family on multi-regime synthetic data.
It identifies which simple candidate policies are hard baselines for a learned
allocator to beat.

## Summary

- cases: `{summary["cases"]}`
- mean selected Sharpe: `{summary["mean_selected_sharpe"]}`
- mean selected minus best Sharpe: `{summary["mean_selected_minus_best_sharpe"]}`
- worst selected minus best Sharpe: `{summary["worst_selected_minus_best_sharpe"]}`
- selected matches best cases: `{summary["selected_matches_best_cases"]}`
- strongest candidate by mean Sharpe: `{summary["strongest_candidate_by_mean_sharpe"]}`
- benchmark-ready: `{summary["benchmark_ready"]}`

## Dataset Cases

{markdown_table(rows, ["dataset", "selected_sharpe", "best_candidate", "best_candidate_sharpe", "selected_minus_best_sharpe"])}

## Candidate Averages

{markdown_table(candidate_rows, ["candidate", "mean_sharpe", "mean_total_return", "best_count"])}

## Interpretation

If a simple candidate repeatedly beats the selector or RL allocator, the next model
should not be tuned until the benchmark gap is explained. This protects the lab from
mistaking model complexity for strategy quality.
"""


if __name__ == "__main__":
    main()
