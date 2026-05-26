from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from report_utils import markdown_table, write_markdown

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair
from quant_ml_lab.strategy_selector import StrategySelectorConfig
from quant_ml_lab.strategy_selector_sac import (
    SACStrategyAllocatorEnvConfig,
    run_strategy_allocator_sac_robustness_matrix,
)
from quant_ml_lab.torch_sac import TorchSACConfig
from quant_ml_lab.walk_forward import WalkForwardConfig


def main() -> None:
    price_sets = {
        "synthetic_seed_404": make_synthetic_pair(SyntheticPairConfig(periods=620, seed=404)),
        "synthetic_seed_405": make_synthetic_pair(SyntheticPairConfig(periods=620, seed=405)),
    }
    report = run_strategy_allocator_sac_robustness_matrix(
        price_sets=price_sets,
        sac_seeds=(61, 62),
        transaction_cost_bps_values=(2.0, 5.0),
        wf_config=WalkForwardConfig(train_size=360, test_size=100, step_size=100),
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        env_config=SACStrategyAllocatorEnvConfig(transaction_cost_bps=2.0),
        sac_config=TorchSACConfig(
            steps=130,
            warmup_steps=32,
            batch_size=32,
            gamma=0.95,
            hidden_dim=32,
            target_entropy=-5.0,
            seed=61,
        ),
    )
    payload = {
        "experiment": "strategy_allocator_sac_robustness",
        "purpose": "Multi-seed and cost-stress validation for SAC strategy allocator.",
        **report.as_dict(),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/strategy_allocator_sac_robustness.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_markdown(
        "docs/benchmark_reports/strategy_allocator_sac_robustness.md",
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
        wf_summary = case["walk_forward_summary"]
        assert isinstance(wf_summary, dict)
        rows.append(
            {
                "dataset": case["dataset_id"],
                "sac_seed": case["sac_seed"],
                "cost_bps": case["transaction_cost_bps"],
                "sharpe_delta": wf_summary["mean_sharpe_delta"],
                "return_delta": wf_summary["mean_total_return_delta"],
                "positive_sharpe_folds": wf_summary["positive_sharpe_delta_folds"],
                "positive_return_folds": wf_summary["positive_return_delta_folds"],
                "robust_ready": wf_summary["robust_ready"],
            }
        )
    return f"""# SAC Allocator Robustness Matrix

This report stress-tests the public SAC strategy allocator across synthetic data seeds,
SAC random seeds, and transaction-cost assumptions. It is intended to falsify fragile
RL results before any private research or paper-trading workflow uses the pattern.

## Summary

- cases: `{summary["cases"]}`
- mean Sharpe delta: `{summary["mean_sharpe_delta"]}`
- median Sharpe delta: `{summary["median_sharpe_delta"]}`
- worst Sharpe delta: `{summary["worst_sharpe_delta"]}`
- mean total return delta: `{summary["mean_total_return_delta"]}`
- worst total return delta: `{summary["worst_total_return_delta"]}`
- positive Sharpe case rate: `{summary["positive_sharpe_rate"]}`
- positive return case rate: `{summary["positive_return_rate"]}`
- robust case rate: `{summary["robust_case_rate"]}`
- robust-ready: `{summary["robust_ready"]}`

## Case Matrix

{markdown_table(rows, ["dataset", "sac_seed", "cost_bps", "sharpe_delta", "return_delta", "positive_sharpe_folds", "positive_return_folds", "robust_ready"])}

## Interpretation

If the robustness gate is false, the correct conclusion is not that SAC is impossible.
The correct conclusion is that the current public environment, data coverage, reward,
and validation design are still insufficient for strategy claims.
"""


if __name__ == "__main__":
    main()
