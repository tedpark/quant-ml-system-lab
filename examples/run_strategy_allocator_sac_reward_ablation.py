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
    run_strategy_allocator_sac_reward_ablation,
)
from quant_ml_lab.torch_sac import TorchSACConfig
from quant_ml_lab.walk_forward import WalkForwardConfig


def main() -> None:
    price_sets = {
        "synthetic_seed_404": make_synthetic_pair(SyntheticPairConfig(periods=620, seed=404)),
        "synthetic_seed_405": make_synthetic_pair(SyntheticPairConfig(periods=620, seed=405)),
    }
    report = run_strategy_allocator_sac_reward_ablation(
        price_sets=price_sets,
        wf_config=WalkForwardConfig(train_size=360, test_size=100, step_size=100),
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        env_config=SACStrategyAllocatorEnvConfig(transaction_cost_bps=2.0),
        sac_config=TorchSACConfig(
            steps=120,
            warmup_steps=32,
            batch_size=32,
            gamma=0.95,
            hidden_dim=32,
            target_entropy=-5.0,
            seed=61,
        ),
    )
    payload = {
        "experiment": "strategy_allocator_sac_reward_ablation",
        "purpose": "Reward component ablation for SAC strategy allocator.",
        **report.as_dict(),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/strategy_allocator_sac_reward_ablation.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    write_markdown(
        "docs/benchmark_reports/strategy_allocator_sac_reward_ablation.md",
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
                "ablation": case["ablation"],
                "dataset": case["dataset_id"],
                "sharpe_delta": wf_summary["mean_sharpe_delta"],
                "return_delta": wf_summary["mean_total_return_delta"],
                "positive_sharpe_folds": wf_summary["positive_sharpe_delta_folds"],
                "positive_return_folds": wf_summary["positive_return_delta_folds"],
                "robust_ready": wf_summary["robust_ready"],
            }
        )
    return f"""# SAC Allocator Reward Ablation

This report removes one public reward component at a time from the SAC strategy allocator.
The goal is to identify whether reward shaping is improving robustness or hiding a fragile
policy behind hand-tuned penalties.

## Summary

- cases: `{summary["cases"]}`
- ablations: `{summary["ablations"]}`
- best ablation by Sharpe: `{summary["best_ablation_by_sharpe"]}`
- worst ablation by Sharpe: `{summary["worst_ablation_by_sharpe"]}`
- best mean Sharpe delta: `{summary["best_mean_sharpe_delta"]}`
- full reward mean Sharpe delta: `{summary["full_reward_mean_sharpe_delta"]}`
- best minus full Sharpe delta: `{summary["best_minus_full_sharpe_delta"]}`
- mean Sharpe delta: `{summary["mean_sharpe_delta"]}`
- worst Sharpe delta: `{summary["worst_sharpe_delta"]}`
- mean total return delta: `{summary["mean_total_return_delta"]}`
- robust case rate: `{summary["robust_case_rate"]}`
- robust-ready: `{summary["robust_ready"]}`

## Case Matrix

{markdown_table(rows, ["ablation", "dataset", "sharpe_delta", "return_delta", "positive_sharpe_folds", "positive_return_folds", "robust_ready"])}

## Interpretation

If removing a penalty improves the result, the current reward is probably over-shaped for
this public environment. If all ablations remain unstable, the bottleneck is more likely
data coverage and environment design than a single reward coefficient.
"""


if __name__ == "__main__":
    main()
