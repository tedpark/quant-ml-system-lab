from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.sizing import compare_sizing_policies
from quant_ml_lab.validation import backtest_pair_baseline
from report_utils import markdown_table, write_markdown


def main() -> None:
    df = make_synthetic_pair(SyntheticPairConfig(periods=520, seed=7))
    _, test = train_test_split_time(df, train_fraction=0.7)
    baseline_result, _ = backtest_pair_baseline(test)
    comparisons = compare_sizing_policies(baseline_result)

    payload = {
        "dataset": "synthetic_pair",
        "disclosure": (
            "Synthetic data only. SAC/PPO/QR-DQN labels are sanitized sizing proxies, "
            "not trained production RL checkpoints."
        ),
        "protocol": {
            "shared_environment": "same synthetic pair test split",
            "shared_signal": "same baseline mean-reversion positions",
            "shared_cost_model": "inferred from baseline transaction-cost settings",
            "role_of_rl": "position sizing / risk multiplier only",
        },
        "comparisons": [comparison.as_dict() for comparison in comparisons],
        "limitations": [
            "The public repo uses deterministic sizing proxies instead of private trained policies.",
            "This report validates comparison structure and metrics, not live trading performance.",
            "A production comparison would add fixed seeds, checkpoints, and full training logs.",
        ],
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/rl_sizing_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    rows = [
        {
            "policy": comparison.policy,
            **comparison.metrics.as_dict(),
            "mean_multiplier": comparison.mean_multiplier,
            "min_multiplier": comparison.min_multiplier,
            "max_multiplier": comparison.max_multiplier,
        }
        for comparison in comparisons
    ]
    table = markdown_table(
        rows,
        [
            "policy",
            "total_return",
            "sharpe",
            "sortino",
            "max_drawdown",
            "win_rate",
            "turnover",
            "mean_multiplier",
            "min_multiplier",
            "max_multiplier",
        ],
    )
    markdown = f"""# RL Sizing Comparison

This report compares sanitized position-sizing policies under one shared synthetic pair test split.

## Protocol

- Dataset: synthetic pair data
- Shared signal: same baseline mean-reversion positions
- Shared cost model: transaction-cost-aware turnover
- RL role: position sizing / risk multiplier only
- Public boundary: deterministic proxies are used instead of private trained checkpoints

## Result

{table}

## Policy Descriptions

""" + "\n".join(
        f"- `{comparison.policy}`: {comparison.description}" for comparison in comparisons
    ) + """

## Limitations

- The public repo uses deterministic sizing proxies instead of private trained policies.
- This report validates comparison structure and metrics, not live trading performance.
- A production comparison would add fixed seeds, checkpoints, and full training logs.
"""
    write_markdown("docs/benchmark_reports/rl_sizing_comparison.md", markdown)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
