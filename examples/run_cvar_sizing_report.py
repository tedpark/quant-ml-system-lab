from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.risk import empirical_cvar
from quant_ml_lab.sizing import apply_sizing_policy
from quant_ml_lab.validation import backtest_pair_baseline
from report_utils import markdown_table, write_markdown


def main() -> None:
    df = make_synthetic_pair(SyntheticPairConfig(periods=520, seed=7))
    _, test = train_test_split_time(df, train_fraction=0.7)
    baseline_result, baseline_metrics = backtest_pair_baseline(test)
    qrdqn_result, qrdqn_proxy = apply_sizing_policy(baseline_result, "qrdqn_cvar_proxy")

    baseline_cvar_5 = empirical_cvar(baseline_result["net_return"].to_numpy(), alpha=0.05)
    baseline_cvar_10 = empirical_cvar(baseline_result["net_return"].to_numpy(), alpha=0.10)
    qrdqn_returns = qrdqn_result["qrdqn_cvar_proxy_net_return"].to_numpy()
    qrdqn_cvar_5 = empirical_cvar(qrdqn_returns, alpha=0.05)
    qrdqn_cvar_10 = empirical_cvar(qrdqn_returns, alpha=0.10)

    payload = {
        "dataset": "synthetic_pair",
        "disclosure": "Synthetic data only. CVaR sizing is a public risk-control example.",
        "baseline": {
            "metrics": baseline_metrics.as_dict(),
            "cvar_5": baseline_cvar_5,
            "cvar_10": baseline_cvar_10,
        },
        "qrdqn_cvar_proxy": {
            "description": qrdqn_proxy.description,
            "metrics": qrdqn_proxy.metrics.as_dict(),
            "mean_multiplier": qrdqn_proxy.mean_multiplier,
            "min_multiplier": qrdqn_proxy.min_multiplier,
            "max_multiplier": qrdqn_proxy.max_multiplier,
            "cvar_5": qrdqn_cvar_5,
            "cvar_10": qrdqn_cvar_10,
        },
        "limitations": [
            "This is a rolling empirical CVaR proxy, not a trained production QR-DQN checkpoint.",
            "The example demonstrates the risk-control contract, not alpha generation.",
        ],
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/cvar_sizing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    rows = [
        {
            "variant": "baseline",
            **baseline_metrics.as_dict(),
            "cvar_5": baseline_cvar_5,
            "cvar_10": baseline_cvar_10,
            "mean_multiplier": 1.0,
        },
        {
            "variant": "qrdqn_cvar_proxy",
            **qrdqn_proxy.metrics.as_dict(),
            "cvar_5": qrdqn_cvar_5,
            "cvar_10": qrdqn_cvar_10,
            "mean_multiplier": qrdqn_proxy.mean_multiplier,
        },
    ]
    table = markdown_table(
        rows,
        [
            "variant",
            "total_return",
            "sharpe",
            "max_drawdown",
            "cvar_5",
            "cvar_10",
            "turnover",
            "mean_multiplier",
        ],
    )
    markdown = f"""# CVaR Sizing

This report compares an unscaled synthetic pair baseline with a public distributional-RL-style CVaR sizing proxy.

## Protocol

- Dataset: synthetic pair data
- Baseline: mean-reversion pair signal
- Risk-control proxy: rolling lower-tail empirical CVaR
- Position multiplier: reduced when downside tail risk exceeds target

## Result

{table}

## Limitations

- This is a rolling empirical CVaR proxy, not a trained production QR-DQN checkpoint.
- The example demonstrates the risk-control contract, not alpha generation.
- Synthetic data does not represent a production universe.
"""
    write_markdown("docs/benchmark_reports/cvar_sizing.md", markdown)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
