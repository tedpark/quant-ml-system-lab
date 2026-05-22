from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.regime import RegimeFilterConfig, backtest_regime_filter
from quant_ml_lab.validation import BacktestConfig
from report_utils import markdown_table, write_markdown


def main() -> None:
    df = make_synthetic_pair(SyntheticPairConfig(periods=520, seed=7))
    train, test = train_test_split_time(df, train_fraction=0.7)
    bt_config = BacktestConfig()
    regime_config = RegimeFilterConfig()
    _, report = backtest_regime_filter(train, test, bt_config, regime_config)

    payload = {
        "dataset": "synthetic_pair",
        "disclosure": "Synthetic data only. This is a validation artifact, not a trading recommendation.",
        "protocol": {
            "train_fraction": 0.7,
            "cost_bps": bt_config.transaction_cost_bps,
            "regime_threshold_estimation": "High-volatility threshold is estimated on the train split only.",
            "vol_lookback": regime_config.vol_lookback,
            "high_vol_quantile": regime_config.high_vol_quantile,
            "high_vol_multiplier": regime_config.high_vol_multiplier,
        },
        "result": report.as_dict(),
        "limitations": [
            "Synthetic data does not represent a production universe.",
            "The regime filter is a public volatility proxy, not a proprietary HMM model.",
            "No parameter search is performed in this public example.",
        ],
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline_vs_regime_filter.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    rows = [
        {"variant": "baseline", **report.baseline.as_dict()},
        {"variant": "regime_filtered", **report.filtered.as_dict()},
    ]
    table = markdown_table(
        rows,
        [
            "variant",
            "total_return",
            "sharpe",
            "sortino",
            "max_drawdown",
            "win_rate",
            "turnover",
            "trades",
        ],
    )
    markdown = f"""# Baseline vs Regime Filter

This report compares a synthetic mean-reversion pair baseline against a public high-volatility regime filter.

## Protocol

- Dataset: synthetic pair data
- Split: first 70% train, final 30% test
- Transaction cost: {bt_config.transaction_cost_bps:.2f} bps per turnover unit
- Regime threshold: estimated on the train split only
- Regime proxy: rolling spread volatility
- High-volatility exposure multiplier: {regime_config.high_vol_multiplier:.2f}

## Result

High-volatility share in test: {report.high_vol_share:.6f}

{table}

## Limitations

- Synthetic data does not represent a production universe.
- The regime filter is a public volatility proxy, not a proprietary HMM model.
- No parameter search is performed in this public example.
"""
    write_markdown("docs/benchmark_reports/baseline_vs_regime_filter.md", markdown)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
