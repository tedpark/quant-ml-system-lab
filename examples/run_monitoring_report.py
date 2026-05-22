from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.monitoring import feature_drift_report, rolling_performance
from quant_ml_lab.validation import backtest_pair_baseline


def main() -> None:
    df = make_synthetic_pair(SyntheticPairConfig(periods=520))
    expected, actual = train_test_split_time(df, train_fraction=0.7)
    backtest_result, _ = backtest_pair_baseline(df)
    drift = feature_drift_report(expected, actual, ["asset_a", "asset_b"])
    performance = rolling_performance(backtest_result["net_return"], window=63)

    report = {
        "dataset": "synthetic_pair",
        "disclosure": "Synthetic data only. Production alert rules are excluded.",
        "feature_drift": [metric.as_dict() for metric in drift],
        "rolling_performance": performance.as_dict(),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/monitoring_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
