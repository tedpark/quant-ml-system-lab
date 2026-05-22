from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.validation import BacktestConfig, backtest_pair_baseline


def main() -> None:
    df = make_synthetic_pair(SyntheticPairConfig())
    train, test = train_test_split_time(df)

    config = BacktestConfig()
    _, train_metrics = backtest_pair_baseline(train, config)
    _, test_metrics = backtest_pair_baseline(test, config)

    report = {
        "dataset": "synthetic_pair",
        "disclosure": "Synthetic data only. No production strategy parameters are included.",
        "config": {
            "entry_z": config.entry_z,
            "exit_z": config.exit_z,
            "lookback": config.lookback,
            "transaction_cost_bps": config.transaction_cost_bps,
        },
        "train": train_metrics.as_dict(),
        "test": test_metrics.as_dict(),
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/sample_backtest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
