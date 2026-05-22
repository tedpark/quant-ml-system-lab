from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.experiments import JsonlExperimentTracker, LocalModelRegistry, ModelRegistryEntry
from quant_ml_lab.validation import BacktestConfig, backtest_pair_baseline


def main() -> None:
    df = make_synthetic_pair(SyntheticPairConfig(periods=520))
    _, test = train_test_split_time(df)
    config = BacktestConfig()
    _, metrics = backtest_pair_baseline(test, config)

    tracker = JsonlExperimentTracker()
    run = tracker.log_run(
        name="synthetic-pair-baseline",
        params={
            "entry_z": config.entry_z,
            "exit_z": config.exit_z,
            "lookback": config.lookback,
            "transaction_cost_bps": config.transaction_cost_bps,
        },
        metrics=metrics.as_dict(),
        tags={"dataset": "synthetic_pair", "visibility": "public_demo"},
    )

    registry = LocalModelRegistry()
    registry.promote(
        ModelRegistryEntry(
            model_name="demo-risk-model",
            model_version=run.run_id[:8],
            stage="demo",
            metrics=metrics.as_dict(),
        )
    )

    report = {
        "disclosure": "Demo tracking only. No production model artifact is included.",
        "run": run.as_dict(),
        "registry": registry.current().as_dict() if registry.current() else None,
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/experiment_demo.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
