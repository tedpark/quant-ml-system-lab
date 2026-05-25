from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.strategy_selector import StrategySelectorConfig
from quant_ml_lab.strategy_selector_dqn import (
    DQNSelectorConfig,
    train_validate_strategy_selector_dqn,
)


def main() -> None:
    prices = make_synthetic_pair(SyntheticPairConfig(periods=820, seed=321))
    train, test = train_test_split_time(prices)
    dataset = build_hmm_rl_dataset(train, test)
    frame = dataset.frame
    rl_train_size = int(len(frame) * 0.65)
    train_frame = frame.iloc[:rl_train_size].copy()
    validation_frame = frame.iloc[rl_train_size:].copy()

    report, _, validation_output = train_validate_strategy_selector_dqn(
        train_frame=train_frame,
        validation_frame=validation_frame,
        feature_columns=dataset.feature_columns,
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        dqn_config=DQNSelectorConfig(episodes=10, batch_size=32, seed=23),
        checkpoint_path=Path("artifacts/strategy_checkpoints/strategy_selector_dqn.pt"),
    )
    payload = {
        "experiment": "strategy_selector_dqn_demo",
        "purpose": "Learned discrete strategy-selector scaffold. Not a live strategy.",
        "features": list(dataset.feature_columns),
        "rows": {
            "rl_train": len(train_frame),
            "rl_validation": len(validation_frame),
        },
        "report": report.as_dict(),
        "validation_tail": validation_output[
            [
                "baseline_position",
                "high_vol_prob",
                "selected_strategy",
                "selected_multiplier",
                "selected_position",
                "selected_net_return",
                "selected_equity",
            ]
        ]
        .tail(5)
        .to_dict(orient="index"),
    }
    payload["validation_tail"] = {
        str(index.date()) if hasattr(index, "date") else str(index): values
        for index, values in payload["validation_tail"].items()
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/strategy_selector_dqn_demo.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
