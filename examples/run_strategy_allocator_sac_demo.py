from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.strategy_selector import StrategySelectorConfig
from quant_ml_lab.strategy_selector_sac import (
    SACStrategyAllocatorEnvConfig,
    train_validate_strategy_allocator_sac,
)
from quant_ml_lab.torch_sac import TorchSACConfig


def main() -> None:
    prices = make_synthetic_pair(SyntheticPairConfig(periods=820, seed=321))
    train, test = train_test_split_time(prices)
    dataset = build_hmm_rl_dataset(train, test)
    frame = dataset.frame
    rl_train_size = int(len(frame) * 0.65)
    train_frame = frame.iloc[:rl_train_size].copy()
    validation_frame = frame.iloc[rl_train_size:].copy()

    report, _, validation_output = train_validate_strategy_allocator_sac(
        train_frame=train_frame,
        validation_frame=validation_frame,
        feature_columns=dataset.feature_columns,
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        env_config=SACStrategyAllocatorEnvConfig(transaction_cost_bps=2.0),
        sac_config=TorchSACConfig(
            steps=700,
            warmup_steps=96,
            batch_size=48,
            gamma=0.95,
            hidden_dim=64,
            target_entropy=-5.0,
            seed=41,
        ),
        checkpoint_path=Path("artifacts/strategy_checkpoints/strategy_allocator_sac.pt"),
    )
    payload = {
        "experiment": "strategy_allocator_sac_demo",
        "purpose": "SAC continuous strategy-family allocator scaffold. Not a live strategy.",
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
                "sac_allocator_position",
                "sac_allocator_net_return",
                "sac_allocator_equity",
                "weight_no_trade",
                "weight_mean_reversion_full",
                "weight_mean_reversion_low_risk",
                "weight_volatility_defensive",
                "weight_cvar_defensive",
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
    Path("reports/strategy_allocator_sac_demo.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
