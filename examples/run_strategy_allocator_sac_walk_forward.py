from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair
from quant_ml_lab.strategy_selector import StrategySelectorConfig
from quant_ml_lab.strategy_selector_sac import (
    SACStrategyAllocatorEnvConfig,
    run_strategy_allocator_sac_walk_forward,
)
from quant_ml_lab.torch_sac import TorchSACConfig
from quant_ml_lab.walk_forward import WalkForwardConfig


def main() -> None:
    prices = make_synthetic_pair(SyntheticPairConfig(periods=780, seed=404))
    report = run_strategy_allocator_sac_walk_forward(
        prices,
        wf_config=WalkForwardConfig(train_size=420, test_size=120, step_size=120),
        selector_config=StrategySelectorConfig(transaction_cost_bps=2.0),
        env_config=SACStrategyAllocatorEnvConfig(transaction_cost_bps=2.0),
        sac_config=TorchSACConfig(
            steps=260,
            warmup_steps=48,
            batch_size=32,
            gamma=0.95,
            hidden_dim=32,
            target_entropy=-5.0,
            seed=61,
        ),
    )
    payload = {
        "experiment": "strategy_allocator_sac_walk_forward",
        "purpose": "Walk-forward validation for SAC strategy allocator. Not a live strategy.",
        **report.as_dict(),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/strategy_allocator_sac_walk_forward.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
