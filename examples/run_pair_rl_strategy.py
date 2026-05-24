from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair
from quant_ml_lab.strategy import PairRLStrategyConfig, build_pair_rl_strategy
from quant_ml_lab.torch_sac import TorchSACConfig
from quant_ml_lab.validation import BacktestConfig


def main() -> None:
    prices = make_synthetic_pair(SyntheticPairConfig(periods=760, seed=123))
    report, validation_output = build_pair_rl_strategy(
        prices,
        config=PairRLStrategyConfig(
            seeds=(3, 7, 11),
            min_validation_rows=50,
            max_validation_drawdown=-0.20,
            min_trades=3,
            require_baseline_outperformance=True,
        ),
        backtest_config=BacktestConfig(
            entry_z=1.25,
            exit_z=0.25,
            lookback=40,
            transaction_cost_bps=2.0,
        ),
        sac_config=TorchSACConfig(
            steps=540,
            warmup_steps=48,
            batch_size=32,
            hidden_dim=64,
            gamma=0.95,
            tau=0.02,
            replay_capacity=20_000,
        ),
    )
    payload = report.as_dict()
    validation_tail = validation_output[
        [
            "baseline_position",
            "sac_sized_position",
            "sac_net_return",
            "sac_equity",
            "high_vol_prob",
        ]
    ].tail(5)
    payload["validation_tail"] = {
        str(index.date()) if hasattr(index, "date") else str(index): values
        for index, values in validation_tail.to_dict(orient="index").items()
    }

    Path("reports").mkdir(exist_ok=True)
    Path("reports/pair_rl_strategy.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
