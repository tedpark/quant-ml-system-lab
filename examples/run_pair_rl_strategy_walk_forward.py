from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair
from quant_ml_lab.strategy import PairRLStrategyConfig, run_pair_rl_strategy_walk_forward
from quant_ml_lab.torch_sac import TorchSACConfig
from quant_ml_lab.validation import BacktestConfig
from quant_ml_lab.walk_forward import WalkForwardConfig


def main() -> None:
    prices = make_synthetic_pair(SyntheticPairConfig(periods=920, seed=321))
    report = run_pair_rl_strategy_walk_forward(
        prices,
        wf_config=WalkForwardConfig(train_size=360, test_size=160, step_size=160),
        strategy_config=PairRLStrategyConfig(
            seeds=(3, 7),
            min_validation_rows=50,
            max_validation_drawdown=-0.20,
            min_trades=3,
            require_baseline_outperformance=True,
            require_regime_response=True,
            min_abs_active_multiplier_shift=0.03,
        ),
        backtest_config=BacktestConfig(
            entry_z=1.25,
            exit_z=0.25,
            lookback=40,
            transaction_cost_bps=2.0,
        ),
        sac_config=TorchSACConfig(
            steps=320,
            warmup_steps=40,
            batch_size=32,
            hidden_dim=64,
            gamma=0.95,
            tau=0.02,
            replay_capacity=20_000,
        ),
    )
    payload = {
        "experiment": "pair_rl_strategy_walk_forward",
        "purpose": "Walk-forward validation for the public pair RL strategy candidate.",
        "disclosure": "Synthetic public pipeline only. No live execution or private alpha.",
        **report.as_dict(),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/pair_rl_strategy_walk_forward.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
