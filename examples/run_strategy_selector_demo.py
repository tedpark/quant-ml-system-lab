from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.strategy_selector import StrategySelectorConfig, run_strategy_selector
from quant_ml_lab.validation import compute_metrics


def main() -> None:
    transaction_cost_bps = 2.0
    prices = make_synthetic_pair(SyntheticPairConfig(periods=760, seed=123))
    train, test = train_test_split_time(prices)
    dataset = build_hmm_rl_dataset(train, test)
    result, report = run_strategy_selector(
        dataset.frame,
        StrategySelectorConfig(
            high_vol_threshold=0.5,
            defensive_multiplier=0.35,
            low_risk_multiplier=0.5,
            transaction_cost_bps=transaction_cost_bps,
        ),
    )
    baseline_turnover = result["baseline_position"].diff().abs().fillna(
        result["baseline_position"].abs()
    )
    baseline_returns = (result["baseline_position"] * -result["spread_return_next"]) - (
        baseline_turnover * (transaction_cost_bps / 10_000.0)
    )
    payload = {
        "experiment": "strategy_selector_demo",
        "purpose": "Regime-aware strategy family selector scaffold. Not a live strategy.",
        "features": list(dataset.feature_columns),
        "baseline_metrics": compute_metrics(baseline_returns, baseline_turnover).as_dict(),
        "selector": report.as_dict(),
        "selected_tail": result[
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
    payload["selected_tail"] = {
        str(index.date()) if hasattr(index, "date") else str(index): values
        for index, values in payload["selected_tail"].items()
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/strategy_selector_demo.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
