from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair
from quant_ml_lab.walk_forward import WalkForwardConfig, run_walk_forward, summarize_walk_forward


def main() -> None:
    df = make_synthetic_pair(SyntheticPairConfig(periods=520))
    folds = run_walk_forward(df, WalkForwardConfig(train_size=180, test_size=60, step_size=60))
    report = {
        "dataset": "synthetic_pair",
        "disclosure": "Synthetic data only. No production strategy parameters are included.",
        "summary": summarize_walk_forward(folds),
        "folds": [fold.as_dict() for fold in folds],
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/walk_forward.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
