from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.hmm_rl import build_hmm_rl_dataset
from quant_ml_lab.torch_sac import TorchSACConfig
from quant_ml_lab.torch_sac_sizing import train_hmm_sac_sizer
from quant_ml_lab.validation import backtest_pair_baseline


def main() -> None:
    df = make_synthetic_pair(SyntheticPairConfig(periods=420))
    train, test = train_test_split_time(df)
    dataset = build_hmm_rl_dataset(train, test)
    baseline, baseline_metrics = backtest_pair_baseline(test)
    report, _ = train_hmm_sac_sizer(
        dataset.frame,
        dataset.feature_columns,
        TorchSACConfig(steps=220, warmup_steps=32, batch_size=32, hidden_dim=32, gamma=0.95),
    )
    payload = {
        "experiment": "forward_hmm_sac_position_sizing",
        "purpose": "Sanitized HMM-regime + PyTorch SAC sizing demo. Not a trading strategy.",
        "disclosure": dataset.disclosure,
        "features": list(dataset.feature_columns),
        "baseline": baseline_metrics.as_dict(),
        "hmm_sac_sizer": report.as_dict(),
        "rows": len(baseline),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/hmm_sac_sizing.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
