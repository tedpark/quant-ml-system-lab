from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quant_ml_lab.benchmarking import benchmark_latency
from quant_ml_lab.serving import PredictionRequest, demo_predict


def main() -> None:
    request = PredictionRequest(request_id="bench", features=[0.4, 0.3, 0.2])
    report = {
        "disclosure": "Local demo latency only. No production model is benchmarked.",
        "predict_latency": benchmark_latency(lambda: demo_predict(request), iterations=1000).as_dict(),
    }
    Path("reports").mkdir(exist_ok=True)
    Path("reports/latency_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
