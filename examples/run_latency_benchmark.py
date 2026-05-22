from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from quant_ml_lab.benchmarking import benchmark_latency
from quant_ml_lab.serving import PredictionRequest, demo_predict
from report_utils import markdown_table, write_markdown


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
    row = {"scope": "demo_predict", **report["predict_latency"]}
    markdown = f"""# Serving Latency

This report measures the local deterministic demo prediction contract. It does not benchmark a production model or checkpoint.

## Protocol

- Endpoint shape: `POST /predict`
- Measurement target: local `demo_predict` function
- Iterations: {report["predict_latency"]["iterations"]}
- Production checkpoints: excluded

## Result

{markdown_table([row], ["scope", "iterations", "p50_ms", "p95_ms", "p99_ms", "max_ms"])}

## Limitations

- Local timings vary by machine and current system load.
- This is a schema/contract benchmark, not a live model benchmark.
"""
    write_markdown("docs/benchmark_reports/serving_latency.md", markdown)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
