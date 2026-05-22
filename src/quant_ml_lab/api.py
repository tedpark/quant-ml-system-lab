from __future__ import annotations

from fastapi import FastAPI

from quant_ml_lab.data import SyntheticPairConfig, make_synthetic_pair, train_test_split_time
from quant_ml_lab.monitoring import feature_drift_report
from quant_ml_lab.serving import ModelInfo, PredictionRequest, PredictionResponse, demo_predict


def create_app() -> FastAPI:
    app = FastAPI(
        title="Quant ML System Lab API",
        version="0.1.0",
        description="Sanitized demo API. No production trading model or strategy is included.",
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "disclosure": "Demo API only. Not connected to live trading.",
        }

    @app.get("/model/info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        return ModelInfo(
            model_name="demo-risk-model",
            model_version="0.1.0",
            feature_count=3,
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(request: PredictionRequest) -> PredictionResponse:
        return demo_predict(request)

    @app.get("/metrics")
    def metrics() -> dict[str, object]:
        df = make_synthetic_pair(SyntheticPairConfig(periods=220))
        expected, actual = train_test_split_time(df)
        drift = feature_drift_report(expected, actual, ["asset_a", "asset_b"])
        return {
            "dataset": "synthetic_pair",
            "disclosure": "Demo monitoring snapshot only. Not connected to live trading.",
            "feature_drift": [metric.as_dict() for metric in drift],
        }

    return app


app = create_app()
