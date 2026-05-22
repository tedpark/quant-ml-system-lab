from __future__ import annotations

from fastapi import FastAPI

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

    return app


app = create_app()
