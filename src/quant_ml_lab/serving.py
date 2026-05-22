from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ModelInfo(BaseModel):
    model_name: str
    model_version: str
    feature_count: int = Field(gt=0)
    disclosure: str = "Public demo schema. No production checkpoint is included."


class PredictionRequest(BaseModel):
    model_name: str = "demo-risk-model"
    features: list[float]
    request_id: str | None = None

    @field_validator("features")
    @classmethod
    def features_must_not_be_empty(cls, value: list[float]) -> list[float]:
        if not value:
            raise ValueError("features must not be empty")
        return value


class PredictionResponse(BaseModel):
    request_id: str | None = None
    model_name: str
    signal: Literal["risk_off", "neutral", "risk_on"]
    confidence: float = Field(ge=0.0, le=1.0)
    disclosure: str = "Synthetic/demo response only. Not investment advice."


def demo_predict(request: PredictionRequest) -> PredictionResponse:
    """Deterministic schema demo that avoids exposing a production model."""
    score = sum(request.features) / len(request.features)
    if score > 0.25:
        signal = "risk_on"
    elif score < -0.25:
        signal = "risk_off"
    else:
        signal = "neutral"
    confidence = min(1.0, abs(score))
    return PredictionResponse(
        request_id=request.request_id,
        model_name=request.model_name,
        signal=signal,
        confidence=confidence,
    )
