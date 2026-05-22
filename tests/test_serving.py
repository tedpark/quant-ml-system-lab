import pytest
from pydantic import ValidationError

from quant_ml_lab.serving import ModelInfo, PredictionRequest, demo_predict


def test_model_info_schema():
    info = ModelInfo(model_name="demo", model_version="0.1.0", feature_count=3)

    assert info.feature_count == 3
    assert "No production checkpoint" in info.disclosure


def test_prediction_request_rejects_empty_features():
    with pytest.raises(ValidationError):
        PredictionRequest(features=[])


def test_demo_predict_returns_bounded_confidence():
    response = demo_predict(PredictionRequest(features=[0.5, 0.25, 0.5], request_id="req-1"))

    assert response.request_id == "req-1"
    assert response.signal == "risk_on"
    assert 0.0 <= response.confidence <= 1.0
