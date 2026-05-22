from fastapi.testclient import TestClient

from quant_ml_lab.api import create_app


def test_health_endpoint():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint():
    client = TestClient(create_app())

    response = client.get("/model/info")

    assert response.status_code == 200
    body = response.json()
    assert body["model_name"] == "demo-risk-model"
    assert body["feature_count"] == 3
    assert "No production checkpoint" in body["disclosure"]


def test_predict_endpoint():
    client = TestClient(create_app())

    response = client.post(
        "/predict",
        json={"request_id": "req-1", "features": [0.4, 0.3, 0.2]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "req-1"
    assert body["signal"] == "risk_on"
    assert 0.0 <= body["confidence"] <= 1.0


def test_predict_rejects_empty_features():
    client = TestClient(create_app())

    response = client.post("/predict", json={"features": []})

    assert response.status_code == 422
