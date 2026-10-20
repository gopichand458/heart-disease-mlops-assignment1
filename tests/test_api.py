import joblib
import pytest
from fastapi.testclient import TestClient

VALID = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
         "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
         "ca": 0, "thal": 6}


@pytest.fixture()
def client(trained_pipeline, tmp_path, monkeypatch):
    model_file = tmp_path / "model.joblib"
    joblib.dump(trained_pipeline, model_file)
    monkeypatch.setenv("MODEL_PATH", str(model_file))
    monkeypatch.setenv("METADATA_PATH", str(tmp_path / "missing.json"))
    monkeypatch.setenv("LOG_DIR", str(tmp_path / "logs"))
    from src.api import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_predict_returns_prediction_and_confidence(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert body["risk_band"] in {"low", "moderate", "high"}


def test_predict_accepts_missing_optional_fields(client):
    payload = {k: v for k, v in VALID.items() if k not in ("ca", "thal")}
    assert client.post("/predict", json=payload).status_code == 200


def test_predict_rejects_invalid_input(client):
    bad = dict(VALID, age=500)
    assert client.post("/predict", json=bad).status_code == 422
    incomplete = {k: v for k, v in VALID.items() if k != "chol"}
    assert client.post("/predict", json=incomplete).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/predict", json=VALID)
    text = client.get("/metrics").text
    assert "api_requests_total" in text and "predictions_total" in text


def test_requests_are_logged_to_file(client, tmp_path):
    client.get("/health")
    log_file = tmp_path / "logs" / "api.log"
    assert log_file.exists() and "/health" in log_file.read_text()
