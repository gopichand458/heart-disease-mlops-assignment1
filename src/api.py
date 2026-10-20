"""FastAPI service exposing the heart disease model.

Endpoints: POST /predict, GET /health, GET /metrics (Prometheus), GET /docs (Swagger UI).
Every request is logged as one JSON line to stdout and to $LOG_DIR/api.log.
"""
import json
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel, Field

from src import predict as predictor

logger = logging.getLogger("heart_api")


def setup_logging() -> None:
    # Re-configure on every startup so LOG_DIR changes (e.g. in tests) take effect.
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)
    log_dir = Path(os.getenv("LOG_DIR", "logs"))
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir / "api.log")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)
    except OSError:
        logger.warning("File logging disabled (cannot write to %s)", log_dir)


REQUESTS = Counter("api_requests_total", "Total HTTP requests", ["endpoint", "method", "status"])
LATENCY = Histogram("api_request_latency_seconds", "Request latency in seconds", ["endpoint"])
PREDICTIONS = Counter("predictions_total", "Predictions served", ["risk_band"])
PREDICTION_ERRORS = Counter("prediction_errors_total", "Failed prediction calls")


class PatientInput(BaseModel):
    age: float = Field(..., ge=1, le=120, description="Age in years")
    sex: int = Field(..., ge=0, le=1, description="1 = male, 0 = female")
    cp: int = Field(..., ge=1, le=4, description="Chest pain type (1-4)")
    trestbps: float = Field(..., ge=50, le=260, description="Resting blood pressure (mm Hg)")
    chol: float = Field(..., ge=50, le=700, description="Serum cholesterol (mg/dl)")
    fbs: int = Field(..., ge=0, le=1, description="Fasting blood sugar > 120 mg/dl")
    restecg: int = Field(..., ge=0, le=2, description="Resting ECG result (0-2)")
    thalach: float = Field(..., ge=40, le=250, description="Max heart rate achieved")
    exang: int = Field(..., ge=0, le=1, description="Exercise-induced angina")
    oldpeak: float = Field(..., ge=0, le=10, description="ST depression induced by exercise")
    slope: int = Field(..., ge=1, le=3, description="Slope of peak exercise ST segment")
    ca: Optional[int] = Field(None, ge=0, le=3, description="Major vessels coloured (0-3)")
    thal: Optional[Literal[3, 6, 7]] = Field(None, description="3 normal, 6 fixed, 7 reversible")

    model_config = {
        "json_schema_extra": {
            "example": {
                "age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
                "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
                "ca": 0, "thal": 6,
            }
        }
    }


class PredictionOut(BaseModel):
    prediction: int
    label: str
    confidence: float
    probability_disease: float
    risk_band: str
    model_version: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    app.state.model = None
    app.state.metadata = {}
    try:
        app.state.model = predictor.load_model()
        app.state.metadata = predictor.load_metadata()
        logger.info(json.dumps({"event": "model_loaded",
                                "version": app.state.metadata.get("version")}))
    except FileNotFoundError as exc:
        logger.error(json.dumps({"event": "model_missing", "error": str(exc)}))
    yield


app = FastAPI(
    title="Heart Disease Risk API",
    description="Predicts heart disease risk from patient health data.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_and_measure(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        REQUESTS.labels(endpoint, request.method, str(status)).inc()
        LATENCY.labels(endpoint).observe(elapsed)
        logger.info(json.dumps({
            "event": "request", "request_id": request_id, "method": request.method,
            "path": request.url.path, "status": status,
            "latency_ms": round(elapsed * 1000, 2),
            "client": request.client.host if request.client else None,
        }))


@app.get("/health")
def health():
    loaded = app.state.model is not None
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded,
            "model_version": app.state.metadata.get("version", "unknown")}


@app.post("/predict", response_model=PredictionOut)
def predict(patient: PatientInput):
    if app.state.model is None:
        PREDICTION_ERRORS.inc()
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        result = predictor.predict_records(app.state.model, [patient.model_dump()])[0]
    except Exception as exc:
        PREDICTION_ERRORS.inc()
        logger.error(json.dumps({"event": "prediction_error", "error": str(exc)}))
        raise HTTPException(status_code=500, detail="Prediction failed") from exc
    PREDICTIONS.labels(result["risk_band"]).inc()
    logger.info(json.dumps({"event": "prediction", **{k: result[k] for k in (
        "prediction", "probability_disease", "risk_band")}}))
    return {**result, "model_version": app.state.metadata.get("version", "unknown")}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
