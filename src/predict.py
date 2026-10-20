"""Inference helpers shared by the API and offline scripts."""
import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from src.config import FEATURES, METADATA_PATH, MODEL_PATH, RISK_HIGH, RISK_LOW


def model_path() -> Path:
    return Path(os.getenv("MODEL_PATH", str(MODEL_PATH)))


def load_model(path=None):
    path = Path(path) if path else model_path()
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}. Run `python -m src.train`.")
    return joblib.load(path)


def load_metadata(path=None) -> dict:
    path = Path(path) if path else Path(os.getenv("METADATA_PATH", str(METADATA_PATH)))
    if path.exists():
        return json.loads(path.read_text())
    return {"version": "unknown", "model_name": "unknown"}


def risk_band(probability: float) -> str:
    if probability < RISK_LOW:
        return "low"
    if probability < RISK_HIGH:
        return "moderate"
    return "high"


def predict_records(model, records: List[Dict]) -> List[Dict]:
    """Predict for a list of patient dicts. Missing ca/thal are imputed by the pipeline."""
    frame = pd.DataFrame(records).reindex(columns=FEATURES).astype(float)
    proba = model.predict_proba(frame)[:, 1]
    results = []
    for p in proba:
        p = float(p)
        pred = int(p >= 0.5)
        results.append({
            "prediction": pred,
            "label": "heart disease likely" if pred else "heart disease unlikely",
            "confidence": round(p if pred else 1 - p, 4),
            "probability_disease": round(p, 4),
            "risk_band": risk_band(p),
        })
    return results
