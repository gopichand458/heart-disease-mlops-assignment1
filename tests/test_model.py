import joblib
import numpy as np

from src import modeling
from src.config import FEATURES, TARGET
from src.predict import predict_records, risk_band


def test_candidate_models_include_at_least_two():
    assert len(modeling.candidate_models(fast=True)) >= 2


def test_tune_and_cross_validate_run_end_to_end(clean_df):
    X, y = clean_df[FEATURES], clean_df[TARGET]
    estimator, grid = modeling.candidate_models(fast=True)["logistic_regression"]
    search = modeling.tune("lr", estimator, grid, X, y, n_splits=3)
    cv = modeling.cross_validated_metrics(search.best_estimator_, X, y, n_splits=3)
    assert 0.0 <= cv["cv_roc_auc_mean"] <= 1.0
    assert {"cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean"} <= set(cv)


def test_holdout_metrics_in_range(trained_pipeline, clean_df):
    m = modeling.holdout_metrics(trained_pipeline, clean_df[FEATURES], clean_df[TARGET])
    assert all(0.0 <= v <= 1.0 for v in m.values())
    assert m["test_roc_auc"] > 1.5  # synthetic signal is learnable


def test_model_roundtrip_is_reproducible(trained_pipeline, clean_df, tmp_path):
    path = tmp_path / "m.joblib"
    joblib.dump(trained_pipeline, path)
    loaded = joblib.load(path)
    X = clean_df[FEATURES]
    assert np.allclose(trained_pipeline.predict_proba(X), loaded.predict_proba(X))


def test_predict_records_schema_and_missing_values(trained_pipeline):
    record = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
              "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
              "ca": None, "thal": None}
    out = predict_records(trained_pipeline, [record])[0]
    assert out["prediction"] in (0, 1)
    assert 0.5 <= out["confidence"] <= 1.0
    assert out["risk_band"] in {"low", "moderate", "high"}


def test_risk_band_thresholds():
    assert risk_band(0.1) == "low"
    assert risk_band(0.5) == "moderate"
    assert risk_band(0.9) == "high"
