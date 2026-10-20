"""Pure scikit-learn modelling logic (no MLflow) so it is easy to unit test."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate  # noqa: E402

from src.config import CV_FOLDS, RANDOM_STATE
from src.features import build_model_pipeline, transformed_feature_names

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def candidate_models(fast: bool = False) -> dict:
    """name -> (estimator, param_grid). ``fast`` shrinks grids (used in tests/CI smoke)."""
    rs = RANDOM_STATE
    models = {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, random_state=rs),
            {"model__C": [0.1, 1.0] if fast else [0.01, 0.1, 1.0, 10.0],
             "model__class_weight": [None, "balanced"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=rs),
            {"model__n_estimators": [50] if fast else [100, 300],
             "model__max_depth": [3, None] if fast else [3, 5, None],
             "model__min_samples_leaf": [1] if fast else [1, 3]},
        ),
        "gradient_boosting": (
            GradientBoostingClassifier(random_state=rs),
            {"model__n_estimators": [50] if fast else [100, 200],
             "model__learning_rate": [0.1] if fast else [0.03, 0.1],
             "model__max_depth": [2] if fast else [2, 3]},
        ),
    }
    return models


def make_cv(n_splits: int = CV_FOLDS) -> StratifiedKFold:
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def tune(name: str, estimator, grid: dict, X, y, n_splits: int = CV_FOLDS) -> GridSearchCV:
    """Grid-search the full pipeline (preprocessing is re-fit inside every fold)."""
    search = GridSearchCV(
        build_model_pipeline(estimator), grid, scoring="roc_auc",
        cv=make_cv(n_splits), n_jobs=-1, refit=True,
    )
    return search.fit(X, y)


def cross_validated_metrics(pipeline, X, y, n_splits: int = CV_FOLDS) -> dict:
    """Mean/std of accuracy, precision, recall, f1 and ROC-AUC over stratified k-fold."""
    scores = cross_validate(pipeline, X, y, cv=make_cv(n_splits), scoring=SCORING, n_jobs=1)
    out = {}
    for metric in SCORING:
        vals = scores[f"test_{metric}"]
        out[f"cv_{metric}_mean"] = float(np.mean(vals))
        out[f"cv_{metric}_std"] = float(np.std(vals))
    return out


def holdout_metrics(pipeline, X_test, y_test) -> dict:
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "test_accuracy": float(accuracy_score(y_test, pred)),
        "test_precision": float(precision_score(y_test, pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, pred, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, proba)),
    }


def plot_roc(pipeline, X_test, y_test, title: str):
    fig, ax = plt.subplots(figsize=(5, 4.5))
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test, ax=ax)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="chance")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion(pipeline, X_test, y_test, title: str):
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_estimator(
        pipeline, X_test, y_test, ax=ax, display_labels=["No disease", "Disease"],
        cmap="Blues", colorbar=False,
    )
    ax.set_title(title)
    fig.tight_layout()
    return fig


def feature_importance(pipeline) -> pd.Series:
    """Importance (tree models) or |coefficient| (linear models) per transformed feature."""
    names = transformed_feature_names(pipeline)
    model = pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.abs(model.coef_[0])
    else:
        return pd.Series(dtype=float)
    return pd.Series(values, index=names).sort_values(ascending=False)


def plot_feature_importance(pipeline, title: str, top: int = 12):
    imp = feature_importance(pipeline).head(top)[::-1]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.barh(imp.index, imp.values, color="#2a6f97")
    ax.set_title(title)
    ax.set_xlabel("importance")
    fig.tight_layout()
    return fig


def plot_model_comparison(results: pd.DataFrame):
    """results: index = model name, columns include cv_*_mean metrics."""
    cols = [f"cv_{m}_mean" for m in ("accuracy", "precision", "recall", "roc_auc")]
    ax = results[cols].rename(columns=lambda c: c.replace("cv_", "").replace("_mean", "")).plot(
        kind="bar", figsize=(8, 4.5), rot=15, colormap="viridis")
    ax.set_ylim(0.5, 1.0)
    ax.set_title("Cross-validated metrics by model")
    ax.set_ylabel("score")
    fig = ax.get_figure()
    fig.tight_layout()
    return fig
