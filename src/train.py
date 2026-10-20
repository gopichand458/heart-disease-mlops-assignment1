"""Train, tune, compare and track models with MLflow; persist the best pipeline.

Usage:
    python -m src.train                # full grids
    python -m src.train --fast         # tiny grids (CI smoke / quick check)
    mlflow ui --backend-store-uri ./mlruns   # inspect runs
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone

import joblib
import matplotlib
import mlflow
import mlflow.sklearn
import pandas as pd
import sklearn
from mlflow.models import infer_signature
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import modeling  # noqa: E402
from src.config import (  # noqa: E402
    CV_FOLDS, EXPERIMENT_NAME, FEATURES, METADATA_PATH, MLFLOW_MODEL_DIR, MLRUNS_DIR,
    MODEL_DIR, MODEL_PATH, RANDOM_STATE, TARGET, TEST_SIZE,
)
from src.data import load_clean  # noqa: E402


def git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return "unknown"


def run(fast: bool = False, tracking_uri: str = None) -> dict:
    df = load_clean()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )

    uri = tracking_uri or os.getenv("MLFLOW_TRACKING_URI") or MLRUNS_DIR.resolve().as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(EXPERIMENT_NAME)

    results, fitted = {}, {}
    with mlflow.start_run(run_name="model-comparison") as parent:
        mlflow.set_tags({"git_sha": git_sha(), "stage": "training"})
        mlflow.log_params({
            "n_rows": len(df), "n_features": len(FEATURES),
            "positive_rate": round(float(y.mean()), 4), "test_size": TEST_SIZE,
            "cv_folds": CV_FOLDS, "random_state": RANDOM_STATE, "fast_mode": fast,
            "sklearn_version": sklearn.__version__,
        })

        for name, (estimator, grid) in modeling.candidate_models(fast).items():
            with mlflow.start_run(run_name=name, nested=True):
                print(f"==> tuning {name}")
                search = modeling.tune(name, estimator, grid, X_train, y_train)
                best = search.best_estimator_
                cv = modeling.cross_validated_metrics(best, X_train, y_train)
                test = modeling.holdout_metrics(best, X_test, y_test)

                mlflow.set_tag("model_type", name)
                mlflow.log_params({k.replace("model__", ""): v
                                   for k, v in search.best_params_.items()})
                mlflow.log_metric("grid_best_cv_roc_auc", float(search.best_score_))
                mlflow.log_metrics({**cv, **test})

                mlflow.log_figure(modeling.plot_roc(best, X_test, y_test, f"ROC - {name}"),
                                  "plots/roc_curve.png")
                mlflow.log_figure(modeling.plot_confusion(best, X_test, y_test,
                                                          f"Confusion - {name}"),
                                  "plots/confusion_matrix.png")
                fig = modeling.plot_feature_importance(best, f"Top features - {name}")
                mlflow.log_figure(fig, "plots/feature_importance.png")
                plt.close("all")

                with tempfile.TemporaryDirectory() as tmp:
                    path = os.path.join(tmp, "grid_search_results.csv")
                    pd.DataFrame(search.cv_results_).to_csv(path, index=False)
                    mlflow.log_artifact(path, "tuning")

                sample = X_train.head(5)
                mlflow.sklearn.log_model(
                    best, artifact_path="model",
                    signature=infer_signature(sample, best.predict(sample)),
                    input_example=sample.head(2),
                )
                results[name] = {**cv, **test, "best_params": search.best_params_}
                fitted[name] = best

        table = pd.DataFrame(results).T
        best_name = table["cv_roc_auc_mean"].astype(float).idxmax()
        best_model = fitted[best_name]

        mlflow.log_param("selected_model", best_name)
        mlflow.log_metric("selected_cv_roc_auc", float(table.loc[best_name, "cv_roc_auc_mean"]))
        mlflow.log_metric("selected_test_roc_auc", float(table.loc[best_name, "test_roc_auc"]))
        mlflow.log_figure(modeling.plot_model_comparison(table.drop(columns=["best_params"])),
                          "plots/model_comparison.png")
        plt.close("all")

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, MODEL_PATH)
        if MLFLOW_MODEL_DIR.exists():
            shutil.rmtree(MLFLOW_MODEL_DIR)
        sample = X_train.head(5)
        mlflow.sklearn.save_model(
            best_model, str(MLFLOW_MODEL_DIR),
            signature=infer_signature(sample, best_model.predict(sample)),
            input_example=sample.head(2),
        )

        metadata = {
            "model_name": best_name,
            "version": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S"),
            "mlflow_run_id": parent.info.run_id,
            "git_sha": git_sha(),
            "sklearn_version": sklearn.__version__,
            "python_version": sys.version.split()[0],
            "features": FEATURES,
            "metrics": {k: v for k, v in results[best_name].items() if k != "best_params"},
            "best_params": {k: (v if isinstance(v, (int, float, str, type(None))) else str(v))
                            for k, v in results[best_name]["best_params"].items()},
        }
        METADATA_PATH.write_text(json.dumps(metadata, indent=2))
        mlflow.log_artifact(str(METADATA_PATH), "final_model")
        mlflow.log_artifact(str(MODEL_PATH), "final_model")

    summary = table.drop(columns=["best_params"]).astype(float).round(4)
    print("\n=== Results (cv_* = 5-fold CV on train, test_* = hold-out) ===")
    print(summary[["cv_accuracy_mean", "cv_precision_mean", "cv_recall_mean",
                   "cv_roc_auc_mean", "test_roc_auc"]].to_string())
    print(f"\nSelected model: {best_name}  (saved to {MODEL_PATH})")
    print(f"MLflow tracking URI: {uri}")
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="use small hyper-parameter grids")
    parser.add_argument("--tracking-uri", default=None)
    args = parser.parse_args()
    run(fast=args.fast, tracking_uri=args.tracking_uri)


if __name__ == "__main__":
    main()
