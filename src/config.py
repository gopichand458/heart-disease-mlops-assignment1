"""Central configuration: paths, column definitions and constants."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "heart_cleveland_raw.csv"
CLEAN_PATH = ROOT / "data" / "processed" / "heart_clean.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best_model.joblib"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
MLFLOW_MODEL_DIR = MODEL_DIR / "mlflow_model"
FIG_DIR = ROOT / "reports" / "figures"
MLRUNS_DIR = ROOT / "mlruns"

UCI_URL = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/"
    "heart-disease/processed.cleveland.data"
)

# Column order of the UCI "processed.cleveland.data" file ('num' is the target)
COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target",
]
NUMERIC = ["age", "trestbps", "chol", "thalach", "oldpeak"]
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "target"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
EXPERIMENT_NAME = "heart-disease-risk"

# Probability thresholds for the human-friendly risk band returned by the API
RISK_LOW = 0.35
RISK_HIGH = 0.65
