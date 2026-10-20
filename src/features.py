"""Feature engineering and the reusable preprocessing pipeline."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import CATEGORICAL, NUMERIC

ENGINEERED = ["pct_max_hr"]
NUMERIC_ALL = NUMERIC + ENGINEERED


def add_clinical_features(X: pd.DataFrame) -> pd.DataFrame:
    """Add domain features.

    ``pct_max_hr``: achieved maximum heart rate as a fraction of the age-predicted
    maximum (220 - age). Low values during exercise are a known cardiac risk marker.
    """
    X = X.copy()
    predicted_max = (220 - X["age"]).clip(lower=1)
    X["pct_max_hr"] = X["thalach"] / predicted_max
    return X


def build_preprocessor() -> Pipeline:
    """Imputation + scaling for numerics, imputation + one-hot for categoricals."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    columns = ColumnTransformer([
        ("num", numeric, NUMERIC_ALL),
        ("cat", categorical, CATEGORICAL),
    ])
    return Pipeline([
        ("clinical", FunctionTransformer(add_clinical_features)),
        ("columns", columns),
    ])


def build_model_pipeline(estimator) -> Pipeline:
    """Full pipeline: preprocessing + estimator (this is what gets deployed)."""
    return Pipeline([("prep", build_preprocessor()), ("model", estimator)])


def transformed_feature_names(pipeline: Pipeline):
    """Names of the columns produced by the preprocessing step of a fitted pipeline."""
    return list(pipeline.named_steps["prep"].named_steps["columns"].get_feature_names_out())
