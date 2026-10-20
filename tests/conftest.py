"""Shared fixtures: a small synthetic dataset with the same schema as the real one."""
import numpy as np
import pandas as pd
import pytest

from src.config import COLUMNS


def make_synthetic(n: int = 160, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    age = rng.integers(30, 78, n)
    thalach = np.clip(220 - age - rng.normal(25, 20, n), 70, 210).round()
    oldpeak = np.clip(rng.gamma(1.2, 1.0, n), 0, 6).round(1)
    cp = rng.integers(1, 5, n)
    risk = 0.03 * (age - 55) + 0.8 * oldpeak + 0.6 * (cp == 4) - 0.02 * (thalach - 140)
    target_raw = (risk + rng.normal(0, 0.8, n) > 0.3).astype(int) * rng.integers(1, 5, n)
    df = pd.DataFrame({
        "age": age, "sex": rng.integers(0, 2, n), "cp": cp,
        "trestbps": rng.normal(131, 17, n).round(), "chol": rng.normal(246, 50, n).round(),
        "fbs": rng.integers(0, 2, n), "restecg": rng.integers(0, 3, n), "thalach": thalach,
        "exang": rng.integers(0, 2, n), "oldpeak": oldpeak, "slope": rng.integers(1, 4, n),
        "ca": rng.integers(0, 4, n).astype(float),
        "thal": rng.choice([3.0, 6.0, 7.0], n), "target": target_raw,
    })[COLUMNS]
    df.loc[[3, 17], "ca"] = np.nan
    df.loc[[5], "thal"] = np.nan
    return df


@pytest.fixture(scope="session")
def raw_df():
    return make_synthetic()


@pytest.fixture(scope="session")
def clean_df(raw_df):
    from src.data import clean
    return clean(raw_df)


@pytest.fixture(scope="session")
def trained_pipeline(clean_df):
    from src.config import FEATURES, TARGET
    from src.features import build_model_pipeline
    from sklearn.linear_model import LogisticRegression
    pipe = build_model_pipeline(LogisticRegression(max_iter=1000))
    return pipe.fit(clean_df[FEATURES], clean_df[TARGET])
