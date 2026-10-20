import numpy as np

from src.config import FEATURES
from src.features import add_clinical_features, build_preprocessor, transformed_feature_names


def test_pct_max_hr_feature(clean_df):
    out = add_clinical_features(clean_df[FEATURES])
    expected = clean_df["thalach"] / (220 - clean_df["age"])
    assert np.allclose(out["pct_max_hr"], expected)
    assert "pct_max_hr" not in clean_df.columns  # input is not mutated


def test_preprocessor_outputs_finite_values_despite_missing(clean_df):
    prep = build_preprocessor()
    out = prep.fit_transform(clean_df[FEATURES])
    assert out.shape[0] == len(clean_df)
    assert np.isfinite(np.asarray(out)).all()


def test_preprocessor_scales_numeric_columns(clean_df):
    prep = build_preprocessor().fit(clean_df[FEATURES])
    out = np.asarray(prep.transform(clean_df[FEATURES]))
    assert abs(out[:, 0].mean()) < 1e-6  # first numeric column (age) is standardised


def test_preprocessor_ignores_unseen_categories(clean_df):
    prep = build_preprocessor().fit(clean_df[FEATURES])
    row = clean_df[FEATURES].head(1).copy()
    row["cp"] = 99
    assert np.isfinite(np.asarray(prep.transform(row))).all()


def test_feature_names_match_width(trained_pipeline, clean_df):
    names = transformed_feature_names(trained_pipeline)
    width = np.asarray(trained_pipeline.named_steps["prep"].transform(clean_df[FEATURES])).shape[1]
    assert len(names) == width
