import numpy as np
import pandas as pd
import pytest

from src.config import COLUMNS, TARGET
from src.data import clean, missing_report


def test_clean_binarises_target(raw_df, clean_df):
    assert set(clean_df[TARGET].unique()) <= {0, 1}
    assert (clean_df[TARGET] == (raw_df.loc[clean_df.index, TARGET] > 0).astype(int)).all()


def test_clean_converts_question_marks_to_nan():
    df = pd.DataFrame([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, "?", "6", 0]], columns=COLUMNS)
    out = clean(df)
    assert np.isnan(out.loc[0, "ca"])
    assert out.loc[0, "thal"] == 6.0


def test_clean_renames_num_column():
    df = pd.DataFrame([[63, 1, 1, 145, 233, 1, 2, 150, 0, 2.3, 3, 0, 6, 2]],
                      columns=COLUMNS[:-1] + ["num"])
    assert clean(df)[TARGET].iloc[0] == 1


def test_clean_drops_duplicates_and_missing_target(raw_df):
    dup = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
    dup.loc[0, TARGET] = np.nan
    out = clean(dup)
    assert not out.duplicated().any()
    assert out[TARGET].notna().all()


def test_clean_raises_on_missing_columns(raw_df):
    with pytest.raises(ValueError):
        clean(raw_df.drop(columns=["chol"]))


def test_missing_report_lists_only_columns_with_nan(clean_df):
    report = missing_report(clean_df)
    assert set(report.index) <= {"ca", "thal"}
    assert (report["missing"] > 0).all()
