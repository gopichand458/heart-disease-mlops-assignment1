"""Data loading and cleaning for the Heart Disease (Cleveland) dataset."""
import argparse
from pathlib import Path

import pandas as pd

from src.config import CLEAN_PATH, COLUMNS, RAW_PATH, TARGET


def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Read the raw CSV. The UCI file marks missing values with '?'."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/download_data.py` first."
        )
    return pd.read_csv(path, na_values=["?"])


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy of the raw data.

    * normalises column names (and renames UCI's ``num`` column to ``target``)
    * coerces every column to numeric ('?' -> NaN)
    * binarises the target: 0 = no disease, 1..4 = disease present
    * drops duplicate rows and rows without a target

    Missing feature values (``ca`` and ``thal``) are deliberately *kept* as NaN.
    They are imputed inside the sklearn Pipeline so that imputation statistics
    are learned on training folds only (no data leakage).
    """
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    if "num" in df.columns and TARGET not in df.columns:
        df = df.rename(columns={"num": TARGET})
    missing_cols = set(COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing expected columns: {sorted(missing_cols)}")
    df = df[COLUMNS]
    for col in COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=[TARGET]).drop_duplicates().reset_index(drop=True)
    df[TARGET] = (df[TARGET] > 0).astype(int)
    return df


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Count and percentage of missing values per column (only columns with NaN)."""
    counts = df.isna().sum()
    report = pd.DataFrame({"missing": counts, "percent": (counts / len(df) * 100).round(2)})
    return report[report["missing"] > 0]


def build_clean_dataset(raw_path: Path = RAW_PATH, out_path: Path = CLEAN_PATH) -> pd.DataFrame:
    df = clean(load_raw(raw_path))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def load_clean(path: Path = CLEAN_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return build_clean_dataset(out_path=path)
    return pd.read_csv(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the raw heart disease CSV")
    parser.add_argument("--raw", default=str(RAW_PATH))
    parser.add_argument("--out", default=str(CLEAN_PATH))
    args = parser.parse_args()
    data = build_clean_dataset(Path(args.raw), Path(args.out))
    print(f"Wrote {len(data)} rows to {args.out}")
    print("Class balance:\n", data[TARGET].value_counts().to_string())
    print("Missing values:\n", missing_report(data).to_string())
