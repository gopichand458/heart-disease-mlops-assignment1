"""Download the UCI Heart Disease (Cleveland) dataset and build the cleaned CSV.

Usage:
    python scripts/download_data.py            # download if missing, then clean
    python scripts/download_data.py --force    # re-download
    python scripts/download_data.py --from-file /path/to/processed.cleveland.data

Manual alternative: download `processed.cleveland.data` from
https://archive.ics.uci.edu/dataset/45/heart+disease and pass it with --from-file.
"""
import argparse
import shutil
import sys
import urllib.request
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import COLUMNS, RAW_PATH, UCI_URL  # noqa: E402
from src.data import build_clean_dataset  # noqa: E402


def fetch(url: str, dest: Path) -> None:
    tmp = dest.with_suffix(".tmp")
    with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as fh:
        shutil.copyfileobj(resp, fh)
    tmp.rename(dest)


def to_csv_with_header(source: Path, dest: Path) -> None:
    df = pd.read_csv(source, header=None, names=COLUMNS, na_values=["?"])
    df.to_csv(dest, index=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--from-file", default=None)
    args = parser.parse_args()

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    if RAW_PATH.exists() and not args.force and not args.from_file:
        print(f"Raw data already present: {RAW_PATH}")
    else:
        download = RAW_PATH.with_suffix(".download")
        if args.from_file:
            shutil.copy(args.from_file, download)
        else:
            print(f"Downloading {UCI_URL}")
            try:
                fetch(UCI_URL, download)
            except Exception as exc:  # network errors vary widely
                print(f"Download failed: {exc}\nUse --from-file with a manual download.")
                return 1
        to_csv_with_header(download, RAW_PATH)
        download.unlink(missing_ok=True)
        print(f"Saved raw data to {RAW_PATH}")

    df = build_clean_dataset()
    print(f"Cleaned dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
