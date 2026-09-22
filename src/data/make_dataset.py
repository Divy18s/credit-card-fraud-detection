"""Download and verify the Kaggle ULB credit card fraud dataset.

Usage:
    python src/data/make_dataset.py

Requires Kaggle API credentials at ~/.kaggle/kaggle.json
(https://www.kaggle.com/settings -> API -> Create New Token).
"""

import sys

from src import config


def download() -> None:
    csv_path = config.RAW_CSV
    if csv_path.exists():
        print(f"[skip] {csv_path} already exists.")
        return

    config.DATA_RAW.mkdir(parents=True, exist_ok=True)

    try:
        import kaggle  # noqa: F401
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
    except Exception:
        sys.exit(
            "Kaggle credentials not found.\n"
            "Option A (new tokens): save your KGAT token as plain text to\n"
            "  ~/.kaggle/access_token   (Windows: C:/Users/<you>/.kaggle/access_token)\n"
            "Option B (classic): kaggle.com/settings -> API -> Create New Token,\n"
            "  save kaggle.json to ~/.kaggle/kaggle.json\n"
            "Then re-run this script."
        )
    print(f"Downloading {config.KAGGLE_DATASET} ...")
    api.dataset_download_files(config.KAGGLE_DATASET, path=str(config.DATA_RAW), unzip=True)
    print(f"Saved to {csv_path}")


def verify() -> None:
    import pandas as pd

    df = pd.read_csv(config.RAW_CSV)
    counts = df[config.TARGET_COL].value_counts()
    print(f"Shape: {df.shape}")
    print(f"Legit: {counts.get(0, 0)} | Fraud: {counts.get(1, 0)}")
    fraud_pct = 100 * counts.get(1, 0) / len(df)
    print(f"Fraud percentage: {fraud_pct:.3f}%")
    assert df.shape == (284807, 31), "Unexpected dataset shape!"


if __name__ == "__main__":
    download()
    verify()
