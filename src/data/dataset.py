"""Raw data loading, feature engineering, and time-based splitting.

Split protocol (per recent fraud-detection literature):
train on the earliest 70% of transactions, validate on the next 15%,
test on the latest 15%. The test set keeps the original imbalanced
class distribution; resampling is never applied to val/test.
"""

import numpy as np
import pandas as pd

from src import config


def load_raw() -> pd.DataFrame:
    return pd.read_csv(config.RAW_CSV)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hour = (df["Time"] // 3600) % 24
    df["log_Amount"] = np.log1p(df["Amount"])
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    drop = {"Time", config.TARGET_COL}
    return [c for c in df.columns if c not in drop]


def time_split(df: pd.DataFrame):
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * config.TRAIN_FRAC)
    val_end = int(n * (config.TRAIN_FRAC + config.VAL_FRAC))
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def random_split(df: pd.DataFrame):
    """Stratified shuffled split - leakage-prone reference protocol used
    only for the ablation quantifying the random-vs-time gap."""
    from sklearn.model_selection import train_test_split

    y = df[config.TARGET_COL]
    tr, rest = train_test_split(df, train_size=config.TRAIN_FRAC,
                                random_state=config.RANDOM_STATE, stratify=y)
    rel = config.VAL_FRAC / (config.VAL_FRAC + (1 - config.TRAIN_FRAC - config.VAL_FRAC))
    va, te = train_test_split(rest, train_size=rel,
                              random_state=config.RANDOM_STATE,
                              stratify=rest[config.TARGET_COL])
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


def get_splits(split: str = "time"):
    """Return X_train, y_train, X_val, y_val, X_test, y_test, feature_names."""
    df = add_features(load_raw())
    feats = feature_columns(df)
    tr, va, te = time_split(df) if split == "time" else random_split(df)
    y = config.TARGET_COL
    return (
        tr[feats], tr[y],
        va[feats], va[y],
        te[feats], te[y],
        feats,
    )
