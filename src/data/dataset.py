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


def sang_split(df: pd.DataFrame):
    """Chronological 66/14/20 split matching Sang (2026, EAI ISMLA).

    On the raw (non-deduplicated) time-sorted CSV these fractions reproduce that
    paper's partition exactly: 187,972 / 39,873 / 56,962 rows with 368 / 49 / 75
    frauds. Used only for the pre-registered replication check; the project's
    primary protocol remains the 70/15/15 split above.
    """
    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)
    train_end = int(n * 0.66)
    val_end = int(n * 0.80)
    return df.iloc[:train_end], df.iloc[train_end:val_end], df.iloc[val_end:]


def sang_random_split(df: pd.DataFrame):
    """Stratified SHUFFLED 66/14/20 - Sang (2026)'s random-split protocol.

    Same fractions as sang_split() but shuffled rather than chronological, which
    reproduces that paper's random-split test partition exactly: 98 fraud cases
    (325 / 69 / 98 across train / val / test). Pairing this with sang_split()
    isolates the protocol effect at fixed fractions, which is the comparison the
    paper itself reports as 0.8815 against 0.7902.
    """
    from sklearn.model_selection import train_test_split

    y = df[config.TARGET_COL]
    tr, rest = train_test_split(df, train_size=0.66,
                                random_state=config.RANDOM_STATE, stratify=y)
    va, te = train_test_split(rest, train_size=0.14 / 0.34,
                              random_state=config.RANDOM_STATE,
                              stratify=rest[config.TARGET_COL])
    return tr.reset_index(drop=True), va.reset_index(drop=True), te.reset_index(drop=True)


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
    """Return X_train, y_train, X_val, y_val, X_test, y_test, feature_names.

    split: "time" (primary protocol, 70/15/15), "sang" (66/14/20 chronological
    replication check), "sang_random" (66/14/20 shuffled, the same paper's random
    protocol), or "random" (shuffled 70/15/15 reference for the leakage ablation).
    """
    df = add_features(load_raw())
    feats = feature_columns(df)
    if split == "time":
        tr, va, te = time_split(df)
    elif split == "sang":
        tr, va, te = sang_split(df)
    elif split == "sang_random":
        tr, va, te = sang_random_split(df)
    else:
        tr, va, te = random_split(df)
    y = config.TARGET_COL
    return (
        tr[feats], tr[y],
        va[feats], va[y],
        te[feats], te[y],
        feats,
    )
