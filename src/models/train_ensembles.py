"""Ensemble tier: deep ensemble (5-seed LSTM) and cross-family averaging
(RF + MLP + LSTM), evaluated under the same protocol.

Usage:
    python -m src.models.train_ensembles
"""

import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.dataset import get_splits
from src.models import deep
from src.models.metrics import best_threshold_f1, evaluate_scores
from src.models.train_classical import run_one as classical_run_one
from src.models.train_deep import SEQ_LEN, train_lstm, train_mlp

SEEDS = [config.RANDOM_STATE, 7, 13, 101, 2024]


def _tail_align(arrays):
    k = min(len(a) for a in arrays)
    return tuple(a[-k:] for a in arrays)


def _eval_row(model, strategy, y_va, val_s, y_te, te_s, off, elapsed):
    y_va, val_s = np.asarray(y_va)[off:], np.asarray(val_s)
    y_te, te_s = np.asarray(y_te)[off:], np.asarray(te_s)
    thr = best_threshold_f1(y_va, val_s)
    row = {"model": model, "strategy": strategy, "spw": None,
           "train_s": round(elapsed, 1)}
    row.update(evaluate_scores(y_te, te_s, thr))
    return row


def main():
    splits = get_splits()
    X_tr, y_tr, X_va, y_va, X_te, y_te, feats = splits
    scaler = StandardScaler().fit(X_tr)
    X_tr_s, X_va_s, X_te_s = (scaler.transform(a) for a in (X_tr, X_va, X_te))
    print(f"seeds={SEEDS}")

    rows = []

    t0 = time.perf_counter()
    lstm_val, lstm_te = [], []
    for s in SEEDS:
        v, t, _ = train_lstm(X_tr_s, y_tr, X_va_s, y_va, X_te_s, fast=False, seed=s)
        lstm_val.append(v)
        lstm_te.append(t)
        print(f"    lstm seed={s} done")
    ens_val = np.mean(lstm_val, axis=0)
    ens_te = np.mean(lstm_te, axis=0)
    elapsed = time.perf_counter() - t0
    rows.append(_eval_row("lstm_ensemble", "mean5", y_va, ens_val, y_te, ens_te,
                          SEQ_LEN - 1, elapsed))

    rf_row = classical_run_one("random_forest", "smote_enn", splits)
    rf_pipe = rf_row["_pipeline"]
    rf_val = rf_pipe.predict_proba(X_va)[:, 1]
    rf_te = rf_pipe.predict_proba(X_te)[:, 1]
    del rf_row["_pipeline"]

    mlp_val, mlp_te, _ = train_mlp(X_tr_s, y_tr, X_va_s, y_va, X_te_s, "raw", False)

    r_v, l_v, m_v = _tail_align([rf_val, ens_val, mlp_val])
    r_t, l_t, m_t = _tail_align([rf_te, ens_te, mlp_te])
    stack_val = (r_v + l_v + m_v) / 3.0
    stack_te = (r_t + l_t + m_t) / 3.0
    rows.append(_eval_row("rf+mlp+lstm", "stack_avg", y_va[-len(stack_val):],
                          stack_val, y_te[-len(stack_te):], stack_te, 0, np.nan))

    new = pd.DataFrame(rows)
    csv = config.RESULTS_CSV
    old = pd.read_csv(csv)
    keys = set(zip(new["model"], new["strategy"]))
    mask = ~old.apply(lambda r: (r["model"], r["strategy"]) in keys, axis=1)
    results = pd.concat([old[mask], new], ignore_index=True)
    results = results.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    results.to_csv(csv, index=False)

    cols = ["model", "strategy", "pr_auc", "roc_auc", "f1", "precision", "recall",
            "mcc", "fp", "fn", "train_s"]
    print("\n=== TEST SET RESULTS (sorted by PR-AUC) ===")
    print(results[[c for c in cols if c in results.columns]].head(12).to_string(index=False))
    print(f"\nSaved -> {csv}")


if __name__ == "__main__":
    main()
