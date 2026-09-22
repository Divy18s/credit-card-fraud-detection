"""Train deep-learning models (MLP, AutoEncoder, LSTM) under the same
protocol as the classical tier: time-based split, resampling/weights on
train only, threshold tuned on validation, metrics on untouched test.

Usage:
    python -m src.models.train_deep [--fast]
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.dataset import get_splits
from src.models import deep
from src.models.metrics import best_threshold_f1, evaluate_scores

SEQ_LEN = 8


def train_mlp(X_tr, y_tr, X_va, y_va, X_te, strategy: str, fast: bool,
              arch: str = "mlp") -> tuple[np.ndarray, np.ndarray, float]:
    if strategy == "smote":
        from imblearn.over_sampling import SMOTE

        X_tr, y_tr = SMOTE(sampling_strategy=1.0, random_state=config.RANDOM_STATE).fit_resample(X_tr, y_tr)
        pos_weight = None
    elif strategy == "gan":
        from src.models.gan import oversample_with_gan

        X_tr, y_tr = oversample_with_gan(np.asarray(X_tr), np.asarray(y_tr))
        pos_weight = None
    elif strategy == "class_weight":
        pos_weight = (y_tr == 0).sum() / max(y_tr.sum(), 1)
    else:
        pos_weight = None

    epochs = 3 if fast else 40
    model = deep.DNN(X_tr.shape[1]) if arch == "dnn" else deep.MLP(X_tr.shape[1])
    t0 = time.perf_counter()
    model = deep.train_classifier(model, X_tr, np.asarray(y_tr, dtype=np.float32),
                                  X_va, np.asarray(y_va, dtype=np.float32),
                                  pos_weight=pos_weight, epochs=epochs)
    elapsed = time.perf_counter() - t0
    return (deep._sigmoid(deep.predict_logits(model, X_va)),
            deep._sigmoid(deep.predict_logits(model, X_te)),
            elapsed)


def train_autoencoder(X_tr, y_tr, X_va, y_va, X_te, fast: bool):
    X_legit = X_tr[np.asarray(y_tr == 0)]
    model = deep.AutoEncoder(X_tr.shape[1])
    t0 = time.perf_counter()
    model = deep.train_autoencoder(model, X_legit, X_va, np.asarray(y_va, dtype=np.float32),
                                   epochs=2 if fast else 50)
    elapsed = time.perf_counter() - t0
    return (deep.reconstruction_error(model, X_va),
            deep.reconstruction_error(model, X_te),
            elapsed)


def _resample_sequences(X_seq: np.ndarray, y_seq: np.ndarray, method: str):
    """Imbalance handling at the sequence level (leakage-safe analog of the
    feature-level SMOTE-ENN used in LSTM fraud papers)."""
    n, seq_len, n_feat = X_seq.shape
    flat = X_seq.reshape(n, seq_len * n_feat)
    if method == "smote_enn":
        from imblearn.combine import SMOTEENN
        from imblearn.over_sampling import SMOTE

        sampler = SMOTEENN(
            smote=SMOTE(sampling_strategy=1.0, random_state=config.RANDOM_STATE, k_neighbors=4),
            random_state=config.RANDOM_STATE,
        )
    elif method == "ros":
        from imblearn.over_sampling import RandomOverSampler

        sampler = RandomOverSampler(sampling_strategy=1.0, random_state=config.RANDOM_STATE)
    else:
        return X_seq, y_seq
    flat_res, y_res = sampler.fit_resample(flat, y_seq)
    order = np.random.RandomState(config.RANDOM_STATE).permutation(len(y_res))
    return (flat_res[order].reshape(-1, seq_len, n_feat).astype(np.float32),
            y_res[order].astype(np.float32))


def train_lstm(X_tr, y_tr, X_va, y_va, X_te, fast: bool,
               bidirectional: bool = False, seq_resample: str | None = None,
               seed: int | None = None):
    y_tr = np.asarray(y_tr, dtype=np.float32)
    y_va = np.asarray(y_va, dtype=np.float32)
    Xtr_s, ytr_s = deep.make_sequences(X_tr, y_tr, SEQ_LEN)
    if seq_resample:
        if seq_resample == "gan":
            from src.models.gan import oversample_with_gan

            Xtr_s, ytr_s = oversample_with_gan(Xtr_s, ytr_s, seq_mode=True)
        else:
            Xtr_s, ytr_s = _resample_sequences(Xtr_s, ytr_s, seq_resample)
    Xva_s, yva_s = deep.make_sequences(X_va, y_va, SEQ_LEN)
    Xte_s, _ = deep.make_sequences(X_te, None, SEQ_LEN)

    pos_weight = None if seq_resample else ((y_tr == 0).sum() / max(y_tr.sum(), 1)).item()
    model = deep.LSTMClassifier(X_tr.shape[1], bidirectional=bidirectional)
    t0 = time.perf_counter()
    model = deep.train_classifier(model, Xtr_s, ytr_s, Xva_s, yva_s,
                                  pos_weight=pos_weight,
                                  epochs=2 if fast else 15,
                                  patience=3 if not fast else 1,
                                  batch=1024,
                                  seed=seed if seed is not None else config.RANDOM_STATE)
    elapsed = time.perf_counter() - t0

    val_scores = deep._sigmoid(deep.predict_logits(model, Xva_s, batch=2048))
    te_scores = deep._sigmoid(deep.predict_logits(model, Xte_s, batch=2048))
    return val_scores, te_scores, elapsed


def _pos_weight(y) -> float:
    y = np.asarray(y)
    return round(float((y == 0).sum() / max(int(y.sum()), 1)), 1)


def run_one(job: str, strategy: str, splits, scaled, fast: bool) -> dict:
    X_tr_s, X_va_s, X_te_s = scaled
    y_tr, y_va, y_te = splits[1], splits[3], splits[5]

    if job in ("mlp", "dnn"):
        val_s, te_s, elapsed = train_mlp(X_tr_s, y_tr, X_va_s, y_va, X_te_s, strategy, fast,
                                         arch=job)
        spw = _pos_weight(y_tr) if strategy == "class_weight" else None
    elif job == "autoencoder":
        val_s, te_s, elapsed = train_autoencoder(X_tr_s, y_tr, X_va_s, y_va, X_te_s, fast)
        spw = None
    elif job == "fttransformer":
        pos_weight = _pos_weight(y_tr) if strategy == "class_weight" else None
        model = deep.FTTransformer(X_tr_s.shape[1])
        t0 = time.perf_counter()
        model = deep.train_classifier(model, X_tr_s, np.asarray(y_tr, dtype=np.float32),
                                      X_va_s, np.asarray(y_va, dtype=np.float32),
                                      pos_weight=pos_weight,
                                      epochs=2 if fast else 30, patience=5)
        elapsed = time.perf_counter() - t0
        val_s = deep._sigmoid(deep.predict_logits(model, X_va_s))
        te_s = deep._sigmoid(deep.predict_logits(model, X_te_s))
        spw = _pos_weight(y_tr) if strategy == "class_weight" else None
        thr = best_threshold_f1(y_va, val_s)
        row = {"model": job, "strategy": strategy, "spw": spw, "train_s": round(elapsed, 1)}
        row.update(evaluate_scores(y_te, te_s, thr))
        return row
    else:
        val_s, te_s, elapsed = train_lstm(X_tr_s, y_tr, X_va_s, y_va, X_te_s, fast,
                                          bidirectional=(job == "bilstm"),
                                          seq_resample={"smote_enn": "smote_enn",
                                                        "gan": "gan"}.get(strategy))
        spw = None if strategy == "smote_enn" else _pos_weight(y_tr)

    if job in ("lstm", "bilstm"):
        off = SEQ_LEN - 1
        thr = best_threshold_f1(np.asarray(y_va)[off:], val_s)
        row_extra = evaluate_scores(np.asarray(y_te)[off:], te_s, thr)
    else:
        thr = best_threshold_f1(y_va, val_s)
        row_extra = evaluate_scores(y_te, te_s, thr)

    row = {"model": job, "strategy": strategy, "spw": spw, "train_s": round(elapsed, 1)}
    row.update(row_extra)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true", help="tiny epoch counts for smoke-testing")
    parser.add_argument("--jobs", default=None, help="comma-separated model names, e.g. lstm,bilstm")
    args = parser.parse_args()

    jobs = [
        ("mlp", "raw"),
        ("mlp", "class_weight"),
        ("mlp", "smote"),
        ("mlp", "gan"),
        ("dnn", "raw"),
        ("autoencoder", "anomaly"),
        ("lstm", "class_weight"),
        ("lstm", "smote_enn"),
        ("lstm", "gan"),
        ("bilstm", "class_weight"),
        ("fttransformer", "class_weight"),
    ]
    if args.jobs:
        keep = set(args.jobs.split(","))
        jobs = [j for j in jobs if j[0] in keep]

    splits = get_splits()
    X_tr, y_tr, X_va, y_va, X_te, y_te, feats = splits
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_va_s = scaler.transform(X_va)
    X_te_s = scaler.transform(X_te)
    print(f"device={deep.DEVICE} | features={len(feats)} | "
          f"train {len(y_tr)} (fraud {int(y_tr.sum())}) | "
          f"val frauds {int(y_va.sum())} | test frauds {int(y_te.sum())}")

    rows = []
    for job, strat in jobs:
        tag = f"{job:>12} + {strat:<12}"
        try:
            row = run_one(job, strat, splits, (X_tr_s, X_va_s, X_te_s), args.fast)
            rows.append(row)
            print(f"{tag} PR-AUC={row['pr_auc']:.4f} F1={row['f1']:.4f} MCC={row['mcc']:.4f} ({row['train_s']}s)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{tag} FAILED: {type(e).__name__}: {e}")

    new = pd.DataFrame(rows)
    csv = config.RESULTS_CSV
    if csv.exists():
        old = pd.read_csv(csv)
        keys = set(zip(new["model"], new["strategy"]))
        mask = ~old.apply(lambda r: (r["model"], r["strategy"]) in keys, axis=1)
        results = pd.concat([old[mask], new], ignore_index=True)
    else:
        results = new
    results = results.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(csv, index=False)

    cols = ["model", "strategy", "spw", "pr_auc", "roc_auc", "f1", "precision", "recall", "mcc", "fp", "fn", "train_s"]
    print("\n=== TEST SET RESULTS (sorted by PR-AUC) ===")
    print(results[[c for c in cols if c in results.columns]].to_string(index=False))
    print(f"\nSaved -> {csv}")


if __name__ == "__main__":
    main()
