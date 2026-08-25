"""Random-split vs time-split ablation.

Reruns a representative config from every modeling tier under BOTH protocols
and writes the side-by-side gap to reports/results_random_split.csv.

The random (stratified, shuffled) split is the leakage-prone protocol used by
most older papers; the time split is the deployment-honest default of this
project. Everything else stays identical.

Usage:
    python -m src.models.ablation_random_split
"""

import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.dataset import get_splits
from src.models.metrics import best_threshold_f1, evaluate_scores
from src.models.train_classical import run_one as classical_run_one

SEQ_LEN = 8
CONFIGS = [
    ("random_forest", "smote_enn", "classical"),
    ("xgboost", "smote_enn", "classical"),
    ("catboost", "gan", "classical"),
    ("mlp", "raw", "deep"),
    ("lstm", "class_weight", "deep"),
    ("graphsage", "class_weight", "gnn"),
]


def eval_scores_row(y_va, val_s, y_te, te_s, off=0):
    thr = best_threshold_f1(np.asarray(y_va)[off:], np.asarray(val_s))
    return evaluate_scores(np.asarray(y_te)[off:], np.asarray(te_s), thr)


def run_deep(job, strategy, splits, fast=False):
    X_tr, y_tr, X_va, y_va, X_te, y_te = splits[:6]
    scaler = StandardScaler().fit(X_tr)
    Xs = [scaler.transform(a).astype(np.float32) for a in (X_tr, X_va, X_te)]
    t0 = time.perf_counter()
    if job == "mlp":
        from src.models.train_deep import train_mlp

        val_s, te_s, _ = train_mlp(Xs[0], y_tr, Xs[1], y_va, Xs[2], strategy, fast)
    else:
        from src.models.train_deep import train_lstm

        val_s, te_s, _ = train_lstm(Xs[0], y_tr, Xs[1], y_va, Xs[2], fast)
    elapsed = time.perf_counter() - t0
    off = SEQ_LEN - 1 if job == "lstm" else 0
    row = eval_scores_row(y_va, val_s, y_te, te_s, off)
    row.update({"train_s": round(elapsed, 1)})
    return row


def run_gnn(strategy, splits, k=10):
    from src.models.gnn import (
        SageNet, _sigmoid, build_knn_graph, predict_logits, train_graph_model,
    )

    X_tr, y_tr, X_va, y_va, X_te, y_te = splits[:6]
    scaler = StandardScaler().fit(X_tr)
    Xs = [scaler.transform(a).astype(np.float32) for a in (X_tr, X_va, X_te)]
    graphs = [(torch.as_tensor(a), build_knn_graph(a, k=k)) for a in Xs]
    ys = [torch.as_tensor(np.asarray(s, dtype=np.float32)) for s in (y_tr, y_va, y_te)]
    pos_weight = float((ys[0] == 0).sum()) / max(float(ys[0].sum()), 1.0)

    model = SageNet(Xs[0].shape[1])
    model = train_graph_model(model, *graphs[0], ys[0], *graphs[1], ys[1],
                              pos_weight=pos_weight, epochs=25, patience=5)
    val_s = _sigmoid(predict_logits(model, graphs[1][0].to("cuda"), graphs[1][1].to("cuda")))
    te_s = _sigmoid(predict_logits(model, graphs[2][0].to("cuda"), graphs[2][1].to("cuda")))
    return eval_scores_row(y_va, val_s, y_te, te_s)


def main():
    all_rows = []
    for split in ("time", "random"):
        splits = get_splits(split=split)
        gan_scaler = StandardScaler().fit(splits[0])
        print(f"\n===== SPLIT = {split.upper()} =====")
        print(f"train frauds {int(splits[1].sum())} | val {int(splits[3].sum())} | "
              f"test {int(splits[5].sum())}")
        for name, strategy, tier in CONFIGS:
            tag = f"{name:>14} + {strategy:<12}"
            try:
                if tier == "classical":
                    row = classical_run_one(name, strategy, splits, gan_scaler=gan_scaler)
                    del row["_pipeline"]
                elif tier == "deep":
                    row = run_deep(job=name, strategy=strategy, splits=splits)
                else:
                    row = run_gnn(strategy, splits)
                row.update({"model": name, "strategy": strategy, "split": split})
                all_rows.append(row)
                print(f"{tag} PR-AUC={row['pr_auc']:.4f} F1={row['f1']:.4f}")
            except Exception as e:
                print(f"{tag} FAILED: {type(e).__name__}: {e}")

    df = pd.DataFrame(all_rows)
    out = config.REPORTS / "results_random_split.csv"
    df.to_csv(out, index=False)

    print("\n=== RANDOM vs TIME GAP ===")
    piv = df.pivot_table(index=["model", "strategy"], columns="split",
                         values="pr_auc", aggfunc="first")
    piv["leakage_gap"] = piv["random"] - piv["time"]
    print(piv.round(4).sort_values("leakage_gap", ascending=False).to_string())
    print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
