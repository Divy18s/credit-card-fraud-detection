"""Train GNN tier (GraphSAGE, GATv2) on k-NN transaction graphs.

Same protocol as all tiers: time-based split, graphs built per-split
(no cross-split edges), pos_weight on TRAIN only, threshold tuned on
validation, metrics reported on the untouched test set.

Usage:
    python -m src.models.train_gnn [--fast]
"""

import argparse
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.dataset import get_splits
from src.models.gnn import (
    DEVICE,
    GatNet,
    SageNet,
    build_knn_graph,
    _sigmoid,
    predict_logits,
    predict_minibatch,
    train_gat_minibatch,
    train_graph_model,
)
from src.models.metrics import best_threshold_f1, evaluate_scores

K_NEIGHBORS = 10


def run_one(name: str, strategy: str, scaled, splits_y, fast: bool) -> dict:
    (x_tr, ei_tr), (x_va, ei_va), (x_te, ei_te) = scaled
    y_tr, y_va, y_te = splits_y

    pos_weight = None
    if strategy == "class_weight":
        pos_weight = float((y_tr == 0).sum()) / max(float(y_tr.sum()), 1.0)

    t0 = time.perf_counter()
    if name == "graphsage":
        model = SageNet(x_tr.shape[1])
        model = train_graph_model(
            model, x_tr, ei_tr, y_tr, x_va, ei_va, y_va,
            pos_weight=pos_weight,
            epochs=2 if fast else 40,
            patience=2 if fast else 6,
            verbose=False,
        )
    else:
        model = GatNet(x_tr.shape[1])
        model = train_gat_minibatch(
            model, x_tr, ei_tr, y_tr,
            pos_weight=pos_weight,
            epochs=2 if fast else 30,
            patience=2 if fast else 5,
        )
    elapsed = time.perf_counter() - t0

    if name == "graphsage":
        val_s = _sigmoid(predict_logits(model, x_va.to(DEVICE), ei_va.to(DEVICE)))
        te_s = _sigmoid(predict_logits(model, x_te.to(DEVICE), ei_te.to(DEVICE)))
    else:
        val_s = _sigmoid(predict_minibatch(model, x_va, ei_va, y_va))
        te_s = _sigmoid(predict_minibatch(model, x_te, ei_te, y_te))

    thr = best_threshold_f1(y_va.numpy(), val_s)
    row = {"model": name, "strategy": strategy,
           "spw": round(pos_weight, 1) if pos_weight else None,
           "train_s": round(elapsed, 1)}
    row.update(evaluate_scores(y_te.numpy(), te_s, thr))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    jobs = [
        ("graphsage", "raw"),
        ("graphsage", "class_weight"),
        ("gat", "class_weight"),
    ]

    splits = get_splits()
    X_tr, y_tr, X_va, y_va, X_te, y_te, feats = splits

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(X_va).astype(np.float32)
    X_te_s = scaler.transform(X_te).astype(np.float32)

    print(f"device={DEVICE} | building k-NN graphs (k={K_NEIGHBORS}) ...")
    graphs = []
    for X_s in (X_tr_s, X_va_s, X_te_s):
        ei = build_knn_graph(X_s, k=K_NEIGHBORS)
        graphs.append((torch.as_tensor(X_s), ei))
        print(f"    nodes={len(X_s):>7}  directed-edges={ei.shape[1] // 2:>9}")

    y_tr_t = torch.as_tensor(np.asarray(y_tr, dtype=np.float32))
    y_va_t = torch.as_tensor(np.asarray(y_va, dtype=np.float32))
    y_te_t = torch.as_tensor(np.asarray(y_te, dtype=np.float32))

    rows = []
    for name, strat in jobs:
        tag = f"{name:>10} + {strat:<12}"
        try:
            row = run_one(name, strat, graphs, (y_tr_t, y_va_t, y_te_t), args.fast)
            rows.append(row)
            print(f"{tag} PR-AUC={row['pr_auc']:.4f} F1={row['f1']:.4f} "
                  f"MCC={row['mcc']:.4f} ({row['train_s']}s)")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"{tag} FAILED: {type(e).__name__}: {e}")

    new = pd.DataFrame(rows)
    csv = config.RESULTS_CSV
    old = pd.read_csv(csv)
    keys = set(zip(new["model"], new["strategy"]))
    mask = ~old.apply(lambda r: (r["model"], r["strategy"]) in keys, axis=1)
    results = pd.concat([old[mask], new], ignore_index=True)
    results = results.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    results.to_csv(csv, index=False)

    cols = ["model", "strategy", "spw", "pr_auc", "roc_auc", "f1",
            "precision", "recall", "mcc", "fp", "fn", "train_s"]
    print("\n=== TOP 12 TEST SET RESULTS (sorted by PR-AUC) ===")
    print(results[[c for c in cols if c in results.columns]].head(12).to_string(index=False))
    print(f"\nSaved -> {csv}")


if __name__ == "__main__":
    main()
