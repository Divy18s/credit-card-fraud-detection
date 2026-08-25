"""Probe: does GraphSAGE improve with a larger optimizer-step budget?

`train_graph_model` is full-batch, so one epoch == one gradient step. The
shipped GNN tier therefore takes at most 40 steps, versus ~5,850 for the
mini-batched GAT and ~3,920 for the MLP. This script measures whether that
asymmetry disadvantages SAGE, by comparing the shipped setting against a
300-step run with early stopping disabled, and printing the val/test AP
trajectory that explains the outcome.

Result (seed 42, scaled k-NN graphs, k=10): 40 steps -> 0.7558;
300 steps without early stopping -> 0.7070. Validation AP keeps improving
while test AP falls, i.e. the extra steps fit the chronologically adjacent
validation window and extrapolate worse to the later test window (temporal
drift, not over-smoothing -- depth is fixed at two layers throughout).

Usage:
    python -m src.models.probe_sage_budget
"""

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.dataset import get_splits
from src.models.gnn import (DEVICE, SageNet, _sigmoid, build_knn_graph,
                            predict_logits, train_graph_model)

K = 10
TRAJECTORY_STEPS = (1, 5, 10, 20, 30, 40, 60, 100, 150, 200, 250, 300)


def _setup():
    X_tr, y_tr, X_va, y_va, X_te, y_te, _ = get_splits()
    scaler = StandardScaler().fit(X_tr)
    Xs = [torch.as_tensor(scaler.transform(a).astype(np.float32))
          for a in (X_tr, X_va, X_te)]
    ys = [torch.as_tensor(np.asarray(s, dtype=np.float32))
          for s in (y_tr, y_va, y_te)]
    pos_w = float((ys[0] == 0).sum()) / max(float(ys[0].sum()), 1.0)
    print(f"building k-NN graphs (scaled space, k={K}) ...", flush=True)
    eis = [build_knn_graph(x.numpy(), k=K) for x in Xs]
    return Xs, ys, eis, pos_w, np.asarray(y_va), np.asarray(y_te)


def _ap(model, x, ei, y):
    return average_precision_score(
        y, _sigmoid(predict_logits(model, x.to(DEVICE), ei.to(DEVICE))))


def run(Xs, ys, eis, pos_w, y_te, epochs, patience, tag):
    torch.manual_seed(config.RANDOM_STATE)
    model = SageNet(Xs[0].shape[1])
    model = train_graph_model(model, Xs[0], eis[0], ys[0], Xs[1], eis[1], ys[1],
                              pos_weight=pos_w, epochs=epochs, patience=patience)
    ap = _ap(model, Xs[2], eis[2], y_te)
    print(f"{tag:38} test PR-AUC = {ap:.4f}", flush=True)
    return ap


def trajectory(Xs, ys, eis, pos_w, y_va, y_te, steps=300):
    """Per-step val/test AP with no early stopping, to expose the divergence."""
    torch.manual_seed(config.RANDOM_STATE)
    model = SageNet(Xs[0].shape[1]).to(DEVICE)
    crit = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_w], device=DEVICE))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    x, ei, y = Xs[0].to(DEVICE), eis[0].to(DEVICE), ys[0].to(DEVICE)
    print(f"\nval/test AP trajectory ({steps} steps, no early stopping):")
    for step in range(1, steps + 1):
        model.train()
        opt.zero_grad()
        crit(model(x, ei), y).backward()
        opt.step()
        if step in TRAJECTORY_STEPS:
            print(f"  step {step:>4}  val AP={_ap(model, Xs[1], eis[1], y_va):.4f}"
                  f"  test AP={_ap(model, Xs[2], eis[2], y_te):.4f}", flush=True)


def main():
    Xs, ys, eis, pos_w, y_va, y_te = _setup()
    run(Xs, ys, eis, pos_w, y_te, 40, 6, "epochs=40  patience=6  (shipped)")
    run(Xs, ys, eis, pos_w, y_te, 300, 300, "epochs=300 patience=300 (no early stop)")
    trajectory(Xs, ys, eis, pos_w, y_va, y_te)


if __name__ == "__main__":
    main()
