"""Bootstrap confidence intervals for the leaderboard's representative models.

Retrains one config per tier once, saves test-set score vectors, then runs a
STRATIFIED bootstrap (frauds and legits resampled separately, preserving the
52/42,670 counts) to produce 95% CIs for PR-AUC without any retraining.

Usage:
    python -m src.models.bootstrap_ci [--n-boot 1000] [--models rf,lstm,...]
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler

from src import config
from src.data.dataset import get_splits
from src.models.metrics import best_threshold_f1

SCORES_DIR = config.REPORTS / "scores"
SEQ_LEN = 8

CONFIGS = {
    "rf_smote_enn": ("classical", "random_forest", "smote_enn"),
    "xgb_smote_enn": ("classical", "xgboost", "smote_enn"),
    "catboost_gan": ("classical", "catboost", "gan"),
    "mlp_raw": ("deep", "mlp", "raw"),
    "lstm_weighted": ("deep", "lstm", "class_weight"),
    "graphsage": ("gnn", "graphsage", "class_weight"),
}


def get_test_scores(key: str, splits):
    tier, name, strategy = CONFIGS[key]
    X_tr, y_tr, X_va, y_va, X_te, y_te = splits[:6]
    t0 = time.perf_counter()

    if tier == "classical":
        from sklearn.preprocessing import StandardScaler as _S

        from src.models.train_classical import run_one as c_run_one

        row = c_run_one(name, strategy, splits,
                        gan_scaler=_S().fit(X_tr))
        pipe = row.pop("_pipeline")
        probs = pipe.predict_proba(X_te)[:, 1]

    elif tier == "deep":
        from src.models.train_deep import train_lstm, train_mlp

        scaler = StandardScaler().fit(X_tr)
        Xs = [scaler.transform(a).astype(np.float32) for a in (X_tr, X_va, X_te)]
        if name == "mlp":
            _, te_s, _ = train_mlp(Xs[0], y_tr, Xs[1], y_va, Xs[2], strategy, False)
            probs = np.asarray(te_s)
            y_out = np.asarray(y_te)
        else:
            _, te_s, _ = train_lstm(Xs[0], y_tr, Xs[1], y_va, Xs[2], False)
            probs = np.asarray(te_s)
            y_out = np.asarray(y_te)[SEQ_LEN - 1:]
        return y_out, probs, time.perf_counter() - t0

    else:
        import torch

        from src.models.gnn import (
            DEVICE, SageNet, _sigmoid, build_knn_graph, predict_logits,
            train_graph_model,
        )

        scaler = StandardScaler().fit(X_tr)
        Xs = [torch.as_tensor(scaler.transform(a).astype(np.float32)) for a in (X_tr, X_va, X_te)]
        ys = [torch.as_tensor(np.asarray(s, dtype=np.float32)) for s in (y_tr, y_va, y_te)]
        pos_w = float((ys[0] == 0).sum()) / max(float(ys[0].sum()), 1.0)
        ei_tr = build_knn_graph(X_tr.astype(np.float32), k=10)
        ei_te = build_knn_graph(X_te.astype(np.float32), k=10)
        model = SageNet(Xs[0].shape[1])
        model = train_graph_model(model, Xs[0], ei_tr, ys[0], Xs[1],
                                  build_knn_graph(X_va.astype(np.float32), k=10),
                                  ys[1], pos_weight=pos_w, epochs=40, patience=6)
        logits = predict_logits(model, Xs[2].to(DEVICE), ei_te.to(DEVICE))
        probs = _sigmoid(logits)
        return np.asarray(y_te), probs, time.perf_counter() - t0

    return np.asarray(y_te), np.asarray(probs), time.perf_counter() - t0


def stratified_bootstrap_ci(y, probs, n_boot=1000, seed=config.RANDOM_STATE):
    rng = np.random.RandomState(seed)
    fraud = np.where(y == 1)[0]
    legit = np.where(y == 0)[0]
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([
            rng.choice(fraud, size=len(fraud), replace=True),
            rng.choice(legit, size=len(legit), replace=True),
        ])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(average_precision_score(y[idx], probs[idx]))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi), int(len(vals))


def merge_and_report(rows):
    out_new = pd.DataFrame(rows)
    ci_path = config.REPORTS / "bootstrap_ci.csv"
    if ci_path.exists():
        old = pd.read_csv(ci_path)
        keys = set(out_new["config"])
        old = old[~old["config"].isin(keys)]
        out = pd.concat([old, out_new], ignore_index=True)
    else:
        out = out_new
    out = out.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    out.to_csv(ci_path, index=False)

    print("\n=== PAIRWISE CI OVERLAP (overlap => statistical tie) ===")
    for i in range(len(out)):
        for j in range(i + 1, len(out)):
            a, b = out.iloc[i], out.iloc[j]
            overlap = not (a.ci_low > b.ci_high or b.ci_low > a.ci_high)
            verdict = "TIE" if overlap else f"{a.config} > {b.config}"
            print(f"  {a.config:<14} vs {b.config:<14}: {verdict}")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = out.sort_values("pr_auc")
    fig, ax = plt.subplots(figsize=(9, 5))
    ypos = np.arange(len(d))
    ax.errorbar(d["pr_auc"], ypos,
                xerr=[d["pr_auc"] - d["ci_low"], d["ci_high"] - d["pr_auc"]],
                fmt="o", color="#c44e52", ecolor="#55a868", capsize=4)
    ax.set_yticks(ypos)
    ax.set_yticklabels(d["config"])
    ax.set_xlabel("Test PR-AUC  (95% stratified-bootstrap CI)")
    ax.set_title("Leaderboard with bootstrap uncertainty")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(config.FIGURES / "bootstrap_ci.png", dpi=150)

    print("\n" + out.drop(columns=[]).to_string(index=False))
    print(f"\nSaved -> {ci_path}")
    print(f"Figure -> {config.FIGURES / 'bootstrap_ci.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--models", default=None,
                        help="comma list from: " + ",".join(CONFIGS))
    args = parser.parse_args()

    keys = args.models.split(",") if args.models else list(CONFIGS)
    splits = get_splits(split="time")

    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for key in keys:
        tier, name, strategy = CONFIGS[key]
        print(f"[{key}] training {name} + {strategy} ...")
        y_te, probs, elapsed = get_test_scores(key, splits)
        np.savez(SCORES_DIR / f"{key}.npz", y=y_te, probs=probs)
        ap = average_precision_score(y_te, probs)
        lo, hi, n_used = stratified_bootstrap_ci(y_te, probs, args.n_boot)
        rows.append({"config": key, "model": name, "strategy": strategy,
                     "pr_auc": round(ap, 4), "ci_low": round(lo, 4),
                     "ci_high": round(hi, 4), "ci_width": round(hi - lo, 4),
                     "n_boot": n_used, "train_s": round(elapsed, 1)})
        print(f"    PR-AUC={ap:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  width={hi - lo:.4f}")

    merge_and_report(rows)


if __name__ == "__main__":
    main()
