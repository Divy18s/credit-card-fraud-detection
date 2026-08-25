"""Train classical ML baselines across imbalance-handling strategies.

Protocol (from the literature):
- time-based 70/15/15 split; resampling applied to TRAIN only
- decision threshold tuned on the validation set (max F1)
- for boosted trees under class_weight, scale_pos_weight is selected on the
  validation set from a small grid (extreme raw ratios collapse without it)
- final metrics reported on the untouched imbalanced test set
- primary metric: PR-AUC (average precision)

Usage:
    python -m src.models.train_classical                     # full sweep
    python -m src.models.train_classical --models lightgbm --strategies class_weight
"""

import argparse
import math
import time

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTEENN
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import config
from src.data.dataset import get_splits
from src.models.metrics import best_threshold_f1, evaluate_scores

RESULTS_CSV = config.REPORTS / "results_classical.csv"
SPW_MODELS = {"xgboost", "lightgbm", "catboost"}


def make_sampler(strategy: str):
    if strategy == "rus":
        return RandomUnderSampler(sampling_strategy=1.0, random_state=config.RANDOM_STATE)
    if strategy == "smote":
        return SMOTE(sampling_strategy=1.0, random_state=config.RANDOM_STATE, k_neighbors=5)
    if strategy == "smote_enn":
        return SMOTEENN(
            smote=SMOTE(sampling_strategy=1.0, random_state=config.RANDOM_STATE),
            random_state=config.RANDOM_STATE,
        )
    return None


def make_classifier(name: str, weighted: bool, n_pos: int, n_neg: int, spw: float | None = None):
    rs = config.RANDOM_STATE
    ratio = n_neg / n_pos
    if name == "logreg":
        clf = LogisticRegression(max_iter=2000, random_state=rs)
        if weighted:
            clf.set_params(class_weight="balanced")
        return clf, [StandardScaler()]
    if name == "decision_tree":
        clf = DecisionTreeClassifier(random_state=rs)
        if weighted:
            clf.set_params(class_weight="balanced")
        return clf, []
    if name == "gaussian_nb":
        return GaussianNB(), [StandardScaler()]
    if name == "linear_svc":
        base = LinearSVC(max_iter=5000, random_state=rs, dual="auto")
        if weighted:
            base.set_params(class_weight="balanced")
        return CalibratedClassifierCV(base, cv=3), [StandardScaler()]
    if name == "random_forest":
        clf = RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, n_jobs=-1, random_state=rs
        )
        if weighted:
            clf.set_params(class_weight="balanced")
        return clf, []
    if name == "xgboost":
        device = "cuda" if _cuda_available() else "hist"
        clf = XGBClassifier(
            n_estimators=500,
            learning_rate=0.1,
            max_depth=6,
            tree_method="hist",
            device=device,
            eval_metric="aucpr",
            random_state=rs,
            n_jobs=-1,
        )
        if weighted:
            clf.set_params(scale_pos_weight=spw if spw is not None else ratio)
        return clf, []
    if name == "lightgbm":
        clf = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            random_state=rs,
            n_jobs=-1,
            verbose=-1,
        )
        if weighted:
            clf.set_params(scale_pos_weight=spw if spw is not None else ratio)
        return clf, []
    if name == "catboost":
        task = "GPU" if _cuda_available() else "CPU"
        kwargs = dict(
            iterations=600,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            random_seed=rs,
        )
        kwargs["task_type"] = task
        if weighted:
            kwargs["class_weights"] = [1.0, spw if spw is not None else ratio]
        return CatBoostClassifier(**kwargs), []
    raise ValueError(f"unknown model: {name}")


def _cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _build_pipe(name: str, strategy: str, weighted: bool, n_pos: int, n_neg: int, spw):
    clf, pre = make_classifier(name, weighted, n_pos, n_neg, spw)
    sampler = make_sampler(strategy)
    steps = [(f"prep_{i}", t) for i, t in enumerate(pre)]
    if sampler is not None:
        steps.append(("sampler", sampler))
    steps.append(("clf", clf))
    return ImbPipeline(steps)


def _gan_augment(X_tr, y_tr, scaler):
    """Generate synthetic frauds with a GAN (in scaled space) and return
    raw-space augmented training data balanced to 1:1."""
    from src.models.gan import oversample_with_gan

    X_s = scaler.transform(X_tr)
    X_aug_s, y_aug = oversample_with_gan(np.asarray(X_s, dtype=np.float32), np.asarray(y_tr))
    n_gen = len(y_aug) - len(y_tr)
    X_raw_gen = scaler.inverse_transform(X_aug_s[len(X_tr):])
    return np.vstack([np.asarray(X_tr), X_raw_gen]), y_aug


def run_one(name: str, strategy: str, splits, gan_scaler=None) -> dict:
    X_tr, y_tr, X_va, y_va, X_te, y_te = splits[:6]
    if strategy == "gan":
        X_fit, y_fit = _gan_augment(X_tr, y_tr, gan_scaler)
        splits_fit = (X_fit, y_fit, X_va, y_va, X_te, y_te)
    else:
        splits_fit = splits
    n_pos = int(np.asarray(splits_fit[1]).sum())
    n_neg = int((np.asarray(splits_fit[1]) == 0).sum())
    weighted = strategy in ("class_weight", "class_weight_fixed")
    tune_spw = strategy == "class_weight"
    ratio = round(n_neg / n_pos, 1)

    if tune_spw and name in SPW_MODELS:
        grid = [1.0, 4.0, 16.0, round(math.sqrt(n_neg / n_pos), 1), 64.0, ratio]
        best_spw, best_val, best_pipe, fit_s = None, -1.0, None, 0.0
        for spw in sorted(set(grid)):
            pipe = _build_pipe(name, strategy, weighted, n_pos, n_neg, spw)
            t0 = time.perf_counter()
            pipe.fit(splits_fit[0], splits_fit[1])
            fit_s += time.perf_counter() - t0
            val_ap = average_precision_score(y_va, pipe.predict_proba(X_va)[:, 1])
            print(f"    spw={spw:<6} val PR-AUC={val_ap:.4f}")
            if val_ap > best_val:
                best_spw, best_val, best_pipe = spw, val_ap, pipe
        pipe, elapsed = best_pipe, fit_s
    else:
        pipe = _build_pipe(name, strategy, weighted, n_pos, n_neg, None)
        t0 = time.perf_counter()
        pipe.fit(splits_fit[0], splits_fit[1])
        elapsed = time.perf_counter() - t0
        best_spw = ratio if weighted else None

    thr = best_threshold_f1(y_va, pipe.predict_proba(X_va)[:, 1])
    row = {
        "model": name,
        "strategy": strategy,
        "spw": best_spw,
        "train_s": round(elapsed, 1),
    }
    row.update(evaluate_scores(y_te, pipe.predict_proba(X_te)[:, 1], thr))
    row["_pipeline"] = pipe
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None, help="comma-separated subset")
    parser.add_argument("--strategies", default=None, help="comma-separated subset")
    parser.add_argument("--split", default="time", choices=["time", "sang", "random"],
                        help="evaluation protocol; 'sang' is the 66/14/20 replication check")
    args = parser.parse_args()

    all_models = ["logreg", "decision_tree", "gaussian_nb", "linear_svc",
                  "random_forest", "xgboost", "lightgbm", "catboost"]
    all_strategies = ["raw", "class_weight", "class_weight_fixed", "rus", "smote", "smote_enn", "gan"]
    models = args.models.split(",") if args.models else all_models
    strategies = args.strategies.split(",") if args.strategies else all_strategies

    splits = get_splits(args.split)
    from sklearn.preprocessing import StandardScaler

    gan_scaler = StandardScaler().fit(splits[0])
    y_tr = splits[1]
    print(
        f"train {len(y_tr)} (fraud {int(y_tr.sum())}) | "
        f"val {len(splits[3])} (fraud {int(splits[3].sum())}) | "
        f"test {len(splits[5])} (fraud {int(splits[5].sum())})"
    )

    rows = []
    for m in models:
        for s in strategies:
            if s == "class_weight_fixed" and m not in SPW_MODELS:
                continue
            if s in ("class_weight", "class_weight_fixed") and m == "gaussian_nb":
                print(f"{m:>14} + {s:<12} SKIPPED (Naive Bayes has no class weights)")
                continue
            tag = f"{m:>14} + {s:<12}"
            try:
                row = run_one(m, s, splits, gan_scaler=gan_scaler)
                rows.append(row)
                print(
                    f"{tag} PR-AUC={row['pr_auc']:.4f} F1={row['f1']:.4f} "
                    f"MCC={row['mcc']:.4f} spw={row['spw']} ({row['train_s']}s)"
                )
                del row["_pipeline"]
            except Exception as e:
                print(f"{tag} FAILED: {type(e).__name__}: {e}")

    new_results = pd.DataFrame(rows)
    if args.split != "time":
        # replication / ablation protocols get their own ledger; the primary
        # results_classical.csv is never overwritten by a non-default split.
        out = config.REPORTS / f"results_{args.split}_split.csv"
        new_results["split"] = args.split
        if out.exists():
            old_rows = pd.read_csv(out)
            keys = set(zip(new_results["model"], new_results["strategy"]))
            keep = ~old_rows.apply(lambda r: (r["model"], r["strategy"]) in keys, axis=1)
            new_results = pd.concat([old_rows[keep], new_results], ignore_index=True)
        new_results.to_csv(out, index=False)
        print(f"Saved -> {out}")
        return
    full_sweep = args.models is None and args.strategies is None
    if RESULTS_CSV.exists() and not full_sweep:
        old = pd.read_csv(RESULTS_CSV)
        keys = set(zip(new_results["model"], new_results["strategy"]))
        mask = ~old.apply(lambda r: (r["model"], r["strategy"]) in keys, axis=1)
        results = pd.concat([old[mask], new_results], ignore_index=True)
    else:
        results = new_results

    results = results.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_CSV, index=False)

    cols = ["model", "strategy", "spw", "pr_auc", "roc_auc", "f1", "precision", "recall", "mcc", "fp", "fn", "train_s"]
    print("\n=== TEST SET RESULTS (sorted by PR-AUC) ===")
    print(results[[c for c in cols if c in results.columns]].to_string(index=False))
    print(f"\nSaved -> {RESULTS_CSV}")


if __name__ == "__main__":
    main()
