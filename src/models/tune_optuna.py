"""Optuna hyperparameter tuning under the fixed evaluation protocol.

Objective: VALIDATION PR-AUC. Test set is touched exactly once per model,
after the study finishes, with the threshold tuned on validation.

Boosters are tuned on RAW training data with scale_pos_weight as a searchable
parameter (subsumes the class_weight strategy at zero resampling cost).
A follow-up fit transfers the best RF params onto the SMOTE-ENN champion combo.

Usage:
    python -m src.models.tune_optuna --models xgboost --trials 60
    python -m src.models.tune_optuna                # all, default budgets
"""

import argparse
import json
import math
import time

import numpy as np
import pandas as pd
import optuna
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src import config
from src.data.dataset import get_splits
from src.models.gnn import DEVICE, SageNet, _sigmoid, predict_logits, train_graph_model
from src.models.metrics import best_threshold_f1, evaluate_scores

optuna.logging.set_verbosity(optuna.logging.WARNING)

PARAMS_JSON = config.REPORTS / "optuna_best_params.json"
DEFAULT_TRIALS = {"random_forest": 25, "xgboost": 60, "lightgbm": 60, "catboost": 40}


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _suggest_spw(trial, ratio):
    return float(trial.suggest_float("scale_pos_weight", 1.0, ratio, log=True))


def _booster_objective(name, X_tr, y_tr, X_va, y_va):
    ratio = float((y_tr == 0).sum()) / max(float(y_tr.sum()), 1.0)
    rs = config.RANDOM_STATE

    if name == "xgboost":
        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 900),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                min_child_weight=trial.suggest_float("min_child_weight", 1, 100, log=True),
                gamma=trial.suggest_float("gamma", 0, 5),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20, log=True),
                scale_pos_weight=_suggest_spw(trial, ratio),
            )
            clf = XGBClassifier(tree_method="hist", device="cuda" if _cuda() else "hist",
                                eval_metric="aucpr", random_state=rs, n_jobs=-1, **params)
            clf.fit(X_tr, y_tr)
            return average_precision_score(y_va, clf.predict_proba(X_va)[:, 1])
    elif name == "lightgbm":
        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 300, 900),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int("num_leaves", 16, 127),
                max_depth=trial.suggest_int("max_depth", 4, 14),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 120),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 0.1, 20, log=True),
                scale_pos_weight=_suggest_spw(trial, ratio),
            )
            clf = LGBMClassifier(random_state=rs, n_jobs=-1, verbose=-1, **params)
            clf.fit(X_tr, y_tr)
            return average_precision_score(y_va, clf.predict_proba(X_va)[:, 1])
    elif name == "catboost":
        def objective(trial):
            params = dict(
                iterations=trial.suggest_int("iterations", 300, 900),
                learning_rate=trial.suggest_float("learning_rate", 0.02, 0.3, log=True),
                depth=trial.suggest_int("depth", 4, 8),
                l2_leaf_reg=trial.suggest_float("l2_leaf_reg", 1, 10, log=True),
                random_strength=trial.suggest_float("random_strength", 0.1, 10, log=True),
            )
            spw = _suggest_spw(trial, ratio)
            clf = CatBoostClassifier(verbose=False, random_seed=rs,
                                     task_type="GPU" if _cuda() else "CPU",
                                     class_weights=[1.0, spw], **params)
            clf.fit(X_tr, y_tr)
            return average_precision_score(y_va, clf.predict_proba(X_va)[:, 1])
    elif name == "random_forest":
        def objective(trial):
            params = dict(
                n_estimators=trial.suggest_int("n_estimators", 200, 600),
                max_depth=trial.suggest_categorical("max_depth", [None, 10, 16, 24, 32]),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 8),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            )
            clf = RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                         random_state=rs, **params)
            clf.fit(X_tr, y_tr)
            return average_precision_score(y_va, clf.predict_proba(X_va)[:, 1])
    else:
        raise ValueError(name)

    return objective


def tune_classical(name, splits, trials):
    X_tr, y_tr, X_va, y_va, X_te, y_te = splits[:6]
    t0 = time.perf_counter()
    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(_booster_objective(name, X_tr, y_tr, X_va, y_va),
                   n_trials=trials, show_progress_bar=False)
    elapsed = time.perf_counter() - t0
    print(f"{name}: best val PR-AUC={study.best_value:.4f} ({trials} trials, {elapsed:.0f}s)")
    print(f"  params: {study.best_params}")
    return study.best_params, elapsed


def finalize_classical_row(name, params, splits):
    X_tr, y_tr, X_va, y_va, X_te, y_te = splits[:6]
    rs = config.RANDOM_STATE
    if name == "xgboost":
        clf = XGBClassifier(tree_method="hist", device="cuda" if _cuda() else "hist",
                            eval_metric="aucpr", random_state=rs, n_jobs=-1, **params)
    elif name == "lightgbm":
        clf = LGBMClassifier(random_state=rs, n_jobs=-1, verbose=-1, **params)
    elif name == "catboost":
        params = dict(params)
        spw = params.pop("scale_pos_weight", None)
        if spw is not None:
            params["class_weights"] = [1.0, spw]
        clf = CatBoostClassifier(verbose=False, random_seed=rs,
                                 task_type="GPU" if _cuda() else "CPU", **params)
    else:
        clf = RandomForestClassifier(class_weight="balanced", n_jobs=-1,
                                     random_state=rs, **params)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    elapsed = time.perf_counter() - t0
    thr = best_threshold_f1(y_va, clf.predict_proba(X_va)[:, 1])
    spw = params.get("scale_pos_weight") or (
        params.get("class_weights")[1] if "class_weights" in params else None)
    row = {"model": name, "strategy": "optuna", "spw": round(spw, 1) if spw else None,
           "train_s": round(elapsed, 1)}
    row.update(evaluate_scores(y_te, clf.predict_proba(X_te)[:, 1], thr))
    return row


import src.models.gnn as gnn_mod


def tune_graphsage(scaled_arrays, splits_y, trials, fast=False):
    (X_tr_s, X_va_s, X_te_s), y_tr_t, y_va_t, y_te_t = scaled_arrays, *splits_y
    ratio = float((y_tr_t == 0).sum()) / max(float(y_tr_t.sum()), 1.0)
    import torch

    from src.models.gnn import build_knn_graph

    graph_cache = {}

    def get_graph(arr, k):
        key = (id(arr), k)
        if key not in graph_cache:
            graph_cache[key] = build_knn_graph(arr.numpy(), k=k)
        return graph_cache[key]

    def objective(trial):
        params = dict(
            k_neighbors=trial.suggest_categorical("k_neighbors", [5, 10, 15, 20]),
            hidden=trial.suggest_categorical("hidden", [32, 64, 128]),
            layers=trial.suggest_int("layers", 1, 2),
            lr=trial.suggest_float("lr", 1e-3, 1e-2, log=True),
            scale_pos_weight=trial.suggest_float("scale_pos_weight", 1.0, ratio, log=True),
        )
        torch.manual_seed(config.RANDOM_STATE)
        model = SageNet(X_tr_s.shape[1], hidden=params["hidden"], n_layers=params["layers"])
        model = gnn_mod.train_graph_model(
            model, X_tr_s, get_graph(X_tr_s, params["k_neighbors"]), y_tr_t,
            X_va_s, get_graph(X_va_s, params["k_neighbors"]), y_va_t,
            pos_weight=params["scale_pos_weight"], lr=params["lr"],
            epochs=2 if fast else 25, patience=2 if fast else 5)
        val_ap = average_precision_score(
            y_va_t.numpy(), _sigmoid(predict_logits(model, X_va_s.to(DEVICE),
                                                    get_graph(X_va_s, params["k_neighbors"]).to(DEVICE))))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return val_ap

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=config.RANDOM_STATE))
    study.optimize(objective, n_trials=trials, show_progress_bar=False)

    print(f"graphsage: best val PR-AUC={study.best_value:.4f} ({trials} trials)")
    print(f"  params: {study.best_params}")
    return study.best_params


def finalize_graphsage(params, scaled_arrays, splits_y):
    import torch

    (X_tr_s, X_va_s, X_te_s), y_tr_t, y_va_t, y_te_t = scaled_arrays, *splits_y
    from src.models.gnn import build_knn_graph

    torch.manual_seed(config.RANDOM_STATE)
    model = SageNet(X_tr_s.shape[1], hidden=params["hidden"], n_layers=params["layers"])
    k = params["k_neighbors"]

    t0 = time.perf_counter()
    model = train_graph_model(
        model, X_tr_s, build_knn_graph(X_tr_s.numpy(), k=k), y_tr_t,
        X_va_s, build_knn_graph(X_va_s.numpy(), k=k), y_va_t,
        pos_weight=params["scale_pos_weight"], lr=params["lr"], epochs=40, patience=6)
    elapsed = time.perf_counter() - t0

    val_s = _sigmoid(predict_logits(model, X_va_s.to(DEVICE),
                                    build_knn_graph(X_va_s.numpy(), k=k).to(DEVICE)))
    te_s = _sigmoid(predict_logits(model, X_te_s.to(DEVICE),
                                   build_knn_graph(X_te_s.numpy(), k=k).to(DEVICE)))
    thr = best_threshold_f1(y_va_t.numpy(), val_s)
    row = {"model": "graphsage", "strategy": "optuna",
           "spw": round(params["scale_pos_weight"], 1), "train_s": round(elapsed, 1)}
    row.update(evaluate_scores(y_te_t.numpy(), te_s, thr))
    return row


def merge_rows(rows):
    new = pd.DataFrame(rows)
    csv = config.RESULTS_CSV
    old = pd.read_csv(csv)
    keys = set(zip(new["model"], new["strategy"]))
    mask = ~old.apply(lambda r: (r["model"], r["strategy"]) in keys, axis=1)
    results = pd.concat([old[mask], new], ignore_index=True)
    results = results.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    results.to_csv(csv, index=False)
    cols = ["model", "strategy", "spw", "pr_auc", "roc_auc", "f1", "precision",
            "recall", "mcc", "fp", "fn", "train_s"]
    print("\n=== TOP 12 TEST SET RESULTS ===")
    print(results[[c for c in cols if c in results.columns]].head(12).to_string(index=False))
    print(f"\nSaved -> {csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=None, help="comma list: xgboost,lightgbm,catboost,random_forest,graphsage")
    parser.add_argument("--trials", type=int, default=None, help="override per-model trial budget")
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    models = (args.models.split(",") if args.models
              else ["random_forest", "xgboost", "lightgbm", "catboost"])

    splits = get_splits()
    X_tr = splits[0]
    scaler = StandardScaler().fit(X_tr)
    X_tr_s = scaler.transform(X_tr).astype(np.float32)
    X_va_s = scaler.transform(splits[2]).astype(np.float32)
    X_te_s = scaler.transform(splits[4]).astype(np.float32)
    import torch

    scaled_arrays = (torch.as_tensor(X_tr_s), torch.as_tensor(X_va_s), torch.as_tensor(X_te_s))
    splits_y = tuple(torch.as_tensor(np.asarray(splits[i], dtype=np.float32)) for i in (1, 3, 5))

    all_params = {}
    if PARAMS_JSON.exists():
        all_params = json.loads(PARAMS_JSON.read_text())

    rows = []
    for name in models:
        if args.fast:
            trials = 2
        else:
            trials = args.trials or DEFAULT_TRIALS.get(name, 30)

        if name == "graphsage":
            params = tune_graphsage(scaled_arrays, splits_y, trials, fast=args.fast)
            all_params[name] = params
            rows.append(finalize_graphsage(params, scaled_arrays, splits_y))
        else:
            params, _ = tune_classical(name, splits, trials)
            all_params[name] = params
            rows.append(finalize_classical_row(name, params, splits))

            if name == "random_forest":
                from imblearn.pipeline import Pipeline as ImbPipeline
                from src.models.train_classical import make_classifier, make_sampler

                clf, pre = make_classifier(name, False, int(splits[1].sum()),
                                           int((splits[1] == 0).sum()), None)
                steps = [(f"prep_{i}", t) for i, t in enumerate(pre)]
                steps.append(("sampler", make_sampler("smote_enn")))
                steps.append(("clf", clf.set_params(**{k: v for k, v in params.items()})))
                pipe = ImbPipeline(steps)
                t0 = time.perf_counter()
                pipe.fit(splits[0], splits[1])
                el = time.perf_counter() - t0
                thr = best_threshold_f1(splits[3], pipe.predict_proba(splits[2])[:, 1])
                row = {"model": "random_forest", "strategy": "optuna_smote_enn",
                       "spw": None, "train_s": round(el, 1)}
                row.update(evaluate_scores(splits[5], pipe.predict_proba(splits[4])[:, 1], thr))
                rows.append(row)

    PARAMS_JSON.parent.mkdir(parents=True, exist_ok=True)
    PARAMS_JSON.write_text(json.dumps(all_params, indent=2, default=str))
    merge_rows(rows)


if __name__ == "__main__":
    main()
