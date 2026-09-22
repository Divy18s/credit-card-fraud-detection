"""SHAP explainability for the champion model (RF + SMOTE-ENN).

Trains the champion under the standard protocol, saves the fitted pipeline
for the demo app, then explains its test-set predictions with TreeExplainer:
global importance, beeswarm summaries, and a single-fraud waterfall.

Usage:
    python -m src.models.explain_shap [--sample 3000]
"""

import argparse

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import shap

from src import config
from src.data.dataset import get_splits
from src.models.train_classical import run_one as classical_run_one


def _class1_shap(shap_values):
    if isinstance(shap_values, list):
        return shap_values[1]
    arr = np.asarray(shap_values)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def _expected_value(explainer):
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)):
        flat = np.asarray(ev).ravel()
        return float(flat[1]) if len(flat) > 1 else float(flat[0])
    return float(ev)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=3000)
    args = parser.parse_args()

    splits = get_splits(split="time")
    X_tr, y_tr, X_va, y_va, X_te, y_te, feats = splits
    print("Training champion (random_forest + smote_enn) ...")
    row = classical_run_one("random_forest", "smote_enn", splits)
    pipe = row.pop("_pipeline")
    print(f"champion test PR-AUC={row['pr_auc']:.4f}")

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = config.MODELS_DIR / "champion_rf_smote_enn.joblib"
    joblib.dump({"pipeline": pipe,
                 "threshold": row["threshold"],
                 "feature_order": list(feats)}, model_path)
    print(f"saved -> {model_path}")

    rf = pipe.named_steps["clf"]
    rng = np.random.RandomState(config.RANDOM_STATE)
    y_te_arr = np.asarray(y_te)
    fraud_idx = np.where(y_te_arr == 1)[0]
    legit_idx = np.where(y_te_arr == 0)[0]
    legit_sample = rng.choice(legit_idx, size=min(args.sample, len(legit_idx)),
                              replace=False)
    sample_idx = np.sort(np.concatenate([fraud_idx, legit_sample]))
    X_sample = X_te.iloc[sample_idx]

    print(f"Computing TreeExplainer SHAP on {len(X_sample)} rows "
          f"({len(fraud_idx)} fraud + {len(legit_sample)} legit) ...")
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(X_sample)
    sv_fraud = _class1_shap(sv)
    ev_fraud = _expected_value(explainer)

    fig_dir = config.FIGURES
    fig_dir.mkdir(parents=True, exist_ok=True)

    expl = shap.Explanation(values=sv_fraud,
                            base_values=np.full(len(X_sample), ev_fraud),
                            data=X_sample.values,
                            feature_names=list(X_sample.columns))
    mean_abs = np.abs(sv_fraud).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    top = order[:15][::-1]
    ax.barh([X_sample.columns[i] for i in top], mean_abs[top], color="#c44e52")
    ax.set_xlabel("mean |SHAP value|  (impact on fraud probability)")
    ax.set_title(f"Global feature importance — RF+SMOTE-ENN champion\n"
                 f"(TreeExplainer, n={len(X_sample)} test txns)")
    fig.tight_layout()
    fig.savefig(fig_dir / "shap_global_importance.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(expl, max_display=15, show=False)
    plt.title("SHAP beeswarm — all sampled test transactions", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_beeswarm.png", dpi=150)
    plt.close()

    fig = plt.figure(figsize=(10, 7))
    shap.plots.beeswarm(expl[:len(fraud_idx)], max_display=15, show=False)
    plt.title("SHAP beeswarm — actual FRAUD transactions only (n=52)", fontsize=12)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_beeswarm_fraud.png", dpi=150)
    plt.close()

    fig = plt.figure(figsize=(10, 8))
    shap.plots.waterfall(expl[0], max_display=12, show=False)
    plt.title("Why THIS transaction was flagged as fraud (test case #1)",
              fontsize=11)
    plt.tight_layout()
    plt.savefig(fig_dir / "shap_waterfall_example.png", dpi=150)
    plt.close()

    lines = [
        "=" * 70,
        "SHAP EXPLAINABILITY SUMMARY - RF + SMOTE-ENN CHAMPION",
        "=" * 70,
        f"Explained rows      : {len(X_sample)} ({len(fraud_idx)} frauds, "
        f"{len(legit_sample)} legit sample)",
        f"Base value (logit)  : {ev_fraud:.4f}",
        "",
        f"{'Rank':<5}{'Feature':<12}{'mean|SHAP|':<12}{'Pushes toward'}",
        "-" * 45,
    ]
    for r, i in enumerate(order[:15], 1):
        col = X_sample.columns[i]
        vals = sv_fraud[:, i]
        corr_with_high = np.corrcoef(X_sample[col].values, vals)[0, 1]
        direction = ("HIGH value -> fraud" if corr_with_high > 0
                     else "LOW value  -> fraud")
        lines.append(f"{r:<5}{col:<12}{mean_abs[i]:<12.4f}{direction}")
    lines += ["", "Figures: reports/figures/shap_*.png"]
    summary_path = config.REPORTS / "shap_summary.txt"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:24]))
    print(f"\nSaved -> {summary_path}")


if __name__ == "__main__":
    main()
