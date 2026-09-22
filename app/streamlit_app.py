"""Streamlit demo: score transactions with the champion RF+SMOTE-ENN model
and explain every prediction with SHAP.

Launch:
    .venv\\Scripts\\streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import matplotlib.pyplot as plt
import shap

from src import config
from src.data.dataset import add_features

MODEL_PATH = config.MODELS_DIR / "champion_rf_smote_enn.joblib"
RAW_FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]
MODEL_FEATURES = RAW_FEATURES[:28] + ["Amount", "log_Amount", "hour_sin", "hour_cos"]

st.set_page_config(page_title="Fraud Detection Demo", page_icon="🕵️", layout="wide")


@st.cache_resource
def load_model():
    obj = joblib.load(MODEL_PATH)
    if isinstance(obj, dict):
        return obj["pipeline"], float(obj["threshold"]), list(obj["feature_order"])
    return obj, 0.7181, None


@st.cache_data
def load_transactions():
    df = pd.read_csv(config.REPORTS / ".." / "data" / "raw" / "creditcard.csv")
    return df.drop(columns=[config.TARGET_COL])


@st.cache_data
def load_labels():
    return pd.read_csv(config.REPORTS / ".." / "data" / "raw" / "creditcard.csv",
                       usecols=["Class"])["Class"]


@st.cache_data
def load_leaderboard():
    return pd.read_csv(config.REPORTS / "results_classical.csv")


@st.cache_data
def load_bootstrap():
    p = config.REPORTS / "bootstrap_ci.csv"
    return pd.read_csv(p) if p.exists() else None


@st.cache_data
def load_split_ablation():
    p = config.REPORTS / "results_random_split.csv"
    return pd.read_csv(p) if p.exists() else None


def engineer(txn: dict) -> pd.DataFrame:
    df = pd.DataFrame([txn])
    df = add_features(df)
    order = FEATURE_ORDER if FEATURE_ORDER else MODEL_FEATURES
    return df[order]


def fraud_shap(model, x_row: pd.DataFrame):
    rf = model.named_steps["clf"]
    explainer = shap.TreeExplainer(rf)
    sv = explainer.shap_values(x_row)
    if isinstance(sv, list):
        vals, ev = sv[1][0], np.asarray(explainer.expected_value).ravel()[1]
    else:
        arr = np.asarray(sv)
        vals = arr[0, :, 1] if arr.ndim == 3 else arr[0]
        ev_arr = np.asarray(explainer.expected_value).ravel()
        ev = float(ev_arr[1]) if len(ev_arr) > 1 else float(ev_arr[0])
    expl = shap.Explanation(values=vals,
                            base_values=float(ev),
                            data=x_row.values[0],
                            feature_names=list(x_row.columns))
    proba = float(model.predict_proba(x_row)[0, 1])
    return expl, proba


def waterfall_fig(expl):
    fig = plt.figure(figsize=(9, 6))
    shap.plots.waterfall(expl, max_display=10, show=False)
    plt.tight_layout()
    return fig


def top_drivers(expl, n=6):
    s = pd.Series(expl.values, index=expl.feature_names)
    top = s.reindex(s.abs().sort_values(ascending=False).head(n).index)
    rows = []
    for feat, val in top.items():
        arrow = "🔺 pushes toward FRAUD" if val > 0 else "🔻 pushes toward LEGIT"
        rows.append({"feature": feat, "value": round(float(expl.data[expl.feature_names.index(feat)]), 3),
                     "shap_impact": round(float(val), 4), "effect": arrow})
    return pd.DataFrame(rows)


model, THRESHOLD, FEATURE_ORDER = load_model()
data = load_transactions()
labels = load_labels()
TEST_START = int(len(data) * 0.85)

st.title("🕵️ Credit Card Fraud Detection")
st.caption("Champion model: Random Forest + SMOTE-ENN · PR-AUC 0.795 · "
           "every prediction explained with SHAP")

tab_single, tab_batch, tab_board = st.tabs(
    ["Single transaction", "Batch CSV", "Leaderboard"])

with tab_board:
    st.subheader("Full experiment leaderboard")
    st.caption("72 time-split runs · validation-tuned thresholds · untouched test set "
               "(42,722 txns / 52 frauds) · source: reports/results_classical.csv")

    board = load_leaderboard().copy()
    board["config"] = board["model"] + " + " + board["strategy"].astype(str)
    tiers = sorted(board["model"].unique())
    picked = st.multiselect("Filter models", tiers, default=tiers,
                            placeholder="Choose one or more models")
    min_ap = st.slider("Minimum PR-AUC", 0.0, 1.0, 0.0, 0.01)

    view = board[board["model"].isin(picked) & (board["pr_auc"] >= min_ap)]
    view = view.sort_values("pr_auc", ascending=False)

    st.dataframe(
        view[["model", "strategy", "spw", "pr_auc", "roc_auc", "f1",
              "precision", "recall", "mcc", "fp", "fn"]],
        use_container_width=True, hide_index=True, height=420,
        column_config={
            "pr_auc": st.column_config.ProgressColumn(
                "PR-AUC", min_value=0.0, max_value=1.0, format="%.4f"),
            "roc_auc": st.column_config.NumberColumn(format="%.4f"),
            "f1": st.column_config.NumberColumn(format="%.4f"),
            "precision": st.column_config.NumberColumn(format="%.4f"),
            "recall": st.column_config.NumberColumn(format="%.4f"),
            "mcc": st.column_config.NumberColumn(format="%.4f"),
        })

    top_n = st.slider("Top-N chart", 5, 25, 12)
    chart_src = view.head(top_n).set_index("config")["pr_auc"]
    st.bar_chart(chart_src, height=380)
    st.caption("Bars are point estimates; differences below ~±0.02 are noise "
               "at this test size — see bootstrap CIs below.")

    ci = load_bootstrap()
    if ci is not None:
        st.subheader("Bootstrap 95% confidence intervals")
        st.caption("Stratified resampling of the test set ×1000. Overlapping "
                   "intervals = statistical tie (all pairs overlap here).")
        st.dataframe(ci[["config", "pr_auc", "ci_low", "ci_high", "ci_width",
                         "n_boot"]], use_container_width=True, hide_index=True)
        st.image(str(config.FIGURES / "bootstrap_ci.png"))

    abl = load_split_ablation()
    if abl is not None:
        st.subheader("Random-split vs time-split ablation")
        piv = abl.pivot_table(index=["model", "strategy"], columns="split",
                              values="pr_auc", aggfunc="first").reset_index()
        piv["leakage_gap"] = (piv["random"] - piv["time"]).round(4)
        st.dataframe(piv.round(4), use_container_width=True, hide_index=True)
        st.caption("Positive leakage_gap = shuffling inflates the score.")

with tab_single:
    left, right = st.columns([1, 1])

    with left:
        st.subheader("Transaction")
        if st.button("🎲 Draw random transaction"):
            st.session_state["txn_idx"] = int(np.random.randint(TEST_START, len(data)))
        if "txn_idx" not in st.session_state:
            st.session_state["txn_idx"] = TEST_START
        base = data.iloc[st.session_state["txn_idx"]]
        show_truth = st.checkbox("Show true label", value=True)
        st.caption(f"Sampling from the held-out test slice "
                   f"(rows {TEST_START:,}–{len(data) - 1:,}) — the champion never saw these.")

        amount = st.slider("Amount ($)", 0.0, 5000.0, float(base["Amount"]), 1.0)
        time_s = st.slider("Time since first txn (s)", 0, 172792, int(base["Time"]))
        hour = int((time_s // 3600) % 24)
        st.markdown(f"**Hour of day:** {hour}:00")

        with st.expander("PCA features (V1–V28)", expanded=False):
            vvals = {}
            for i in range(1, 29):
                col = f"V{i}"
                vvals[col] = st.number_input(col, value=float(base[col]),
                                             format="%.4f", key=col)

        txn = {"Time": time_s, "Amount": amount, **vvals}
        x_row = engineer(txn)

    with right:
        expl, proba = fraud_shap(model, x_row)
        verdict = "🚨 FRAUD ALERT" if proba >= THRESHOLD else "✅ Legitimate"
        st.metric("Fraud probability", f"{proba * 100:.2f}%")
        st.markdown(f"### {verdict}")
        st.progress(min(proba, 1.0))
        st.caption(f"decision threshold = {THRESHOLD:.4f} (validation-tuned)")
        if show_truth:
            true_label = int(labels.iloc[st.session_state["txn_idx"]])
            st.info(f"True label: {'FRAUD' if true_label else 'legit'}")

        st.subheader("Why this decision")
        st.dataframe(top_drivers(expl), use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("SHAP waterfall")
    st.caption("How each feature pushed the score from the baseline toward fraud/legit.")
    st.pyplot(waterfall_fig(expl))

with tab_batch:
    st.subheader("Score a CSV of transactions")
    st.caption("Required columns: Time, V1–V28, Amount (Class optional).")
    up = st.file_uploader("Upload CSV", type="csv")
    if up is not None:
        batch = pd.read_csv(up)
        missing = [c for c in ["Time", "Amount", *[f"V{i}" for i in range(1, 29)]]
                   if c not in batch.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            order = FEATURE_ORDER if FEATURE_ORDER else MODEL_FEATURES
            Xb = add_features(batch)[order]
            probs = model.predict_proba(Xb)[:, 1]
            out = batch.copy()
            out["fraud_probability"] = probs.round(4)
            out["prediction"] = (probs >= THRESHOLD).astype(int)
            flagged = int(out["prediction"].sum())
            c1, c2, c3 = st.columns(3)
            c1.metric("Transactions", len(out))
            c2.metric("Flagged", flagged)
            c3.metric("Flag rate", f"{100 * flagged / len(out):.2f}%")
            if "Class" in out.columns:
                from sklearn.metrics import average_precision_score

                ap = average_precision_score(out["Class"], probs)
                st.success(f"PR-AUC vs provided labels: **{ap:.4f}**")
            st.dataframe(out.head(50), use_container_width=True)
            st.download_button("Download scored CSV",
                               out.to_csv(index=False).encode(),
                               "scored_transactions.csv", "text/csv")

st.sidebar.header("Model card")
st.sidebar.markdown(
    """
    **Trained on**: Kaggle ULB creditcard (284,807 txns / 492 frauds)

    **Protocol**: time-based split, resampling on train only,
    threshold tuned on validation, test untouched

    **Test performance**: PR-AUC 0.795 · F1 0.848 · MCC 0.855
    (97.5% precision @ 75% recall, 1 false positive / 42,722 txns)

    **Explainer**: SHAP TreeExplainer (game-theoretic feature attribution)
    """)
