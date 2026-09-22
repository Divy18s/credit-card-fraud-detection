"""Imbalance-aware evaluation: threshold selection and metric computation."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_threshold_f1(y_true, scores) -> float:
    """Pick the decision threshold maximizing F1 on the validation set."""
    p, r, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / (p + r + 1e-12)
    idx = int(np.argmax(f1[:-1]))
    return float(thr[idx])


def evaluate_scores(y_true, scores, threshold: float) -> dict:
    """Full metric suite on the original imbalanced distribution."""
    scores = np.asarray(scores)
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "threshold": round(threshold, 6),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, preds)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }
