"""Binary classification metrics with class-wise breakdown and bootstrap CIs.

Positive class = 1 (pneumonia). ``y_score`` is the predicted probability of class 1.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def confusion(y_true, y_pred) -> dict:
    y_true, y_pred = np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)
    return {
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def _div(a, b):
    return float(a / b) if b else float("nan")


def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """ECE over equal-width bins of the predicted probability of the predicted class."""
    y_true, y_score = np.asarray(y_true), np.asarray(y_score, dtype=float)
    pred = (y_score >= 0.5).astype(int)
    conf = np.where(pred == 1, y_score, 1 - y_score)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        if m.any():
            ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def binary_metrics(y_true, y_score, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)
    if y_true.shape != y_score.shape:
        raise ValueError(f"shape mismatch {y_true.shape} vs {y_score.shape}")
    if len(y_true) == 0:
        raise ValueError("empty inputs")
    y_pred = (y_score >= threshold).astype(int)
    c = confusion(y_true, y_pred)
    sens = _div(c["tp"], c["tp"] + c["fn"])
    spec = _div(c["tn"], c["tn"] + c["fp"])
    ppv = _div(c["tp"], c["tp"] + c["fp"])
    npv = _div(c["tn"], c["tn"] + c["fn"])
    f1_pos = _div(2 * ppv * sens, ppv + sens) if not np.isnan(ppv + sens) else float("nan")
    f1_neg = _div(2 * npv * spec, npv + spec) if not np.isnan(npv + spec) else float("nan")
    both = len(np.unique(y_true)) == 2
    return {
        "n": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float((y_pred == y_true).mean()),
        "balanced_accuracy": float(np.nanmean([sens, spec])),
        "auroc": float(roc_auc_score(y_true, y_score)) if both else float("nan"),
        "auprc": float(average_precision_score(y_true, y_score)) if both else float("nan"),
        "macro_f1": float(np.nanmean([f1_pos, f1_neg])),
        "ece": expected_calibration_error(y_true, y_score),
        "confusion": c,
        "per_class": {
            "normal": {"recall(specificity)": spec, "precision(npv)": npv, "f1": f1_neg,
                       "support": int((y_true == 0).sum())},
            "pneumonia": {"recall(sensitivity)": sens, "precision(ppv)": ppv, "f1": f1_pos,
                          "support": int((y_true == 1).sum())},
        },
    }


def youden_threshold(y_true, y_score) -> float:
    """Threshold maximising sensitivity + specificity - 1 (choose on validation data only)."""
    fpr, tpr, thr = roc_curve(np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float))
    j = tpr - fpr
    t = float(thr[int(np.argmax(j))])
    return float(min(max(t, 0.0), 1.0))


def paired_bootstrap_auroc(y_true, score_a, score_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """AUROC(a) - AUROC(b) on the same cases, with a percentile 95% CI from resampling cases.

    ``p_two_sided`` is the share of resamples on the other side of zero, doubled (a rough
    bootstrap p-value; prefer the CI).
    """
    y, a, b = np.asarray(y_true).astype(int), np.asarray(score_a, float), np.asarray(score_b, float)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], a[idx]) - roc_auc_score(y[idx], b[idx]))
    d = np.array(diffs)
    obs = float(roc_auc_score(y, a) - roc_auc_score(y, b))
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"auroc_diff": obs, "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p_two_sided": float(min(1.0, p)), "n_boot": int(len(d))}


def bootstrap_ci(y_true, y_score, threshold: float = 0.5, n_boot: int = 1000, seed: int = 0,
                 keys=("accuracy", "balanced_accuracy", "auroc", "auprc", "macro_f1")) -> dict:
    """Percentile 95% CIs from resampling test cases with replacement."""
    y_true, y_score = np.asarray(y_true).astype(int), np.asarray(y_score, dtype=float)
    rng = np.random.default_rng(seed)
    vals = {k: [] for k in keys}
    n = len(y_true)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        m = binary_metrics(y_true[idx], y_score[idx], threshold)
        for k in keys:
            vals[k].append(m[k])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in vals.items() if v}
