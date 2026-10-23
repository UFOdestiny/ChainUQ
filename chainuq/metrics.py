"""Response-level UQ metrics (positive class = correct response).

Definitions follow the paper appendix: ECE uses the top-label confidence
max(p, 1-p) with the 0.5-threshold prediction and 15 equal-width bins; Risk@90
is the 0.5-threshold error rate on the 90% most confident responses; Accuracy
thresholds the score at a value fixed on the validation split.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ece(y, p, n_bins: int = 15) -> float:
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    conf = np.where(p >= 0.5, p, 1.0 - p)
    acc = ((p >= 0.5).astype(int) == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            out += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(out)


def brier(y, p) -> float:
    return float(np.mean((np.asarray(p, dtype=np.float64) - np.asarray(y)) ** 2))


def risk_at(y, p, coverage: float = 0.9) -> float:
    y, p = np.asarray(y), np.asarray(p, dtype=np.float64)
    conf = np.where(p >= 0.5, p, 1.0 - p)
    keep = np.argsort(-conf, kind="stable")[: int(np.ceil(coverage * len(y)))]
    return float(1.0 - np.mean((p[keep] >= 0.5).astype(int) == y[keep]))


def auroc(y, s) -> float:
    y = np.asarray(y)
    return float(roc_auc_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def prauc(y, s) -> float:
    y = np.asarray(y)
    return float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan")


def best_threshold(y_val, s_val) -> float:
    """Accuracy-maximising threshold on validation scores."""
    y_val, s_val = np.asarray(y_val), np.asarray(s_val, dtype=np.float64)
    cands = np.unique(s_val)
    if cands.size > 512:
        cands = np.quantile(s_val, np.linspace(0, 1, 513))
    accs = [np.mean((s_val >= t).astype(int) == y_val) for t in cands]
    return float(cands[int(np.argmax(accs))])


def discrimination(y, s, threshold: float) -> dict:
    y, s = np.asarray(y), np.asarray(s, dtype=np.float64)
    return {
        "acc": float(np.mean((s >= threshold).astype(int) == y)),
        "auroc": auroc(y, s),
        "prauc": prauc(y, s),
    }


def calibration(y, p) -> dict:
    return {"ece": ece(y, p), "brier": brier(y, p), "risk90": risk_at(y, p, 0.9)}
