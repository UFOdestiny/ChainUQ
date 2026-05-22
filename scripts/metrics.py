"""
metrics.py - Binary evaluation metrics for multi-hop reasoning UQ.

Provides:
  - Accuracy, Precision, Recall, F1
  - ECE (Expected Calibration Error)
  - MEC (alias of MCE in this codebase)
  - PR-AUC (Precision-Recall Area Under Curve)
  - ROC-AUC
  - Brier score, NLL
  - AURC (Area Under Risk-Coverage)
  - Risk@Coverage
"""

import numpy as np
from scipy.special import expit
from sklearn.metrics import (
    accuracy_score,
    auc,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Expected Calibration Error for binary classification.

    Args:
        labels:     (N,) binary labels (0 or 1)
        probs:      (N,) predicted probabilities for the positive class

    Returns:
        ECE value (lower is better)
    """
    confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
    predictions = (probs >= 0.5).astype(int)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total = len(labels)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        count = in_bin.sum()
        if count == 0:
            continue
        avg_confidence = confidences[in_bin].mean()
        avg_accuracy = accuracies[in_bin].mean()
        ece += (count / total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_mce(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Maximum Calibration Error for binary classification."""
    confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
    predictions = (probs >= 0.5).astype(int)
    accuracies = (predictions == labels).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    max_err = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        if not np.any(in_bin):
            continue
        avg_confidence = confidences[in_bin].mean()
        avg_accuracy = accuracies[in_bin].mean()
        max_err = max(max_err, abs(avg_accuracy - avg_confidence))
    return float(max_err)


def compute_mec(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """MEC alias (kept equal to MCE for compatibility with prior experiments)."""
    return compute_mce(labels, probs, n_bins=n_bins)


def compute_ace(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    """Adaptive Calibration Error with equal-mass confidence bins."""
    confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
    predictions = (probs >= 0.5).astype(int)
    accuracies = (predictions == labels).astype(float)
    n = len(labels)
    if n == 0:
        return float("nan")

    order = np.argsort(confidences)
    bins = np.array_split(order, max(1, n_bins))
    errs = []
    for idx in bins:
        if idx.size == 0:
            continue
        avg_confidence = confidences[idx].mean()
        avg_accuracy = accuracies[idx].mean()
        errs.append(abs(avg_accuracy - avg_confidence))
    if not errs:
        return float("nan")
    return float(np.mean(errs))


def compute_brier_decomposition(
    labels: np.ndarray,
    probs: np.ndarray,
    n_bins: int = 15,
) -> dict[str, float]:
    """Brier decomposition into reliability/resolution/uncertainty."""
    labels = labels.astype(np.int64)
    probs = probs.astype(np.float64)
    if labels.size == 0:
        return {
            "brier_reliability": float("nan"),
            "brier_resolution": float("nan"),
            "brier_uncertainty": float("nan"),
        }

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    p_bar = float(labels.mean())
    reliability = 0.0
    resolution = 0.0
    total = float(labels.size)
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        in_bin = (probs > lo) & (probs <= hi)
        count = float(in_bin.sum())
        if count <= 0:
            continue
        p_k = float(probs[in_bin].mean())
        o_k = float(labels[in_bin].mean())
        weight = count / total
        reliability += weight * (p_k - o_k) ** 2
        resolution += weight * (o_k - p_bar) ** 2

    uncertainty = p_bar * (1.0 - p_bar)
    return {
        "brier_reliability": float(reliability),
        "brier_resolution": float(resolution),
        "brier_uncertainty": float(uncertainty),
    }


def compute_binary_pr_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Binary PR-AUC."""
    if len(np.unique(labels)) < 2:
        return float("nan")
    precision, recall, _ = precision_recall_curve(labels, probs)
    return float(auc(recall, precision))


def _to_probs(logits_or_probs: np.ndarray) -> np.ndarray:
    scores = logits_or_probs.astype(np.float64)
    if scores.min() < 0 or scores.max() > 1:
        return expit(scores)
    return scores


def _risk_coverage_curve(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return coverage and selective risk curve points.

    Coverage increases from low to high by keeping highest-confidence samples first.
    """
    labels_int = labels.astype(int)
    confidence = np.maximum(probs, 1.0 - probs)
    order = np.argsort(-confidence)  # highest confidence first
    correct_sorted = (labels_int[order] == (probs[order] >= 0.5).astype(int)).astype(np.float64)

    n = len(labels_int)
    ks = np.arange(1, n + 1, dtype=np.int64)
    coverage = ks / float(n)
    cumulative_acc = np.cumsum(correct_sorted) / ks
    risk = 1.0 - cumulative_acc
    return coverage, risk


def compute_aurc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Area under selective risk-coverage curve."""
    if labels.size == 0:
        return float("nan")
    coverage, risk = _risk_coverage_curve(labels, probs)
    return float(np.trapz(risk, coverage))


def compute_risk_at_coverage(labels: np.ndarray, probs: np.ndarray, target_coverage: float) -> float:
    """Selective risk after retaining top-confidence target_coverage fraction."""
    if labels.size == 0:
        return float("nan")
    target_coverage = float(np.clip(target_coverage, 0.0, 1.0))
    n = len(labels)
    k = int(max(1, np.ceil(target_coverage * n)))
    confidence = np.maximum(probs, 1.0 - probs)
    order = np.argsort(-confidence)
    top_idx = order[:k]
    preds = (probs[top_idx] >= 0.5).astype(int)
    acc = (preds == labels[top_idx].astype(int)).mean()
    return float(1.0 - acc)


def compute_naurc(labels: np.ndarray, probs: np.ndarray) -> float:
    """Normalized AURC (excess AURC over oracle ordering)."""
    if labels.size == 0:
        return float("nan")
    labels_int = labels.astype(int)
    preds = (probs >= 0.5).astype(int)
    correct = (labels_int == preds).astype(np.float64)
    n = len(labels_int)
    ks = np.arange(1, n + 1, dtype=np.int64)
    coverage = ks / float(n)

    # Observed selective risk
    _, risk = _risk_coverage_curve(labels_int, probs)
    aurc = float(np.trapz(risk, coverage))

    # Oracle ordering: all correct first, then incorrect
    oracle_order = np.argsort(-correct)
    oracle_acc = np.cumsum(correct[oracle_order]) / ks
    oracle_risk = 1.0 - oracle_acc
    oracle_aurc = float(np.trapz(oracle_risk, coverage))
    return float(aurc - oracle_aurc)


def find_optimal_threshold(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    """Find the threshold that maximizes F1 score.

    Returns (optimal_threshold, optimal_f1).
    """
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return 0.5, 0.0
    best_th, best_f1 = 0.5, 0.0
    for th in np.arange(0.01, 0.99, 0.01):
        preds = (probs >= th).astype(int)
        f1_val = float(f1_score(labels, preds, zero_division=0))
        if f1_val > best_f1:
            best_f1 = f1_val
            best_th = float(th)
    return best_th, best_f1


def compute_all_metrics(
    labels: np.ndarray,
    logits_or_probs: np.ndarray,
    threshold: float = 0.5,
    predictions: np.ndarray | None = None,
) -> dict:
    """Compute all binary evaluation metrics.

    Args:
        labels:          (N,) binary labels (1=correct/non-hallucination, 0=hallucination)
        logits_or_probs: (N,) raw logits or probabilities.
                         If any value > 1 or < 0, treated as logits and
                         passed through sigmoid.

    Returns:
        dict with accuracy, precision, recall, f1, roc_auc, pr_auc, ece,
        plus optimal_threshold and optimal_f1.
    """
    nan_result = {k: float("nan") for k in [
        "accuracy", "precision", "recall", "f1", "roc_auc",
        "pr_auc", "ece", "mec", "mce", "ace", "nll", "brier", "aurc", "naurc",
        "brier_reliability", "brier_resolution", "brier_uncertainty",
        "risk_at_80_cov", "risk_at_90_cov", "threshold",
        "optimal_threshold", "optimal_f1",
    ]}

    # Filter out unknown/pending labels (-1)
    if predictions is not None:
        predictions = np.asarray(predictions, dtype=np.int64).reshape(-1)
    valid_mask = labels >= 0
    if valid_mask.sum() == 0:
        return nan_result
    if valid_mask.sum() < len(labels):
        labels = labels[valid_mask]
        logits_or_probs = logits_or_probs[valid_mask]
        if predictions is not None:
            predictions = predictions[valid_mask]

    probs = _to_probs(logits_or_probs)
    probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
    labels_int = labels.astype(int)

    if predictions is None:
        predictions = (probs >= threshold).astype(int)
    else:
        if predictions.shape[0] != labels_int.shape[0]:
            raise ValueError(
                f"Predictions length {predictions.shape[0]} does not match labels length {labels_int.shape[0]}."
            )

    has_both_classes = len(np.unique(labels_int)) >= 2

    try:
        nll = float(log_loss(labels_int, probs, labels=[0, 1]))
    except ValueError:
        nll = float("nan")
    try:
        brier = float(brier_score_loss(labels_int, probs))
    except ValueError:
        brier = float("nan")

    opt_th, opt_f1 = find_optimal_threshold(labels_int, probs)
    brier_parts = compute_brier_decomposition(labels_int, probs)

    return {
        "accuracy": float(accuracy_score(labels_int, predictions)),
        "precision": float(precision_score(labels_int, predictions, zero_division=0)),
        "recall": float(recall_score(labels_int, predictions, zero_division=0)),
        "f1": float(f1_score(labels_int, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels_int, probs)) if has_both_classes else float("nan"),
        "pr_auc": compute_binary_pr_auc(labels_int, probs),
        "ece": compute_ece(labels_int, probs),
        "mec": compute_mec(labels_int, probs),
        "mce": compute_mce(labels_int, probs),
        "ace": compute_ace(labels_int, probs),
        "nll": nll,
        "brier": brier,
        "aurc": compute_aurc(labels_int, probs),
        "naurc": compute_naurc(labels_int, probs),
        **brier_parts,
        "risk_at_80_cov": compute_risk_at_coverage(labels_int, probs, 0.8),
        "risk_at_90_cov": compute_risk_at_coverage(labels_int, probs, 0.9),
        "threshold": float(threshold),
        "optimal_threshold": opt_th,
        "optimal_f1": opt_f1,
    }
