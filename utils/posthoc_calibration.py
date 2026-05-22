"""Post-hoc calibration utilities for reasoning-aware risk scoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score
from sklearn.isotonic import IsotonicRegression


DEFAULT_REASONING_FEATURES = [
    "base_logit",
    "conclusion_logit",
    "conclusion_prob",
    "reasoning_mean_logit",
    "reasoning_min_logit",
    "reasoning_max_logit",
    "reasoning_std_logit",
    "reasoning_mean_prob",
    "reasoning_min_prob",
    "reasoning_max_prob",
    "reasoning_std_prob",
    "reasoning_var_prob",
    "reasoning_gap_prob",
    "base_minus_conclusion_logit",
    "reasoning_count",
    "neg_mean_ll",
    "neg_top1_logp",
    "neg_margin_logp",
    "n_hops",
    "reasoning_claim_count",
    "reasoning_label_mean",
    "reasoning_label_std",
    "reasoning_label_match_rate",
    "reasoning_label_gap_mean",
    "reasoning_label_majority_gap",
    "reasoning_label_valid_count",
    "reasoning_feature_gap_l2",
    "reasoning_feature_gap_mean_abs",
    "reasoning_feature_gap_max_abs",
    "reasoning_feature_gap_cosine",
    "reasoning_tokens_per_claim",
    "reasoning_active_tokens_per_claim",
    "sidecar_reasoning_tokens",
    "sidecar_reasoning_active_tokens",
    "sidecar_reasoning_mean_l2",
    "sidecar_reasoning_std_l2",
    "sidecar_reasoning_mean_abs",
    "sidecar_reasoning_std_abs",
]

VALID_POSTHOC_METHODS = (
    "platt_base",
    "temperature_scaling",
    "isotonic_regression",
    "reasoning_logistic",
    "reasoning_logistic_isotonic",
    "reasoning_logistic_blend",
    "binwise_hybrid",
)
VALID_POSTHOC_METHODS_SET = set(VALID_POSTHOC_METHODS)
REASONING_POSTHOC_METHODS = {
    "reasoning_logistic",
    "reasoning_logistic_isotonic",
    "reasoning_logistic_blend",
    "binwise_hybrid",
}

VALID_POSTHOC_FEATURE_MODES = {"auto", "compact", "full"}
DEFAULT_POSTHOC_FEATURE_MODE = "auto"
DEFAULT_MIN_SAMPLES_FOR_FULL_FEATURES = 1500
COMPACT_ONLY_REASONING_METHODS = {
    "reasoning_logistic_isotonic",
    "reasoning_logistic_blend",
}
ANCHOR_METHOD_ORDER = (
    "isotonic_regression",
    "platt_base",
    "temperature_scaling",
)
COMPACT_REASONING_LOGISTIC_C = 0.25


def _to_1d_float_array(value: Any) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=np.float64)
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return arr


def _resolve_feature_mode(
    feature_mode: str,
    n_samples: int,
    min_samples_for_full: int,
) -> tuple[str, str | None]:
    requested_mode = str(feature_mode or DEFAULT_POSTHOC_FEATURE_MODE).strip().lower()
    if requested_mode not in VALID_POSTHOC_FEATURE_MODES:
        raise ValueError(
            f"Unsupported post-hoc feature mode {requested_mode!r}. "
            f"Expected one of: {sorted(VALID_POSTHOC_FEATURE_MODES)}"
        )
    threshold = max(1, int(min_samples_for_full))
    if requested_mode == "auto":
        if int(n_samples) >= threshold:
            return "full", None
        return "compact", f"auto_compact_for_sample_count:{n_samples}<{threshold}"
    return requested_mode, None


def build_reasoning_feature_matrix_with_mode(
    *,
    sample_rows_meta: list[dict],
    sample_logits: np.ndarray,
    claim_probs: np.ndarray,
    claim_logits: np.ndarray,
    feature_mode: str = DEFAULT_POSTHOC_FEATURE_MODE,
    min_samples_for_full: int = DEFAULT_MIN_SAMPLES_FOR_FULL_FEATURES,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Build sample-level post-hoc features from sample + claim predictions."""
    n_samples = int(len(sample_rows_meta))
    requested_mode = str(feature_mode or DEFAULT_POSTHOC_FEATURE_MODE).strip().lower()
    effective_mode, mode_reason = _resolve_feature_mode(
        feature_mode=requested_mode,
        n_samples=n_samples,
        min_samples_for_full=min_samples_for_full,
    )
    include_dense_cached_features = effective_mode == "full"

    conclusion_dim = 0
    reasoning_dim = 0
    if include_dense_cached_features:
        for row in sample_rows_meta:
            conclusion_dim = max(conclusion_dim, int(_to_1d_float_array(row.get("conclusion_cached_feature_mean")).shape[0]))
            reasoning_dim = max(reasoning_dim, int(_to_1d_float_array(row.get("reasoning_cached_feature_mean")).shape[0]))

    feature_names = list(DEFAULT_REASONING_FEATURES)
    if include_dense_cached_features:
        feature_names.extend([f"conclusion_cached_f{i}" for i in range(conclusion_dim)])
        feature_names.extend([f"reasoning_cached_f{i}" for i in range(reasoning_dim)])

    features: list[list[float]] = []
    for sample_idx, row in enumerate(sample_rows_meta):
        claim_indices = row.get("claim_indices", []) or []
        claim_meta = row.get("claim_meta", []) or []

        reasoning_probs = []
        reasoning_logits = []
        conclusion_prob = None
        conclusion_logit = None

        for local_idx, claim_idx in enumerate(claim_indices):
            if claim_idx >= len(claim_probs):
                continue
            meta = claim_meta[local_idx] if local_idx < len(claim_meta) else {}
            claim_type = str(meta.get("claim_type", "unknown")).strip().lower()
            prob = float(claim_probs[claim_idx])
            logit = float(claim_logits[claim_idx])
            if claim_type == "reasoning":
                reasoning_probs.append(prob)
                reasoning_logits.append(logit)
            elif claim_type == "conclusion":
                conclusion_prob = prob
                conclusion_logit = logit

        if conclusion_prob is None:
            # Fallback: map sample logit to prob as a conclusion proxy.
            base_logit = float(sample_logits[sample_idx]) if sample_idx < len(sample_logits) else 0.0
            conclusion_logit = base_logit
            conclusion_prob = float(expit(base_logit))
        if not reasoning_probs:
            reasoning_probs = [float(conclusion_prob)]
            reasoning_logits = [float(conclusion_logit)]

        rp = np.asarray(reasoning_probs, dtype=np.float64)
        rl = np.asarray(reasoning_logits, dtype=np.float64)
        base_logit = float(sample_logits[sample_idx]) if sample_idx < len(sample_logits) else 0.0
        token_stats = row.get("token_stats", {}) or {}
        reasoning_stats = row.get("reasoning_feature_stats", {}) or {}
        reasoning_label_stats = row.get("reasoning_label_stats", {}) or {}
        reasoning_feature_gap_stats = row.get("reasoning_feature_gap_stats", {}) or {}
        reasoning_claim_count = float(row.get("reasoning_claim_count", 0))
        reasoning_tokens = float(reasoning_stats.get("tokens", np.nan))
        reasoning_active_tokens = float(reasoning_stats.get("active_tokens", np.nan))
        if reasoning_claim_count > 0 and np.isfinite(reasoning_tokens):
            reasoning_tokens_per_claim = reasoning_tokens / reasoning_claim_count
        else:
            reasoning_tokens_per_claim = float("nan")
        if reasoning_claim_count > 0 and np.isfinite(reasoning_active_tokens):
            reasoning_active_tokens_per_claim = reasoning_active_tokens / reasoning_claim_count
        else:
            reasoning_active_tokens_per_claim = float("nan")
        row_features = [
            base_logit,
            float(conclusion_logit),
            float(conclusion_prob),
            float(np.mean(rl)),
            float(np.min(rl)),
            float(np.max(rl)),
            float(np.std(rl)),
            float(np.mean(rp)),
            float(np.min(rp)),
            float(np.max(rp)),
            float(np.std(rp)),
            float(np.var(rp)),
            float(abs(float(conclusion_prob) - float(np.mean(rp)))),
            float(base_logit - float(conclusion_logit)),
            float(len(rp)),
            float(token_stats.get("neg_mean_ll", np.nan)),
            float(token_stats.get("neg_top1_logp", np.nan)),
            float(token_stats.get("neg_margin_logp", np.nan)),
            float(row.get("n_hops", 0)),
            reasoning_claim_count,
            float(reasoning_label_stats.get("mean", np.nan)),
            float(reasoning_label_stats.get("std", np.nan)),
            float(reasoning_label_stats.get("match_rate", np.nan)),
            float(reasoning_label_stats.get("label_gap_mean", np.nan)),
            float(reasoning_label_stats.get("majority_gap", np.nan)),
            float(reasoning_label_stats.get("verified_count", 0)),
            float(reasoning_feature_gap_stats.get("l2", np.nan)),
            float(reasoning_feature_gap_stats.get("mean_abs", np.nan)),
            float(reasoning_feature_gap_stats.get("max_abs", np.nan)),
            float(reasoning_feature_gap_stats.get("cosine", np.nan)),
            float(reasoning_tokens_per_claim),
            float(reasoning_active_tokens_per_claim),
            reasoning_tokens,
            reasoning_active_tokens,
            float(reasoning_stats.get("mean_l2", np.nan)),
            float(reasoning_stats.get("std_l2", np.nan)),
            float(reasoning_stats.get("mean_abs", np.nan)),
            float(reasoning_stats.get("std_abs", np.nan)),
        ]

        if include_dense_cached_features and conclusion_dim > 0:
            conclusion_vec = _to_1d_float_array(row.get("conclusion_cached_feature_mean"))
            padded = np.full((conclusion_dim,), np.nan, dtype=np.float64)
            n = min(conclusion_dim, conclusion_vec.shape[0])
            if n > 0:
                padded[:n] = conclusion_vec[:n]
            row_features.extend(padded.tolist())

        if include_dense_cached_features and reasoning_dim > 0:
            reasoning_vec = _to_1d_float_array(row.get("reasoning_cached_feature_mean"))
            padded = np.full((reasoning_dim,), np.nan, dtype=np.float64)
            n = min(reasoning_dim, reasoning_vec.shape[0])
            if n > 0:
                padded[:n] = reasoning_vec[:n]
            row_features.extend(padded.tolist())

        features.append(row_features)

    feature_meta = {
        "requested_feature_mode": requested_mode,
        "effective_feature_mode": effective_mode,
        "feature_mode_reason": mode_reason,
        "include_dense_cached_features": bool(include_dense_cached_features),
        "min_samples_for_full": max(1, int(min_samples_for_full)),
        "n_samples": n_samples,
        "conclusion_dense_dim": int(conclusion_dim if include_dense_cached_features else 0),
        "reasoning_dense_dim": int(reasoning_dim if include_dense_cached_features else 0),
    }
    return np.asarray(features, dtype=np.float64), feature_names, feature_meta


def _impute_with_means(X: np.ndarray, means: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=np.float64)
    if means is None:
        cols = []
        for col in range(X.shape[1]):
            values = X[:, col]
            finite = values[np.isfinite(values)]
            cols.append(float(finite.mean()) if finite.size else 0.0)
        means = np.asarray(cols, dtype=np.float64)
    X_clean = X.copy()
    for col in range(X_clean.shape[1]):
        mask = ~np.isfinite(X_clean[:, col])
        if np.any(mask):
            X_clean[mask, col] = means[col]
    return X_clean, means


def _fit_logistic(X: np.ndarray, y: np.ndarray, *, c: float = 1.0) -> dict[str, Any]:
    clf = LogisticRegression(max_iter=4000, C=float(max(c, 1e-4)))
    clf.fit(X, y.astype(int))
    return {
        "coef": clf.coef_[0].astype(np.float64).tolist(),
        "intercept": float(clf.intercept_[0]),
        "regularization_c": float(max(c, 1e-4)),
    }


def _predict_logistic(params: dict[str, Any], X: np.ndarray) -> np.ndarray:
    coef = np.asarray(params["coef"], dtype=np.float64)
    intercept = float(params["intercept"])
    logits = X @ coef + intercept
    return expit(logits)


def _base_error_probs(base_logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    temperature = float(max(float(temperature), 1e-6))
    probs = 1.0 - expit(np.asarray(base_logits, dtype=np.float64) / temperature)
    return np.clip(probs.astype(np.float64), 1e-8, 1.0 - 1e-8)


def _fit_temperature_scaling(base_logits: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int64)
    logits = np.asarray(base_logits, dtype=np.float64).reshape(-1)
    best_t = 1.0
    best_nll = float("inf")
    for t in np.arange(0.05, 5.01, 0.05):
        probs = _base_error_probs(logits, temperature=float(t))
        try:
            nll = float(log_loss(labels, probs, labels=[0, 1]))
        except ValueError:
            continue
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return {
        "temperature": float(best_t),
        "objective": "nll",
        "best_nll": float(best_nll),
    }


def _predict_temperature_scaling(params: dict[str, Any], base_logits: np.ndarray) -> np.ndarray:
    return _base_error_probs(base_logits, temperature=float(params.get("temperature", 1.0)))


def _fit_isotonic_regression(base_logits: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    base_risk = _base_error_probs(base_logits, temperature=1.0)
    model = IsotonicRegression(y_min=1e-8, y_max=1.0 - 1e-8, out_of_bounds="clip")
    model.fit(base_risk.astype(np.float64), np.asarray(y, dtype=np.float64))
    return {
        "x_thresholds": np.asarray(model.X_thresholds_, dtype=np.float64).tolist(),
        "y_thresholds": np.asarray(model.y_thresholds_, dtype=np.float64).tolist(),
        "input_space": "base_error_probability",
    }


def _predict_isotonic_regression(params: dict[str, Any], base_logits: np.ndarray) -> np.ndarray:
    base_risk = _base_error_probs(base_logits, temperature=1.0)
    x_thresholds = np.asarray(params.get("x_thresholds", []), dtype=np.float64)
    y_thresholds = np.asarray(params.get("y_thresholds", []), dtype=np.float64)
    if x_thresholds.size == 0 or y_thresholds.size == 0:
        raise ValueError("Isotonic calibrator is missing threshold arrays.")
    if x_thresholds.size == 1:
        out = np.full(base_risk.shape, float(y_thresholds[0]), dtype=np.float64)
    else:
        out = np.interp(
            base_risk,
            x_thresholds,
            y_thresholds,
            left=float(y_thresholds[0]),
            right=float(y_thresholds[-1]),
        )
    return np.clip(out.astype(np.float64), 1e-8, 1.0 - 1e-8)


def _logit_from_probs(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probs, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    return np.log(clipped / (1.0 - clipped))


def _fit_isotonic_on_error_probs(error_probs: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    model = IsotonicRegression(y_min=1e-8, y_max=1.0 - 1e-8, out_of_bounds="clip")
    model.fit(np.asarray(error_probs, dtype=np.float64), np.asarray(y, dtype=np.float64))
    return {
        "x_thresholds": np.asarray(model.X_thresholds_, dtype=np.float64).tolist(),
        "y_thresholds": np.asarray(model.y_thresholds_, dtype=np.float64).tolist(),
        "input_space": "reasoning_error_probability",
    }


def _predict_isotonic_on_error_probs(params: dict[str, Any], error_probs: np.ndarray) -> np.ndarray:
    probs = np.clip(np.asarray(error_probs, dtype=np.float64), 1e-8, 1.0 - 1e-8)
    x_thresholds = np.asarray(params.get("x_thresholds", []), dtype=np.float64)
    y_thresholds = np.asarray(params.get("y_thresholds", []), dtype=np.float64)
    if x_thresholds.size == 0 or y_thresholds.size == 0:
        raise ValueError("Isotonic calibrator is missing threshold arrays.")
    if x_thresholds.size == 1:
        out = np.full(probs.shape, float(y_thresholds[0]), dtype=np.float64)
    else:
        out = np.interp(
            probs,
            x_thresholds,
            y_thresholds,
            left=float(y_thresholds[0]),
            right=float(y_thresholds[-1]),
        )
    return np.clip(out.astype(np.float64), 1e-8, 1.0 - 1e-8)


def _binary_ece(labels: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), 1e-8, 1.0 - 1e-8)
    if labels.size == 0:
        return float("nan")
    confidences = np.where(probs >= 0.5, probs, 1.0 - probs)
    predictions = (probs >= 0.5).astype(np.int64)
    accuracies = (predictions == labels).astype(np.float64)
    boundaries = np.linspace(0.0, 1.0, int(max(1, n_bins)) + 1)
    total = float(labels.size)
    ece = 0.0
    for i in range(len(boundaries) - 1):
        lo, hi = boundaries[i], boundaries[i + 1]
        in_bin = (confidences > lo) & (confidences <= hi)
        count = float(in_bin.sum())
        if count <= 0:
            continue
        avg_conf = float(confidences[in_bin].mean())
        avg_acc = float(accuracies[in_bin].mean())
        ece += (count / total) * abs(avg_acc - avg_conf)
    return float(ece)


def _brier_score(labels: np.ndarray, probs: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probs = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), 1e-8, 1.0 - 1e-8)
    if labels.size == 0:
        return float("nan")
    return float(np.mean((labels - probs) ** 2))


def _auc_metrics(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), 1e-8, 1.0 - 1e-8)
    if labels.size == 0 or len(np.unique(labels)) < 2:
        return float("nan"), float("nan")
    return (
        float(average_precision_score(labels, probs)),
        float(roc_auc_score(labels, probs)),
    )


def _candidate_metrics(labels: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), 1e-8, 1.0 - 1e-8)
    pr_auc, roc_auc = _auc_metrics(labels, probs)
    return {
        "ece": _binary_ece(labels, probs, n_bins=15),
        "nll": float(log_loss(labels, probs, labels=[0, 1])),
        "brier": _brier_score(labels, probs),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def _calibration_objective(metrics: dict[str, float]) -> float:
    return (
        float(metrics.get("ece", float("inf")))
        + 0.10 * float(metrics.get("nll", float("inf")))
        + 0.10 * float(metrics.get("brier", float("inf")))
    )


def _blend_selection_objective(metrics: dict[str, float]) -> float:
    """Keep calibration primary while mildly rewarding ranking quality."""
    objective = _calibration_objective(metrics)
    pr_auc = float(metrics.get("pr_auc", float("nan")))
    roc_auc = float(metrics.get("roc_auc", float("nan")))
    if np.isfinite(pr_auc):
        objective -= 0.02 * pr_auc
    if np.isfinite(roc_auc):
        objective -= 0.01 * roc_auc
    return float(objective)


def _compact_reasoning_indices(feature_names: list[str]) -> list[int]:
    indices: list[int] = []
    for idx, name in enumerate(feature_names):
        if str(name).startswith("conclusion_cached_f") or str(name).startswith("reasoning_cached_f"):
            continue
        indices.append(idx)
    return indices


def _select_feature_indices(
    X: np.ndarray,
    feature_names: list[str],
    *,
    mode: str,
) -> tuple[np.ndarray, list[str], list[int]]:
    if mode not in COMPACT_ONLY_REASONING_METHODS:
        indices = list(range(X.shape[1]))
        return X, list(feature_names), indices
    indices = _compact_reasoning_indices(feature_names)
    if not indices:
        indices = list(range(X.shape[1]))
    return X[:, indices], [feature_names[i] for i in indices], indices


def _fit_base_anchor_candidate(mode: str, base_logits: np.ndarray, y: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    base_logits = np.asarray(base_logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(y, dtype=np.int64).reshape(-1)
    if mode == "platt_base":
        model = _fit_logistic(base_logits.reshape(-1, 1), labels)
        return model, _predict_logistic(model, base_logits.reshape(-1, 1))
    if mode == "temperature_scaling":
        model = _fit_temperature_scaling(base_logits, labels)
        return model, _predict_temperature_scaling(model, base_logits)
    if mode == "isotonic_regression":
        model = _fit_isotonic_regression(base_logits, labels)
        return model, _predict_isotonic_regression(model, base_logits)
    raise ValueError(f"Unsupported base anchor mode {mode!r}")


def _predict_base_anchor_candidate(mode: str, model: dict[str, Any], base_logits: np.ndarray) -> np.ndarray:
    base_logits = np.asarray(base_logits, dtype=np.float64).reshape(-1)
    if mode == "platt_base":
        return _predict_logistic(model, base_logits.reshape(-1, 1))
    if mode == "temperature_scaling":
        return _predict_temperature_scaling(model, base_logits)
    if mode == "isotonic_regression":
        return _predict_isotonic_regression(model, base_logits)
    raise ValueError(f"Unsupported base anchor mode {mode!r}")


def _fit_best_base_anchor(
    base_logits: np.ndarray,
    y: np.ndarray,
    *,
    eval_base_logits: np.ndarray | None = None,
    eval_y: np.ndarray | None = None,
) -> dict[str, Any]:
    best_candidate = None
    best_objective = float("inf")
    score_logits = np.asarray(eval_base_logits if eval_base_logits is not None else base_logits, dtype=np.float64).reshape(-1)
    score_labels = np.asarray(eval_y if eval_y is not None else y, dtype=np.int64).reshape(-1)
    for mode in ANCHOR_METHOD_ORDER:
        model, probs = _fit_base_anchor_candidate(mode, base_logits, y)
        eval_probs = _predict_base_anchor_candidate(mode, model, score_logits)
        metrics = _candidate_metrics(score_labels, eval_probs)
        objective = _calibration_objective(metrics)
        candidate = {
            "mode": mode,
            "model": model,
            "metrics": metrics,
            "objective": float(objective),
        }
        if objective < best_objective:
            best_candidate = candidate
            best_objective = float(objective)
    if best_candidate is None:
        raise RuntimeError("Failed to fit any base anchor candidate.")
    return best_candidate


def _fit_reasoning_bin_assignment(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[dict[str, Any], np.ndarray]:
    gate_model = _fit_logistic(X, y, c=COMPACT_REASONING_LOGISTIC_C)
    scores = _predict_logistic(gate_model, X)
    n_samples = int(len(scores))
    target_bins = 3 if n_samples >= 3000 else 2
    min_bin_samples = max(96, n_samples // max(6, 2 * target_bins))
    chosen_edges = None
    chosen_bin_ids = None
    for n_bins in range(target_bins, 1, -1):
        quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
        edges = np.asarray(np.quantile(scores, quantiles), dtype=np.float64)
        edges = np.unique(edges)
        if edges.size != n_bins - 1:
            continue
        bin_ids = np.digitize(scores, edges, right=False)
        counts = np.bincount(bin_ids, minlength=n_bins)
        if counts.min() >= min_bin_samples:
            chosen_edges = edges
            chosen_bin_ids = bin_ids
            break
    if chosen_edges is None or chosen_bin_ids is None:
        median = float(np.median(scores))
        chosen_edges = np.asarray([median], dtype=np.float64)
        chosen_bin_ids = np.digitize(scores, chosen_edges, right=False)
    return {
        "gate_model": gate_model,
        "bin_edges": chosen_edges.astype(np.float64).tolist(),
        "n_bins": int(len(chosen_edges) + 1),
    }, chosen_bin_ids.astype(np.int64)


def _assign_reasoning_bins(bin_assigner: dict[str, Any], X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = _predict_logistic(bin_assigner["gate_model"], X)
    edges = np.asarray(bin_assigner.get("bin_edges", []), dtype=np.float64)
    if edges.size == 0:
        return scores, np.zeros(scores.shape[0], dtype=np.int64)
    return scores, np.digitize(scores, edges, right=False).astype(np.int64)


def _fit_reasoning_conditioned_isotonic(
    base_logits: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    bin_assigner, bin_ids = _fit_reasoning_bin_assignment(X, y)
    n_bins = int(bin_assigner["n_bins"])
    bin_models: list[dict[str, Any]] = []
    global_model = _fit_isotonic_regression(base_logits, y)
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        if int(mask.sum()) < 32 or len(np.unique(np.asarray(y)[mask])) < 2:
            bin_models.append(global_model)
            continue
        bin_models.append(_fit_isotonic_regression(np.asarray(base_logits)[mask], np.asarray(y)[mask]))
    return {
        "bin_assigner": bin_assigner,
        "bin_models": bin_models,
    }


def _predict_reasoning_conditioned_isotonic(
    params: dict[str, Any],
    base_logits: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    _, bin_ids = _assign_reasoning_bins(params["bin_assigner"], X)
    out = np.zeros(np.asarray(base_logits).shape[0], dtype=np.float64)
    for bin_idx, model in enumerate(params["bin_models"]):
        mask = bin_ids == bin_idx
        if np.any(mask):
            out[mask] = _predict_isotonic_regression(model, np.asarray(base_logits)[mask])
    return np.clip(out, 1e-8, 1.0 - 1e-8)


def _fit_reasoning_guided_blend(
    base_logits: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    anchor = _fit_best_base_anchor(base_logits, y)
    anchor_probs = _predict_base_anchor_candidate(anchor["mode"], anchor["model"], base_logits)
    reasoning_model = _fit_logistic(X, y, c=COMPACT_REASONING_LOGISTIC_C)
    reasoning_probs = _predict_logistic(reasoning_model, X)
    n_samples = int(len(reasoning_probs))
    target_bins = 3 if n_samples >= 3000 else 2
    edges = np.asarray([], dtype=np.float64)
    for n_bins in range(target_bins, 1, -1):
        candidate_edges = np.asarray(
            np.quantile(reasoning_probs, np.linspace(0.0, 1.0, n_bins + 1)[1:-1]),
            dtype=np.float64,
        )
        candidate_edges = np.unique(candidate_edges)
        if candidate_edges.size != n_bins - 1:
            continue
        candidate_bins = np.digitize(reasoning_probs, candidate_edges, right=False).astype(np.int64)
        counts = np.bincount(candidate_bins, minlength=n_bins)
        if counts.min() >= max(64, n_samples // (3 * n_bins)):
            edges = candidate_edges
            break
    if edges.size == 0 and n_samples > 0:
        edges = np.asarray([float(np.median(reasoning_probs))], dtype=np.float64)
    bin_ids = np.digitize(reasoning_probs, edges, right=False).astype(np.int64)
    n_bins = int(edges.size + 1)
    alphas: list[float] = []
    blended_raw = np.zeros_like(reasoning_probs, dtype=np.float64)
    for bin_idx in range(n_bins):
        mask = bin_ids == bin_idx
        local_reasoning = reasoning_probs[mask]
        local_anchor = anchor_probs[mask]
        local_y = np.asarray(y)[mask]
        if local_y.size == 0:
            alphas.append(0.0)
            continue
        best_alpha = 0.0
        best_probs = np.clip(local_anchor, 1e-8, 1.0 - 1e-8)
        best_objective = _blend_selection_objective(_candidate_metrics(local_y, best_probs))
        for alpha in np.linspace(0.0, 1.0, 21):
            candidate = np.clip(alpha * local_reasoning + (1.0 - alpha) * local_anchor, 1e-8, 1.0 - 1e-8)
            objective = _blend_selection_objective(_candidate_metrics(local_y, candidate))
            if objective < best_objective:
                best_alpha = float(alpha)
                best_probs = candidate
                best_objective = float(objective)
        alphas.append(float(best_alpha))
        blended_raw[mask] = best_probs
    final_isotonic = _fit_isotonic_on_error_probs(blended_raw, y)
    return {
        "anchor_mode": anchor["mode"],
        "anchor_model": anchor["model"],
        "reasoning_model": reasoning_model,
        "bin_edges": edges.astype(np.float64).tolist(),
        "n_bins": int(n_bins),
        "bin_alphas": [float(x) for x in alphas],
        "final_isotonic_model": final_isotonic,
    }


def _predict_reasoning_guided_blend(
    params: dict[str, Any],
    base_logits: np.ndarray,
    X: np.ndarray,
) -> np.ndarray:
    anchor_probs = _predict_base_anchor_candidate(
        str(params.get("anchor_mode", "platt_base")),
        params["anchor_model"],
        base_logits,
    )
    reasoning_probs = _predict_logistic(params["reasoning_model"], X)
    edges = np.asarray(params.get("bin_edges", []), dtype=np.float64)
    bin_ids = np.digitize(reasoning_probs, edges, right=False).astype(np.int64)
    alphas = [float(x) for x in params.get("bin_alphas", [0.1, 0.1])]
    raw = np.zeros_like(reasoning_probs, dtype=np.float64)
    for bin_idx in range(max(len(alphas), 1)):
        mask = bin_ids == bin_idx
        if not np.any(mask):
            continue
        alpha = float(alphas[min(bin_idx, len(alphas) - 1)])
        raw[mask] = np.clip(alpha * reasoning_probs[mask] + (1.0 - alpha) * anchor_probs[mask], 1e-8, 1.0 - 1e-8)
    return _predict_isotonic_on_error_probs(params["final_isotonic_model"], raw)


def _feature_mean(X: np.ndarray, feature_names: list[str], feature_name: str) -> float:
    if feature_name not in feature_names:
        return float("nan")
    col = feature_names.index(feature_name)
    values = X[:, col]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(finite.mean())


def _calibration_fit_meta(X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> dict[str, Any]:
    labels = np.asarray(y, dtype=np.int64)
    class_counts = np.bincount(labels, minlength=2)
    return {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]) if X.ndim == 2 else 0,
        "class_counts": [int(x) for x in class_counts.tolist()],
        "mean_reasoning_count": _feature_mean(X, feature_names, "reasoning_count"),
        "mean_reasoning_claim_count": _feature_mean(X, feature_names, "reasoning_claim_count"),
    }


def _resolve_effective_mode(requested_mode: str, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> tuple[str, str | None, dict[str, Any]]:
    mode = str(requested_mode or "reasoning_logistic").strip().lower()
    if mode not in VALID_POSTHOC_METHODS_SET:
        raise ValueError(
            f"Unsupported post-hoc method {mode!r}. Expected one of: {list(VALID_POSTHOC_METHODS)}"
        )
    fit_meta = _calibration_fit_meta(X, y, feature_names)
    reasons: list[str] = []

    n_samples = int(fit_meta["n_samples"])
    n_features = int(fit_meta["n_features"])
    feature_to_sample_ratio = float(n_features) / float(max(n_samples, 1))
    class_counts = fit_meta["class_counts"]
    min_class = min(class_counts) if class_counts else 0
    mean_reasoning_count = float(fit_meta["mean_reasoning_count"])
    mean_reasoning_claim_count = float(fit_meta["mean_reasoning_claim_count"])
    fit_meta["feature_to_sample_ratio"] = feature_to_sample_ratio

    if mode in {"reasoning_logistic", "binwise_hybrid"} and min_class < 24:
        reasons.append(f"minority_class_too_small:{min_class}")

    if mode == "reasoning_logistic":
        if feature_to_sample_ratio > 8.0:
            reasons.append(f"feature_to_sample_ratio_too_high:{feature_to_sample_ratio:.2f}>8.00")
        if np.isfinite(mean_reasoning_count) and np.isfinite(mean_reasoning_claim_count):
            if mean_reasoning_count <= 1.05 and mean_reasoning_claim_count <= 1.05:
                reasons.append("reasoning_signal_too_weak")

    if mode == "binwise_hybrid" and n_samples < 96:
        reasons.append(f"too_few_samples_for_binwise:{n_samples}<96")

    if reasons:
        fit_meta["requested_mode"] = mode
        fit_meta["effective_mode"] = "platt_base"
        return "platt_base", ";".join(reasons), fit_meta

    fit_meta["requested_mode"] = mode
    fit_meta["effective_mode"] = mode
    return mode, None, fit_meta


def fit_posthoc_calibrator(
    *,
    mode: str,
    X_tune: np.ndarray,
    y_tune: np.ndarray,
    feature_names: list[str],
    bin_threshold: float = 0.2,
) -> dict[str, Any]:
    """Fit a post-hoc calibrator for hallucination risk (positive class = error)."""
    requested_mode = str(mode or "reasoning_logistic").strip().lower()
    X_tune = np.asarray(X_tune, dtype=np.float64)
    y_tune = np.asarray(y_tune, dtype=np.int64)

    X_clean, means = _impute_with_means(X_tune)
    base_col = feature_names.index("base_logit") if "base_logit" in feature_names else 0
    mode, fallback_reason, fit_meta = _resolve_effective_mode(requested_mode, X_clean, y_tune, feature_names)
    selected_X, selected_feature_names, selected_feature_indices = _select_feature_indices(
        X_clean,
        feature_names,
        mode=mode,
    )
    selected_means = means[selected_feature_indices] if len(selected_feature_indices) != len(means) else means

    if mode == "platt_base":
        params = _fit_logistic(X_clean[:, [base_col]], y_tune)
        return {
            "mode": mode,
            "requested_mode": requested_mode,
            "fallback_reason": fallback_reason,
            "fit_meta": fit_meta,
            "feature_names": feature_names,
            "impute_means": means.tolist(),
            "base_col": int(base_col),
            "model": params,
        }

    if mode == "temperature_scaling":
        params = _fit_temperature_scaling(X_clean[:, base_col], y_tune)
        return {
            "mode": mode,
            "requested_mode": requested_mode,
            "fallback_reason": fallback_reason,
            "fit_meta": fit_meta,
            "feature_names": feature_names,
            "impute_means": means.tolist(),
            "base_col": int(base_col),
            "model": params,
        }

    if mode == "isotonic_regression":
        params = _fit_isotonic_regression(X_clean[:, base_col], y_tune)
        return {
            "mode": mode,
            "requested_mode": requested_mode,
            "fallback_reason": fallback_reason,
            "fit_meta": fit_meta,
            "feature_names": feature_names,
            "impute_means": means.tolist(),
            "base_col": int(base_col),
            "model": params,
        }

    if mode == "reasoning_logistic_isotonic":
        model = _fit_reasoning_conditioned_isotonic(X_clean[:, base_col], selected_X, y_tune)
        return {
            "mode": mode,
            "requested_mode": requested_mode,
            "fallback_reason": fallback_reason,
            "fit_meta": fit_meta,
            "feature_names": feature_names,
            "selected_feature_names": selected_feature_names,
            "selected_feature_indices": [int(i) for i in selected_feature_indices],
            "impute_means": means.tolist(),
            "selected_impute_means": selected_means.tolist(),
            "base_col": int(base_col),
            "model": model,
        }

    if mode == "reasoning_logistic_blend":
        model = _fit_reasoning_guided_blend(X_clean[:, base_col], selected_X, y_tune)
        return {
            "mode": mode,
            "requested_mode": requested_mode,
            "fallback_reason": fallback_reason,
            "fit_meta": fit_meta,
            "feature_names": feature_names,
            "selected_feature_names": selected_feature_names,
            "selected_feature_indices": [int(i) for i in selected_feature_indices],
            "impute_means": means.tolist(),
            "selected_impute_means": selected_means.tolist(),
            "base_col": int(base_col),
            "model": model,
        }

    if mode == "binwise_hybrid":
        base_risk = expit(-X_clean[:, base_col])
        lo_mask = base_risk < float(bin_threshold)
        hi_mask = ~lo_mask

        def _fit_subset(mask: np.ndarray) -> dict[str, Any]:
            if mask.sum() < 32 or len(np.unique(y_tune[mask])) < 2:
                return _fit_logistic(X_clean, y_tune)
            return _fit_logistic(X_clean[mask], y_tune[mask])

        return {
            "mode": mode,
            "requested_mode": requested_mode,
            "fallback_reason": fallback_reason,
            "fit_meta": fit_meta,
            "feature_names": feature_names,
            "impute_means": means.tolist(),
            "base_col": int(base_col),
            "bin_threshold": float(bin_threshold),
            "low_model": _fit_subset(lo_mask),
            "high_model": _fit_subset(hi_mask),
        }

    # Default: reasoning_logistic
    params = _fit_logistic(X_clean, y_tune)
    return {
        "mode": "reasoning_logistic",
        "requested_mode": requested_mode,
        "fallback_reason": fallback_reason,
        "fit_meta": fit_meta,
        "feature_names": feature_names,
        "impute_means": means.tolist(),
        "base_col": int(base_col),
        "model": params,
    }


def apply_posthoc_calibrator(calibrator: dict[str, Any], X_eval: np.ndarray) -> np.ndarray:
    """Apply fitted calibrator and return calibrated error probabilities."""
    mode = str(calibrator.get("mode", "reasoning_logistic")).strip().lower()
    if mode not in VALID_POSTHOC_METHODS_SET:
        raise ValueError(
            f"Unsupported saved post-hoc calibrator mode {mode!r}. "
            f"Expected one of: {list(VALID_POSTHOC_METHODS)}"
        )
    means = np.asarray(calibrator.get("impute_means", []), dtype=np.float64)
    X_clean, _ = _impute_with_means(X_eval, means=means if means.size else None)
    base_col = int(calibrator.get("base_col", 0))
    selected_feature_indices = [int(i) for i in (calibrator.get("selected_feature_indices", []) or [])]

    if mode == "platt_base":
        return _predict_logistic(calibrator["model"], X_clean[:, [base_col]])

    if mode == "temperature_scaling":
        return _predict_temperature_scaling(calibrator["model"], X_clean[:, base_col])

    if mode == "isotonic_regression":
        return _predict_isotonic_regression(calibrator["model"], X_clean[:, base_col])

    if mode == "reasoning_logistic_isotonic":
        selected_X = X_clean[:, selected_feature_indices] if selected_feature_indices else X_clean
        return _predict_reasoning_conditioned_isotonic(calibrator["model"], X_clean[:, base_col], selected_X)

    if mode == "reasoning_logistic_blend":
        selected_X = X_clean[:, selected_feature_indices] if selected_feature_indices else X_clean
        return _predict_reasoning_guided_blend(calibrator["model"], X_clean[:, base_col], selected_X)

    if mode == "binwise_hybrid":
        base_risk = expit(-X_clean[:, base_col])
        lo_mask = base_risk < float(calibrator.get("bin_threshold", 0.2))
        out = np.zeros(X_clean.shape[0], dtype=np.float64)
        if np.any(lo_mask):
            out[lo_mask] = _predict_logistic(calibrator["low_model"], X_clean[lo_mask])
        if np.any(~lo_mask):
            out[~lo_mask] = _predict_logistic(calibrator["high_model"], X_clean[~lo_mask])
        return out

    return _predict_logistic(calibrator["model"], X_clean)


def save_posthoc_calibrator(calibrator: dict[str, Any], path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(calibrator, f, indent=2)


def load_posthoc_calibrator(path: str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
