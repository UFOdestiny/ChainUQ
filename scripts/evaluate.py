#!/usr/bin/env python3
"""
evaluate.py - Phase 3/4: Evaluate ChainUQ heads on cached features.

Runs supervised evaluation for a trained head, supports threshold transfer, and
optionally fits or applies post-hoc calibration on cached reasoning metadata.

Dataset-agnostic: reads dataset_name from the Phase 1 manifest and uses the
dataset class's eval_config + difficulty_field for per-difficulty breakdowns.

Usage:
    python scripts/evaluate.py --head_path /path/to/final_model --cache_dir /path/to/cache
"""

import os
import sys
import json
import argparse
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.special import expit
from sklearn.metrics import log_loss
from config import Config
from data.cached_features import CachedFeatureDataset
from data.datasets import get_dataset_cls
from models.heads import build_head
from models.wrapper import CachedFeatureModel
from scripts.metrics import (
    compute_all_metrics,
    find_optimal_threshold,
)
from utils.common import collate_claim_cached_features
from utils.efficiency import (
    get_cpu_peak_memory_gb,
    get_gpu_peak_memory_gb,
    reset_gpu_peak_memory,
)
from utils.log import WorkflowLogger, configure_logging, get_logger
from utils.pipeline import PipelineRunner, StageSpec
from utils.number_utils import first_number
from utils.reporting import write_predictions, write_report
from utils.contracts import EvalReport
from utils.threshold_policy import build_threshold_protocol, resolve_threshold_mode
from utils.posthoc_calibration import (
    apply_posthoc_calibrator,
    build_reasoning_feature_matrix_with_mode,
    fit_posthoc_calibrator,
    load_posthoc_calibrator,
    save_posthoc_calibrator,
    VALID_POSTHOC_METHODS,
)

log = get_logger(__name__)

@dataclass(frozen=True)
class EvalContext:
    dataset_name: str
    base_model_name: str
    difficulty_field: str


def _read_manifest(cache_dir: str, split: str) -> dict:
    """Read manifest.json from a split directory."""
    manifest_path = Path(cache_dir) / split / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r") as f:
        return json.load(f)


def _build_eval_context(cache_dir: str, split: str, cfg=None) -> EvalContext:
    """Resolve manifest-backed evaluation metadata once per cache/split."""
    manifest = _read_manifest(cache_dir, split)
    dataset_name = manifest.get("dataset_name", "unknown")
    model_path = manifest.get("model_path", "")
    if model_path:
        base_model_name = Path(model_path).name
    elif cfg is not None:
        base_model_name = Path(cfg.model.pretrained_model_name_or_path).name or "unknown"
    else:
        base_model_name = "unknown"

    difficulty_field = "n_hops"
    if dataset_name and dataset_name != "unknown":
        try:
            difficulty_field = get_dataset_cls(dataset_name).difficulty_field
        except ValueError:
            pass

    return EvalContext(
        dataset_name=dataset_name,
        base_model_name=base_model_name,
        difficulty_field=difficulty_field,
    )


def _load_existing_report(report_path: str, report_label: str):
    """Load an existing JSON report if it is present and non-empty."""
    if not os.path.exists(report_path):
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)
        if report:
            return report
        raise ValueError("empty report")
    except (json.JSONDecodeError, IOError, ValueError):
        log.warning("Existing %s report is corrupt — will re-evaluate.", report_label)
        return None


def _resolve_report_artifact_path(report_dir: Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(str(value))
    if not path.is_absolute():
        path = report_dir / path
    return str(path.resolve())


def _build_posthoc_status_payload(report: dict, report_path: str) -> dict:
    report_dir = Path(report_path).resolve().parent
    threshold_protocol = report.get("threshold_protocol", {}) or {}
    posthoc = report.get("posthoc", {}) or threshold_protocol.get("posthoc", {}) or {}
    split_name = str(report.get("split", "") or "").strip().lower()
    enabled = bool(posthoc.get("enabled", False))
    applied = bool(posthoc.get("applied", False))

    artifacts: dict[str, str] = {
        "evaluation_report": str(Path(report_path).resolve()),
    }
    diagnostics_path = _resolve_report_artifact_path(report_dir, report.get("diagnostics_path"))
    if diagnostics_path:
        artifacts["diagnostics"] = diagnostics_path
    predictions_path = _resolve_report_artifact_path(report_dir, report.get("predictions_path"))
    if predictions_path:
        artifacts["predictions"] = predictions_path
    calibrator_path = _resolve_report_artifact_path(report_dir, posthoc.get("calibrator_path"))
    if calibrator_path:
        artifacts["calibrator"] = calibrator_path

    required_artifacts = ["evaluation_report", "diagnostics"]
    if "predictions" in artifacts:
        required_artifacts.append("predictions")
    if split_name == "test" and enabled:
        required_artifacts.append("calibrator")

    return {
        "status": "complete",
        "report_path": str(Path(report_path).resolve()),
        "head_type": report.get("head_type"),
        "split": report.get("split"),
        "method_type": report.get("method_type"),
        "posthoc_enabled": enabled,
        "posthoc_applied": applied,
        "posthoc_mode": posthoc.get("mode"),
        "required_artifacts": required_artifacts,
        "artifacts": artifacts,
    }


def _write_posthoc_status(report: dict, report_path: str) -> str:
    status_payload = _build_posthoc_status_payload(report, report_path)
    status_path = str(Path(report_path).resolve().parent / "posthoc_status.json")
    write_report(status_path, status_payload, kind="posthoc_status")
    return status_path


def _load_thresholds_from_report(report_path: str, default_threshold: float):
    """Load transferred sample/claim thresholds from a previous evaluation report."""
    if not report_path or not os.path.exists(report_path):
        raise RuntimeError(f"Threshold source report not found: {report_path}")

    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    threshold_protocol = report.get("threshold_protocol", {}) or {}
    applied = threshold_protocol.get("applied_thresholds", {}) or {}
    sample_threshold = float(applied.get("sample", default_threshold))
    if not np.isfinite(sample_threshold):
        sample_threshold = float(default_threshold)
    difficulty_thresholds_raw = applied.get("difficulty", {}) or {}
    difficulty_thresholds = {}
    for key, value in difficulty_thresholds_raw.items():
        try:
            k = int(key)
            v = float(value)
            if np.isfinite(v):
                difficulty_thresholds[k] = v
        except (TypeError, ValueError):
            continue
    sample_temperature = float(applied.get("sample_temperature", 1.0))
    if not np.isfinite(sample_temperature) or sample_temperature <= 0:
        sample_temperature = 1.0
    source_meta = {
        "dataset_name": report.get("dataset_name"),
        "split": report.get("split"),
        "tune_split": threshold_protocol.get("tune_split"),
        "tune_sample_count": int(report.get("tune_sample_count", 0) or 0),
        "difficulty_field": report.get("difficulty_field"),
        "difficulty_thresholds": difficulty_thresholds,
        "sample_temperature": sample_temperature,
    }
    return sample_threshold, source_meta


def _load_threshold_from_train_results(head_path: str):
    """Load the validation-selected sample threshold stored during training."""
    if not head_path:
        return None, None

    train_results_path = Path(head_path).resolve().parent / "train_results.json"
    if not train_results_path.is_file():
        return None, None

    with train_results_path.open("r", encoding="utf-8") as f:
        train_results = json.load(f)

    eval_metrics = train_results.get("eval_metrics", {}) or {}
    sample_threshold = eval_metrics.get("eval_threshold", None)
    if sample_threshold is None:
        raise RuntimeError(
            f"train_results.json is missing eval_metrics.eval_threshold: {train_results_path}"
        )
    try:
        sample_threshold = float(sample_threshold)
    except (TypeError, ValueError):
        raise RuntimeError(
            f"Invalid eval_metrics.eval_threshold in {train_results_path}: {sample_threshold!r}"
        )
    if not np.isfinite(sample_threshold):
        raise RuntimeError(
            f"Non-finite eval_metrics.eval_threshold in {train_results_path}: {sample_threshold!r}"
        )

    source_meta = {
        "path": str(train_results_path),
        "sample_threshold": sample_threshold,
        "eval_metrics": eval_metrics,
    }
    return sample_threshold, source_meta


def _preview_applied_sample_threshold(args) -> tuple[float, str]:
    """Best-effort preview of the sample threshold shown in evaluation logs."""
    default_threshold = float(args.threshold)
    train_results_threshold, _ = _load_threshold_from_train_results(
        args.head_path,
    )
    mode = resolve_threshold_mode(args, train_results_threshold)

    if mode == "fit_on_eval_split":
        return default_threshold, f"sample fit on eval split={args.split}"

    if mode == "source_report":
        sample_threshold, _ = _load_thresholds_from_report(
            args.threshold_source_report,
            default_threshold=default_threshold,
        )
        return float(sample_threshold), "sample from source_report"

    if mode == "train_results" and train_results_threshold is not None:
        return float(train_results_threshold), "sample from train_results"

    if mode == "tuned_on_split":
        return default_threshold, f"default before tuning on {args.threshold_tune_split}"

    return default_threshold, "sample default/fixed"


def parse_args():
    posthoc_feature_mode_default = str(os.getenv("POSTHOC_FEATURE_MODE", "auto")).strip().lower()
    posthoc_min_samples_raw = os.getenv("POSTHOC_MIN_SAMPLES_FOR_FULL", "1500")
    try:
        posthoc_min_samples_default = int(posthoc_min_samples_raw)
    except ValueError:
        posthoc_min_samples_default = 1500
    parser = argparse.ArgumentParser(description="Evaluate uncertainty methods")
    parser.add_argument("--head_path", type=str, default=None,
                        help="Path to trained head directory (head_config.json + head_weights.pth)")
    parser.add_argument("--cache_dir", type=str, default=None,
                        help="Path to cached features from Phase 1")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=0)
    parser.add_argument("--n_hop_values", type=str, default=None,
                        help="Difficulty filter. Comma-sep ints.")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Decision threshold for binary classification")
    parser.add_argument("--threshold_tune_split", type=str, default="validation",
                        help="Split used for threshold calibration when no source report is provided")
    parser.add_argument("--threshold_source_report", type=str, default=None,
                        help="Evaluation report path to source thresholds/protocol from")
    parser.add_argument("--fit_thresholds_on_eval_split", action="store_true",
                        help="Fit thresholds directly on current eval split (disabled for test)")
    parser.add_argument("--require_tune_split", action="store_true",
                        help="Fail if tune split cannot be loaded when tuning is requested")
    parser.add_argument("--enable_temp_scaling", action="store_true",
                        help="Fit temperature scaling on tune/eval split according to threshold mode")
    parser.add_argument("--enable_difficulty_thresholds", action="store_true",
                        help="Fit per-difficulty thresholds on tune/eval split according to threshold mode")
    parser.add_argument("--difficulty_threshold_min_samples", type=int, default=100,
                        help="Minimum samples required for a difficulty-specific threshold")
    parser.add_argument("--enable_posthoc", action="store_true", default=True,
                        help="Enable post-hoc calibration (default: on)")
    parser.add_argument("--no_enable_posthoc", dest="enable_posthoc", action="store_false")
    parser.add_argument("--posthoc_method", type=str, default="reasoning_logistic",
                        choices=list(VALID_POSTHOC_METHODS),
                        help="Post-hoc calibrator method")
    parser.add_argument("--posthoc_tune_split", type=str, default="validation",
                        help="Split used to fit post-hoc calibrator")
    parser.add_argument("--posthoc_feature_mode", type=str, default=posthoc_feature_mode_default,
                        choices=["auto", "compact", "full"],
                        help="Feature regime used by post-hoc calibration")
    parser.add_argument("--posthoc_min_samples_for_full", type=int, default=posthoc_min_samples_default,
                        help="Minimum tuning samples required before auto mode enables full dense features")
    parser.add_argument("--posthoc_model_path", type=str, default=None,
                        help="Optional path to an existing post-hoc calibrator JSON file")
    parser.add_argument("--prediction_cache_dir", type=str, default=None,
                        help="Optional directory for reusable split-level prediction caches")
    parser.add_argument("--save_predictions", action="store_true", default=True,
                        help="Write per-sample prediction details")
    parser.add_argument("--no_save_predictions", dest="save_predictions", action="store_false")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing outputs")
    return parser.parse_args()


def _build_supervised_prediction_rows(
    sample_rows_meta,
    sample_labels,
    sample_preds,
    sample_probs,
    sample_logits,
    claim_labels,
    claim_preds,
    claim_probs,
    claim_logits,
    include_claims: bool = False,
):
    """Build per-sample prediction payload with both sample and claim diagnostics."""
    prediction_rows = []

    for sample in sample_rows_meta:
        sample_idx = int(sample.get("sample_id", -1))
        if sample_idx < 0 or sample_idx >= len(sample_labels):
            continue
        sample_entry = {
            "sample_id": sample_idx,
            "dataset_sample_id": int(sample.get("dataset_sample_id", -1)),
            "dataset_sample_uid": str(sample.get("dataset_sample_uid", "")),
            "question": sample.get("question", ""),
            "generated_text": sample.get("generated_text", ""),
            "n_hops": int(sample.get("n_hops", 0)),
            "sample_label": int(sample_labels[sample_idx]),
            "sample_prediction": int(sample_preds[sample_idx]),
            "sample_probability": float(sample_probs[sample_idx]),
            "sample_logit": float(sample_logits[sample_idx]),
            "sample_correct": int(sample_labels[sample_idx] == sample_preds[sample_idx]),
            "token_stats": sample.get("token_stats", {}),
        }
        if include_claims:
            sample_entry["claims"] = []
            claim_indices = sample.get("claim_indices", [])
            claim_meta = sample.get("claim_meta", [])
            for local_idx, claim_idx in enumerate(claim_indices):
                if claim_idx >= len(claim_labels):
                    continue
                claim = claim_meta[local_idx] if local_idx < len(claim_meta) else {}
                sample_entry["claims"].append({
                    "claim_text": claim.get("claim_text", ""),
                    "claim_type": claim.get("claim_type", "unknown"),
                    "claim_type_id": int(claim.get("claim_type_id", 2)),
                    "label": int(claim_labels[claim_idx]),
                    "prediction": int(claim_preds[claim_idx]),
                    "probability": float(claim_probs[claim_idx]),
                    "logit": float(claim_logits[claim_idx]),
                    "correct": int(claim_labels[claim_idx] == claim_preds[claim_idx]),
                })

        prediction_rows.append(sample_entry)

    return prediction_rows


HEAVY_REASONING_POSTHOC_METHODS = {
    "reasoning_logistic",
    "reasoning_logistic_isotonic",
    "reasoning_logistic_blend",
    "binwise_hybrid",
}


def _normalized_posthoc_method_name(value: str | None) -> str:
    return str(value or "").strip().lower()


def _posthoc_method_needs_heavy_reasoning(value: str | None) -> bool:
    return _normalized_posthoc_method_name(value) in HEAVY_REASONING_POSTHOC_METHODS


def _to_probs(logits_or_probs: np.ndarray) -> np.ndarray:
    scores = logits_or_probs.astype(np.float64)
    if scores.min() < 0 or scores.max() > 1:
        return expit(scores)
    return scores


def _predict_logits_for_dataset(
    model,
    dataset,
    device,
    batch_size: int,
    collect_prediction_rows: bool = False,
    include_heavy_prediction_rows: bool = False,
    split_name: str | None = None,
):
    """Run a single prediction pass and return logits + aligned metadata arrays."""
    from torch.utils.data import DataLoader
    import time

    reset_gpu_peak_memory(device)
    eval_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_claim_cached_features,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    all_claim_logits = []
    all_claim_labels = []
    all_claim_type_ids = []
    all_claim_difficulties = []

    all_sample_logits = []
    all_sample_labels = []
    all_sample_difficulties = []

    sample_rows_meta = []
    global_sample_idx = 0
    global_claim_idx = 0

    total_samples = len(dataset)
    total_batches = len(eval_loader)
    log.info(
        "Prediction pass start%s: samples=%d batches=%d batch_size=%d collect_rows=%s heavy_rows=%s",
        f" split={split_name}" if split_name else "",
        total_samples,
        total_batches,
        batch_size,
        bool(collect_prediction_rows),
        bool(include_heavy_prediction_rows),
    )
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(eval_loader, start=1):
            features = batch["features"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            claim_masks = batch.get("claim_masks")
            claim_types = batch.get("claim_types")
            claim_labels = batch.get("claim_labels")

            output = model(
                features=features,
                attention_mask=attention_mask,
                claim_masks=claim_masks,
                claim_types=claim_types,
                claim_labels=claim_labels,
                labels=None,
            )
            claim_logits_batch = output.logits.detach()
            if claim_logits_batch.ndim > 1:
                claim_logits_batch = claim_logits_batch.squeeze(-1)
            claim_logits_batch = claim_logits_batch.view(-1).cpu().numpy()

            sample_logits_batch = getattr(output, "sample_logits", None)
            if sample_logits_batch is None:
                raise RuntimeError(
                    "CachedFeatureModel did not return sample_logits during evaluation; "
                    "fallback sample aggregation has been removed."
                )
            sample_logits_batch = sample_logits_batch.detach().cpu().squeeze(-1).numpy()

            batch_sample_labels = batch.get("labels").detach().cpu().view(-1).numpy()
            batch_sample_difficulties = batch.get("sample_n_hops") or [0] * len(batch_sample_labels)
            batch_sample_ids = batch.get("sample_ids") or [-1] * len(batch_sample_labels)
            batch_sample_uids = batch.get("sample_uids") or [""] * len(batch_sample_labels)
            batch_questions = batch.get("sample_questions") or [""] * len(batch_sample_labels)
            batch_generated_texts = batch.get("sample_generated_texts") or [""] * len(batch_sample_labels)
            batch_reasoning_claim_counts = batch.get("sample_reasoning_claim_counts") or [0] * len(batch_sample_labels)
            batch_reasoning_verifieds = (
                batch.get("sample_reasoning_verifieds") or [None] * len(batch_sample_labels)
                if include_heavy_prediction_rows
                else [None] * len(batch_sample_labels)
            )
            batch_reasoning_feature_stats = (
                batch.get("sample_reasoning_feature_stats") or [None] * len(batch_sample_labels)
                if include_heavy_prediction_rows
                else [None] * len(batch_sample_labels)
            )
            batch_reasoning_feature_means = (
                batch.get("sample_reasoning_feature_means") or [None] * len(batch_sample_labels)
                if include_heavy_prediction_rows
                else [None] * len(batch_sample_labels)
            )
            batch_token_probs = (
                batch.get("sample_token_probs") or [None] * len(batch_sample_labels)
                if include_heavy_prediction_rows
                else [None] * len(batch_sample_labels)
            )
            batch_log_likelihoods = (
                batch.get("sample_log_likelihoods") or [None] * len(batch_sample_labels)
                if include_heavy_prediction_rows
                else [None] * len(batch_sample_labels)
            )

            cursor = 0
            for sample_idx_in_batch, labels_tensor in enumerate(claim_labels):
                claim_labels_np = labels_tensor.detach().cpu().view(-1).numpy()
                n_claims = int(claim_labels_np.shape[0])

                claim_types_tensor = claim_types[sample_idx_in_batch]
                claim_types_np = claim_types_tensor.detach().cpu().view(-1).numpy()
                remaining_claim_logits = max(0, int(claim_logits_batch.shape[0] - cursor))
                n_claim_logits = min(n_claims, remaining_claim_logits)
                claim_logits_np = claim_logits_batch[cursor: cursor + n_claim_logits]
                cursor += n_claim_logits

                records = []
                if batch.get("claim_records") is not None and sample_idx_in_batch < len(batch["claim_records"]):
                    records = batch["claim_records"][sample_idx_in_batch] or []

                row_claim_indices = []
                row_claim_meta = []
                sample_hops = int(batch_sample_difficulties[sample_idx_in_batch])

                for local_claim_idx in range(n_claims):
                    if local_claim_idx >= n_claim_logits:
                        continue
                    type_val = int(claim_types_np[local_claim_idx]) if local_claim_idx < len(claim_types_np) else 2
                    rec = records[local_claim_idx] if local_claim_idx < len(records) else None
                    if rec is None:
                        continue

                    label_val = int(claim_labels_np[local_claim_idx])

                    all_claim_logits.append(float(claim_logits_np[local_claim_idx]))
                    all_claim_labels.append(label_val)
                    all_claim_type_ids.append(type_val)
                    all_claim_difficulties.append(int(rec.get("n_hops", sample_hops)) if isinstance(rec, dict) else sample_hops)

                    if collect_prediction_rows:
                        row_claim_indices.append(global_claim_idx)
                        row_claim_meta.append({
                            "claim_text": str(rec.get("claim_text", "")) if isinstance(rec, dict) else "",
                            "claim_type": str(rec.get("claim_type", "unknown")) if isinstance(rec, dict) else "unknown",
                            "claim_type_id": int(rec.get("claim_type_id", type_val)) if isinstance(rec, dict) else int(type_val),
                        })
                    global_claim_idx += 1

                all_sample_logits.append(float(sample_logits_batch[sample_idx_in_batch]))
                all_sample_labels.append(int(batch_sample_labels[sample_idx_in_batch]))
                all_sample_difficulties.append(sample_hops)

                if collect_prediction_rows:
                    token_stats = {
                        "neg_mean_ll": float("nan"),
                        "neg_top1_logp": float("nan"),
                        "neg_margin_logp": float("nan"),
                    }
                    reasoning_label_stats = {
                        "verified_count": 0,
                        "mean": float("nan"),
                        "std": float("nan"),
                        "match_rate": float("nan"),
                        "label_gap_mean": float("nan"),
                        "majority_gap": float("nan"),
                    }
                    reasoning_feature_gap_stats = {
                        "l2": float("nan"),
                        "mean_abs": float("nan"),
                        "max_abs": float("nan"),
                        "cosine": float("nan"),
                    }
                    reasoning_feature_stats = {}
                    conclusion_feature_mean_np = np.asarray([], dtype=np.float32)
                    reasoning_feature_mean_np = np.asarray([], dtype=np.float32)

                    if include_heavy_prediction_rows:
                        token_probs_tensor = batch_token_probs[sample_idx_in_batch]
                        ll_tensor = batch_log_likelihoods[sample_idx_in_batch]
                        reasoning_verified = batch_reasoning_verifieds[sample_idx_in_batch]
                        feature_i = features[sample_idx_in_batch]
                        attn_i = attention_mask[sample_idx_in_batch].to(dtype=feature_i.dtype)
                        conclusion_idx = 0
                        if claim_types is not None:
                            type_row = claim_types[sample_idx_in_batch].view(-1)
                            limit = min(int(type_row.numel()), int(claim_masks[sample_idx_in_batch].shape[0]))
                            for local_idx in range(limit):
                                if int(type_row[local_idx].item()) == 1:
                                    conclusion_idx = local_idx
                                    break

                        conclusion_mask = claim_masks[sample_idx_in_batch][conclusion_idx].to(feature_i.device).to(dtype=feature_i.dtype)
                        if conclusion_mask.shape[0] > feature_i.shape[0]:
                            conclusion_mask = conclusion_mask[: feature_i.shape[0]]
                        elif conclusion_mask.shape[0] < feature_i.shape[0]:
                            pad_len = feature_i.shape[0] - conclusion_mask.shape[0]
                            conclusion_mask = torch.nn.functional.pad(conclusion_mask, (0, pad_len), value=0.0)
                        conclusion_mask = conclusion_mask * attn_i
                        if float(conclusion_mask.sum().item()) > 0:
                            conclusion_feature_mean = (feature_i * conclusion_mask.unsqueeze(-1)).sum(dim=0) / conclusion_mask.sum().clamp(min=1.0)
                        else:
                            conclusion_feature_mean = (feature_i * attn_i.unsqueeze(-1)).sum(dim=0) / attn_i.sum().clamp(min=1.0)
                        reasoning_feature_mean = batch_reasoning_feature_means[sample_idx_in_batch]
                        if isinstance(token_probs_tensor, torch.Tensor) and isinstance(ll_tensor, torch.Tensor):
                            valid_len = min(int(token_probs_tensor.shape[0]), int(ll_tensor.shape[0]))
                            if valid_len > 0:
                                tp = token_probs_tensor[:valid_len].float()
                                ll = ll_tensor[:valid_len].float()
                                token_stats["neg_mean_ll"] = float((-ll).mean().item())
                                token_stats["neg_top1_logp"] = float((-tp[:, 0]).mean().item())
                                if tp.shape[1] >= 2:
                                    token_stats["neg_margin_logp"] = float((-(tp[:, 0] - tp[:, 1])).mean().item())

                        if reasoning_verified is not None:
                            verified_vals = []
                            for value in reasoning_verified:
                                try:
                                    label_value = int(value)
                                except Exception:
                                    continue
                                if label_value in (0, 1):
                                    verified_vals.append(label_value)
                            if verified_vals:
                                verified_arr = np.asarray(verified_vals, dtype=np.float64)
                                sample_label = float(batch_sample_labels[sample_idx_in_batch])
                                majority_label = float(verified_arr.mean() >= 0.5)
                                reasoning_label_stats = {
                                    "verified_count": int(verified_arr.size),
                                    "mean": float(verified_arr.mean()),
                                    "std": float(verified_arr.std()),
                                    "match_rate": float(np.mean(verified_arr == sample_label)),
                                    "label_gap_mean": float(np.mean(np.abs(verified_arr - sample_label))),
                                    "majority_gap": float(abs(majority_label - sample_label)),
                                }

                        if isinstance(reasoning_feature_mean, torch.Tensor):
                            reasoning_vec = reasoning_feature_mean.detach().to(feature_i.device).float().view(-1)
                            conclusion_vec = conclusion_feature_mean.detach().to(feature_i.device).float().view(-1)
                            n = min(int(reasoning_vec.numel()), int(conclusion_vec.numel()))
                            if n > 0:
                                reasoning_vec = reasoning_vec[:n]
                                conclusion_vec = conclusion_vec[:n]
                                diff = reasoning_vec - conclusion_vec
                                reasoning_feature_gap_stats = {
                                    "l2": float(diff.norm(p=2).item()),
                                    "mean_abs": float(diff.abs().mean().item()),
                                    "max_abs": float(diff.abs().max().item()),
                                    "cosine": float(F.cosine_similarity(reasoning_vec.unsqueeze(0), conclusion_vec.unsqueeze(0)).item()),
                                }

                        reasoning_feature_stats = batch_reasoning_feature_stats[sample_idx_in_batch] or {}
                        conclusion_feature_mean_np = (
                            conclusion_feature_mean.detach()
                            .to(dtype=torch.float32)
                            .cpu()
                            .numpy()
                            .copy()
                        )
                        reasoning_feature_mean_np = (
                            reasoning_feature_mean.detach()
                            .to(dtype=torch.float32)
                            .cpu()
                            .numpy()
                            .copy()
                            if isinstance(reasoning_feature_mean, torch.Tensor)
                            else np.asarray([], dtype=np.float32)
                        )

                    sample_rows_meta.append({
                        "sample_id": int(global_sample_idx),
                        "dataset_sample_id": int(batch_sample_ids[sample_idx_in_batch]),
                        "dataset_sample_uid": str(batch_sample_uids[sample_idx_in_batch]),
                        "question": str(batch_questions[sample_idx_in_batch]),
                        "generated_text": str(batch_generated_texts[sample_idx_in_batch]),
                        "n_hops": sample_hops,
                        "reasoning_claim_count": int(batch_reasoning_claim_counts[sample_idx_in_batch]),
                        "reasoning_label_stats": reasoning_label_stats,
                        "reasoning_feature_stats": reasoning_feature_stats,
                        "reasoning_feature_gap_stats": reasoning_feature_gap_stats,
                        "conclusion_cached_feature_mean": conclusion_feature_mean_np,
                        "reasoning_cached_feature_mean": reasoning_feature_mean_np,
                        "claim_indices": row_claim_indices,
                        "claim_meta": row_claim_meta,
                        "token_stats": token_stats,
                    })
                global_sample_idx += 1

            elapsed_s = time.time() - t0
            log.info(
                "Prediction pass progress%s: batch=%d/%d samples=%d/%d claims=%d rows=%d heavy_rows=%s elapsed=%.1fs",
                f" split={split_name}" if split_name else "",
                batch_idx,
                total_batches,
                global_sample_idx,
                total_samples,
                len(all_claim_logits),
                len(sample_rows_meta),
                bool(include_heavy_prediction_rows),
                elapsed_s,
            )

    inference_runtime_s = time.time() - t0
    inference_cpu_peak_gb = get_cpu_peak_memory_gb()
    inference_gpu_peak_gb = get_gpu_peak_memory_gb(device)

    return (
        np.array(all_claim_logits, dtype=np.float64),
        np.array(all_sample_logits, dtype=np.float64),
        np.array(all_claim_labels, dtype=np.int64),
        np.array(all_claim_type_ids, dtype=np.int64),
        np.array(all_claim_difficulties, dtype=np.int64),
        np.array(all_sample_labels, dtype=np.int64),
        np.array(all_sample_difficulties, dtype=np.int64),
        sample_rows_meta,
        inference_runtime_s,
        inference_cpu_peak_gb,
        inference_gpu_peak_gb,
    )


def _compute_metrics_with_fixed_predictions(labels, probs, preds, threshold_value):
    """Compute full metric set when predictions are produced externally."""
    return compute_all_metrics(
        labels=np.asarray(labels, dtype=np.int64),
        logits_or_probs=np.asarray(probs, dtype=np.float64),
        threshold=float(threshold_value),
        predictions=np.asarray(preds, dtype=np.int64),
    )


def _summarize_array(values: np.ndarray) -> dict:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    summary = {
        "count": int(arr.size),
        "finite_count": int(finite.size),
        "mean": float("nan"),
        "std": float("nan"),
        "min": float("nan"),
        "max": float("nan"),
        "p05": float("nan"),
        "p50": float("nan"),
        "p95": float("nan"),
    }
    if finite.size == 0:
        return summary

    q05, q50, q95 = np.quantile(finite, [0.05, 0.50, 0.95])
    summary.update(
        {
            "mean": float(finite.mean()),
            "std": float(finite.std()),
            "min": float(finite.min()),
            "max": float(finite.max()),
            "p05": float(q05),
            "p50": float(q50),
            "p95": float(q95),
        }
    )
    return summary


def _histogram_payload(values: np.ndarray, bins: int = 10, value_range: tuple[float, float] = (0.0, 1.0)) -> dict:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {"bin_edges": [], "counts": [], "fractions": []}

    counts, edges = np.histogram(finite, bins=bins, range=value_range)
    total = int(counts.sum())
    fractions = (counts.astype(np.float64) / total) if total > 0 else np.zeros_like(counts, dtype=np.float64)
    return {
        "bin_edges": [float(x) for x in edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
        "fractions": [float(x) for x in fractions.tolist()],
    }


def _build_difficulty_diagnostics(
    difficulties: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
) -> list[dict]:
    if difficulties is None:
        return []

    difficulty_arr = np.asarray(difficulties, dtype=np.int64).reshape(-1)
    out = []
    for difficulty in sorted(set(difficulty_arr.tolist())):
        mask = difficulty_arr == difficulty
        if not np.any(mask):
            continue
        labels_d = labels[mask]
        probs_d = probs[mask]
        preds_d = preds[mask]
        out.append(
            {
                "difficulty": int(difficulty),
                "count": int(mask.sum()),
                "label_positive_rate": float(labels_d.mean()) if labels_d.size else float("nan"),
                "predicted_positive_rate": float(preds_d.mean()) if preds_d.size else float("nan"),
                "mean_probability": float(probs_d.mean()) if probs_d.size else float("nan"),
                "probability_summary": _summarize_array(probs_d),
            }
        )
    return out


def _build_sample_meta_diagnostics(sample_rows_meta: list[dict]) -> dict:
    if not sample_rows_meta:
        return {}

    claim_counts = np.asarray(
        [len(row.get("claim_indices", []) or []) for row in sample_rows_meta],
        dtype=np.float64,
    )
    reasoning_claim_counts = np.asarray(
        [int(row.get("reasoning_claim_count", 0) or 0) for row in sample_rows_meta],
        dtype=np.float64,
    )
    n_hops = np.asarray(
        [int(row.get("n_hops", 0) or 0) for row in sample_rows_meta],
        dtype=np.float64,
    )
    neg_mean_ll = np.asarray(
        [
            float((row.get("token_stats", {}) or {}).get("neg_mean_ll", np.nan))
            for row in sample_rows_meta
        ],
        dtype=np.float64,
    )
    neg_margin_logp = np.asarray(
        [
            float((row.get("token_stats", {}) or {}).get("neg_margin_logp", np.nan))
            for row in sample_rows_meta
        ],
        dtype=np.float64,
    )

    return {
        "claim_count": _summarize_array(claim_counts),
        "reasoning_claim_count": _summarize_array(reasoning_claim_counts),
        "n_hops": _summarize_array(n_hops),
        "token_neg_mean_ll": _summarize_array(neg_mean_ll),
        "token_neg_margin_logp": _summarize_array(neg_margin_logp),
        "single_claim_fraction": float(np.mean(claim_counts <= 1)) if claim_counts.size else float("nan"),
        "single_reasoning_claim_fraction": float(np.mean(reasoning_claim_counts <= 1)) if reasoning_claim_counts.size else float("nan"),
        "single_hop_fraction": float(np.mean(n_hops <= 1)) if n_hops.size else float("nan"),
    }


def _build_eval_diagnostics(
    *,
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    difficulties: np.ndarray | None,
    default_threshold: float,
    applied_threshold: float,
    logits: np.ndarray | None = None,
    pre_probs: np.ndarray | None = None,
    sample_rows_meta: list[dict] | None = None,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probs = np.clip(np.asarray(probs, dtype=np.float64).reshape(-1), 1e-8, 1.0 - 1e-8)
    preds = np.asarray(preds, dtype=np.int64).reshape(-1)
    logits_arr = None if logits is None else np.asarray(logits, dtype=np.float64).reshape(-1)

    diagnostics = {
        "sample_count": int(labels.size),
        "label_positive_rate": float(labels.mean()) if labels.size else float("nan"),
        "predicted_positive_rate": float(preds.mean()) if preds.size else float("nan"),
        "probability_summary": _summarize_array(probs),
        "probability_histogram": {
            "all": _histogram_payload(probs),
            "label_0": _histogram_payload(probs[labels == 0]),
            "label_1": _histogram_payload(probs[labels == 1]),
        },
        "threshold_analysis": {
            "reference_threshold": float(default_threshold),
            "applied_threshold": float(applied_threshold),
            "threshold_delta": float(applied_threshold - default_threshold),
            "predicted_positive_rate_at_reference": float((probs >= float(default_threshold)).mean()) if probs.size else float("nan"),
            "predicted_positive_rate_at_applied": float((probs >= float(applied_threshold)).mean()) if probs.size else float("nan"),
        },
        "difficulty_breakdown": _build_difficulty_diagnostics(difficulties, labels, probs, preds),
    }
    if logits_arr is not None:
        diagnostics["logit_summary"] = _summarize_array(logits_arr)

    if pre_probs is not None:
        pre_probs_arr = np.clip(np.asarray(pre_probs, dtype=np.float64).reshape(-1), 1e-8, 1.0 - 1e-8)
        diagnostics["posthoc_shift"] = {
            "probability_summary_before": _summarize_array(pre_probs_arr),
            "mean_abs_probability_shift": float(np.abs(probs - pre_probs_arr).mean()) if probs.size else float("nan"),
            "max_abs_probability_shift": float(np.abs(probs - pre_probs_arr).max()) if probs.size else float("nan"),
            "predicted_positive_rate_before_at_applied": float((pre_probs_arr >= float(applied_threshold)).mean()) if pre_probs_arr.size else float("nan"),
            "predicted_positive_rate_after_at_applied": float((probs >= float(applied_threshold)).mean()) if probs.size else float("nan"),
        }

    if sample_rows_meta:
        diagnostics["sample_meta"] = _build_sample_meta_diagnostics(sample_rows_meta)

    return diagnostics


def _tune_temperature(labels: np.ndarray, logits: np.ndarray) -> float:
    """Grid-search sample temperature on validation split by minimizing NLL."""
    labels = labels.astype(np.int64)
    logits = logits.astype(np.float64)
    best_t = 1.0
    best_nll = float("inf")
    # Wide but cheap grid. Keeps behavior deterministic and robust.
    for t in np.arange(0.5, 3.01, 0.05):
        probs = expit(logits / float(t))
        probs = np.clip(probs, 1e-8, 1.0 - 1e-8)
        try:
            nll = float(log_loss(labels, probs, labels=[0, 1]))
        except ValueError:
            continue
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t


def _tune_difficulty_thresholds(
    labels: np.ndarray,
    probs: np.ndarray,
    difficulties: np.ndarray,
    min_samples: int,
) -> dict[int, float]:
    """Tune per-difficulty thresholds where support is sufficient."""
    tuned = {}
    for d in sorted(set(difficulties.tolist())):
        mask = difficulties == d
        if int(mask.sum()) < int(min_samples):
            continue
        labels_d = labels[mask].astype(int)
        if len(np.unique(labels_d)) < 2:
            continue
        th_d, _ = find_optimal_threshold(labels_d, probs[mask].astype(np.float64))
        tuned[int(d)] = float(th_d)
    return tuned


def _predict_with_difficulty_thresholds(
    probs: np.ndarray,
    difficulties: np.ndarray,
    global_threshold: float,
    difficulty_thresholds: dict[int, float] | None,
) -> np.ndarray:
    if not difficulty_thresholds:
        return (probs >= global_threshold).astype(int)
    preds = np.zeros_like(probs, dtype=np.int64)
    for i in range(len(probs)):
        d = int(difficulties[i])
        th = float(difficulty_thresholds.get(d, global_threshold))
        preds[i] = int(probs[i] >= th)
    return preds


def _compute_per_difficulty_fixed(
    difficulty_arr: np.ndarray,
    labels: np.ndarray,
    probs: np.ndarray,
    preds: np.ndarray,
    difficulty_field: str,
    default_threshold: float,
    difficulty_thresholds: dict[int, float] | None = None,
):
    out = {}
    unique_vals = sorted(set(difficulty_arr.tolist()))
    for d in unique_vals:
        mask = difficulty_arr == d
        if mask.sum() <= 0:
            continue
        threshold_d = float((difficulty_thresholds or {}).get(int(d), default_threshold))
        d_metrics = _compute_metrics_with_fixed_predictions(
            labels=labels[mask],
            probs=probs[mask],
            preds=preds[mask],
            threshold_value=threshold_d,
        )
        out[int(d)] = d_metrics
        log.info(
            "  %s=%d: n=%d, acc=%.4f, f1=%.4f, pr_auc=%.4f, roc_auc=%.4f",
            difficulty_field,
            d,
            mask.sum(),
            d_metrics["accuracy"],
            d_metrics["f1"],
            d_metrics["pr_auc"],
            d_metrics["roc_auc"],
        )
    return out


PREDICTION_CACHE_SCHEMA_VERSION = 2


def _prediction_cache_file_name(
    split_name: str,
    n_hop_values: list[int] | None,
    *,
    load_reasoning_sidecar: bool,
    collect_prediction_rows: bool,
    include_heavy_prediction_rows: bool,
) -> str:
    split_token = str(split_name or "unknown").replace(os.sep, "_")
    hop_token = "-".join(str(int(v)) for v in n_hop_values) if n_hop_values else "all"
    return (
        f"{split_token}__nhops_{hop_token}"
        f"__reasoning_{int(load_reasoning_sidecar)}"
        f"__rows_{int(collect_prediction_rows)}"
        f"__heavy_{int(include_heavy_prediction_rows)}.npz"
    )


def _save_prediction_cache(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray([PREDICTION_CACHE_SCHEMA_VERSION], dtype=np.int64),
        claim_logits=np.asarray(payload["claim_logits"], dtype=np.float64),
        sample_logits=np.asarray(payload["sample_logits"], dtype=np.float64),
        claim_labels=np.asarray(payload["claim_labels"], dtype=np.int64),
        claim_type_ids=np.asarray(payload["claim_type_ids"], dtype=np.int64),
        claim_difficulties=np.asarray(payload["claim_difficulties"], dtype=np.int64),
        sample_labels=np.asarray(payload["sample_labels"], dtype=np.int64),
        sample_difficulties=np.asarray(payload["sample_difficulties"], dtype=np.int64),
        sample_rows_meta=np.asarray(payload.get("sample_rows_meta", []), dtype=object),
        dataset_filter_stats=np.asarray([payload.get("dataset_filter_stats", {})], dtype=object),
        inference_runtime_s=np.asarray([payload.get("inference_runtime_s", 0.0)], dtype=np.float64),
        inference_cpu_peak_gb=np.asarray([payload.get("inference_cpu_peak_gb", 0.0)], dtype=np.float64),
        inference_gpu_peak_gb=np.asarray([payload.get("inference_gpu_peak_gb", 0.0)], dtype=np.float64),
    )


def _load_prediction_cache(path: Path) -> dict[str, object] | None:
    try:
        with np.load(path, allow_pickle=True) as data:
            version = int(np.asarray(data["schema_version"]).reshape(-1)[0])
            if version != PREDICTION_CACHE_SCHEMA_VERSION:
                return None
            dataset_filter_stats_raw = data["dataset_filter_stats"].tolist()
            dataset_filter_stats = dataset_filter_stats_raw[0] if dataset_filter_stats_raw else {}
            return {
                "claim_logits": np.asarray(data["claim_logits"], dtype=np.float64),
                "sample_logits": np.asarray(data["sample_logits"], dtype=np.float64),
                "claim_labels": np.asarray(data["claim_labels"], dtype=np.int64),
                "claim_type_ids": np.asarray(data["claim_type_ids"], dtype=np.int64),
                "claim_difficulties": np.asarray(data["claim_difficulties"], dtype=np.int64),
                "sample_labels": np.asarray(data["sample_labels"], dtype=np.int64),
                "sample_difficulties": np.asarray(data["sample_difficulties"], dtype=np.int64),
                "sample_rows_meta": data["sample_rows_meta"].tolist(),
                "dataset_filter_stats": dataset_filter_stats,
                "inference_runtime_s": float(np.asarray(data["inference_runtime_s"], dtype=np.float64).reshape(-1)[0]),
                "inference_cpu_peak_gb": float(np.asarray(data["inference_cpu_peak_gb"], dtype=np.float64).reshape(-1)[0]),
                "inference_gpu_peak_gb": float(np.asarray(data["inference_gpu_peak_gb"], dtype=np.float64).reshape(-1)[0]),
            }
    except Exception:
        return None


def evaluate_supervised(args, cache_dir, n_hop_values, output_dir, eval_ctx: EvalContext, cfg: Config) -> EvalReport:
    """Evaluate a trained supervised head on cached features."""
    workflow = WorkflowLogger(log, "eval", width=cfg.output.log_banner_width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    head_config_path = os.path.join(args.head_path, "head_config.json")
    with open(head_config_path, "r") as f:
        head_cfg = json.load(f)
    head_type = head_cfg.get("head_type", "")

    workflow.event(
        "supervised.model",
        head_type=head_type,
        feature_dim=head_cfg.get("feature_dim"),
        num_classes=head_cfg.get("num_classes", 1),
        head_dim=head_cfg.get("head_dim"),
        n_layers=head_cfg.get("n_layers"),
        n_heads=head_cfg.get("n_heads"),
        trainable_params=head_cfg.get("trainable_params", 0),
    )

    posthoc_enabled = bool(getattr(args, "enable_posthoc", True))
    feature_dim = head_cfg["feature_dim"]
    num_classes = head_cfg.get("num_classes", 1)
    head = build_head(
        head_type=head_cfg["head_type"],
        feature_dim=feature_dim,
        num_classes=num_classes,
        head_dim=head_cfg.get("head_dim", 512),
        n_layers=head_cfg.get("n_layers", 2),
        n_heads=head_cfg.get("n_heads", 8),
        dropout=head_cfg.get("dropout", 0.1),
    ).to(device)

    head_weights_path = os.path.join(args.head_path, "head_weights.pth")
    head.load_state_dict(torch.load(head_weights_path, map_location="cpu"))
    head.to(device)

    model = CachedFeatureModel(
        head=head,
        num_classes=num_classes,
        loss_type=head_cfg.get("loss_type", "bce"),
        pos_weight=float(head_cfg.get("loss_pos_weight", 1.0)),
        focal_gamma=float(head_cfg.get("focal_gamma", 2.0)),
        sample_pos_weight=float(head_cfg.get("sample_pos_weight", 1.0)),
    )
    model.to(device)
    model.eval()

    resolved_posthoc_method = _normalized_posthoc_method_name(args.posthoc_method)
    preloaded_posthoc_calibrator = None
    if posthoc_enabled and args.posthoc_model_path:
        preloaded_posthoc_calibrator = load_posthoc_calibrator(args.posthoc_model_path)
        resolved_posthoc_method = _normalized_posthoc_method_name(
            preloaded_posthoc_calibrator.get("mode", resolved_posthoc_method)
        )
    posthoc_needs_heavy_reasoning = bool(
        posthoc_enabled and _posthoc_method_needs_heavy_reasoning(resolved_posthoc_method)
    )

    prediction_cache_dir = str(getattr(args, "prediction_cache_dir", "") or "").strip()
    split_prediction_payloads: dict[str, dict[str, object]] = {}

    def _prediction_cache_path(
        split_name: str,
        *,
        load_reasoning_sidecar: bool,
        collect_prediction_rows: bool,
        include_heavy_prediction_rows: bool,
    ) -> Path | None:
        if not prediction_cache_dir:
            return None
        return Path(prediction_cache_dir) / _prediction_cache_file_name(
            split_name,
            n_hop_values,
            load_reasoning_sidecar=load_reasoning_sidecar,
            collect_prediction_rows=collect_prediction_rows,
            include_heavy_prediction_rows=include_heavy_prediction_rows,
        )

    def _get_split_predictions(
        split_name: str,
        *,
        load_reasoning_sidecar: bool,
        collect_prediction_rows: bool,
        include_heavy_prediction_rows: bool = False,
    ) -> dict[str, object]:
        effective_collect_prediction_rows = bool(collect_prediction_rows)
        effective_include_heavy_prediction_rows = bool(
            include_heavy_prediction_rows and effective_collect_prediction_rows
        )
        effective_load_reasoning_sidecar = bool(
            load_reasoning_sidecar or effective_include_heavy_prediction_rows
        )
        cache_key = (
            f"{split_name}"
            f"|reasoning={int(effective_load_reasoning_sidecar)}"
            f"|rows={int(effective_collect_prediction_rows)}"
            f"|heavy={int(effective_include_heavy_prediction_rows)}"
        )
        canonical_key = cache_key
        if canonical_key in split_prediction_payloads:
            return split_prediction_payloads[canonical_key]
        if cache_key in split_prediction_payloads:
            return split_prediction_payloads[cache_key]

        cache_path = _prediction_cache_path(
            split_name,
            load_reasoning_sidecar=effective_load_reasoning_sidecar,
            collect_prediction_rows=effective_collect_prediction_rows,
            include_heavy_prediction_rows=effective_include_heavy_prediction_rows,
        )
        if cache_path is not None and cache_path.is_file():
            cached_payload = _load_prediction_cache(cache_path)
            if cached_payload is not None:
                log.info("Loaded prediction cache for split=%s from %s", split_name, cache_path)
                split_prediction_payloads[canonical_key] = cached_payload
                split_prediction_payloads[cache_key] = cached_payload
                return cached_payload
            log.warning("Prediction cache at %s is unreadable or outdated; recomputing.", cache_path)

        log.info(
            "Preparing split=%s predictions: cache=%s load_reasoning_sidecar=%s collect_rows=%s heavy_rows=%s method=%s",
            split_name,
            str(cache_path) if cache_path is not None else "disabled",
            effective_load_reasoning_sidecar,
            effective_collect_prediction_rows,
            effective_include_heavy_prediction_rows,
            resolved_posthoc_method or "none",
        )
        split_ds = CachedFeatureDataset(
            cache_dir=cache_dir,
            split=split_name,
            n_hop_values=n_hop_values,
            max_samples=args.max_samples,
            preload_all=False,
            max_cached_chunks=0,
            mem_budget_gb=cfg.cache.eval_mem_budget_gb,
            load_reasoning_sidecar=effective_load_reasoning_sidecar,
            skip_no_verified_claims=False,
        )
        dataset_filter_stats = split_ds.filter_stats()
        if len(split_ds) == 0:
            split_ds.clear_cache()
            payload = {
                "claim_logits": np.asarray([], dtype=np.float64),
                "sample_logits": np.asarray([], dtype=np.float64),
                "claim_labels": np.asarray([], dtype=np.int64),
                "claim_type_ids": np.asarray([], dtype=np.int64),
                "claim_difficulties": np.asarray([], dtype=np.int64),
                "sample_labels": np.asarray([], dtype=np.int64),
                "sample_difficulties": np.asarray([], dtype=np.int64),
                "sample_rows_meta": [],
                "dataset_filter_stats": dataset_filter_stats,
                "inference_runtime_s": 0.0,
                "inference_cpu_peak_gb": 0.0,
                "inference_gpu_peak_gb": 0.0,
            }
            if cache_path is not None:
                _save_prediction_cache(cache_path, payload)
                log.info("Saved empty prediction cache for split=%s to %s", split_name, cache_path)
            split_prediction_payloads[canonical_key] = payload
            split_prediction_payloads[cache_key] = payload
            return payload

        (
            claim_logits_arr,
            sample_logits_arr,
            claim_labels_arr,
            claim_type_ids_arr,
            claim_difficulties_arr,
            sample_labels_arr,
            sample_difficulties_arr,
            sample_rows_meta_arr,
            inference_runtime_s,
            inference_cpu_peak_gb,
            inference_gpu_peak_gb,
        ) = _predict_logits_for_dataset(
            model=model,
            dataset=split_ds,
            device=device,
            batch_size=args.batch_size,
            collect_prediction_rows=effective_collect_prediction_rows,
            include_heavy_prediction_rows=effective_include_heavy_prediction_rows,
            split_name=split_name,
        )
        split_ds.clear_cache()
        payload = {
            "claim_logits": claim_logits_arr,
            "sample_logits": sample_logits_arr,
            "claim_labels": claim_labels_arr,
            "claim_type_ids": claim_type_ids_arr,
            "claim_difficulties": claim_difficulties_arr,
            "sample_labels": sample_labels_arr,
            "sample_difficulties": sample_difficulties_arr,
            "sample_rows_meta": sample_rows_meta_arr,
            "dataset_filter_stats": dataset_filter_stats,
            "inference_runtime_s": inference_runtime_s,
            "inference_cpu_peak_gb": inference_cpu_peak_gb,
            "inference_gpu_peak_gb": inference_gpu_peak_gb,
        }
        if cache_path is not None:
            log.info("Saving prediction cache for split=%s to %s", split_name, cache_path)
            _save_prediction_cache(cache_path, payload)
            log.info("Saved prediction cache for split=%s to %s", split_name, cache_path)
        split_prediction_payloads[canonical_key] = payload
        split_prediction_payloads[cache_key] = payload
        return payload

    threshold = args.threshold
    posthoc_info = {
        "enabled": posthoc_enabled,
        "applied": False,
        "mode": None,
        "tune_split": None,
        "tune_sample_count": 0,
        "calibrator_path": None,
        "feature_mode_requested": str(args.posthoc_feature_mode).strip().lower(),
        "feature_mode_effective": None,
        "feature_mode_reason": None,
        "feature_meta": None,
    }

    test_payload = _get_split_predictions(
        args.split,
        load_reasoning_sidecar=posthoc_needs_heavy_reasoning,
        collect_prediction_rows=bool(getattr(args, "save_predictions", True) or posthoc_enabled),
        include_heavy_prediction_rows=posthoc_needs_heavy_reasoning,
    )
    if len(test_payload["sample_labels"]) == 0:
        workflow.warning("supervised.empty_split", cache_dir=cache_dir, split=args.split)
        threshold = float(args.threshold)
        nan_metrics = {k: float("nan") for k in [
            "accuracy", "precision", "recall", "f1", "roc_auc",
            "pr_auc", "ece", "mec", "mce", "ace", "nll", "brier", "aurc", "naurc",
            "brier_reliability", "brier_resolution", "brier_uncertainty",
            "risk_at_80_cov", "risk_at_90_cov", "threshold",
        ]}
        return {
            "method_type": "supervised",
            "head_type": head_cfg["head_type"],
            "base_model_name": eval_ctx.base_model_name,
            "head_path": args.head_path,
            "dataset_name": eval_ctx.dataset_name,
            "split": args.split,
            "total_samples": 0,
            "overall_metrics": nan_metrics,
            "per_difficulty_metrics": {},
            "dataset_filter_stats": test_payload.get("dataset_filter_stats", {}),
            "threshold_protocol": {
                "mode": "none",
                "default_threshold": threshold,
                "tune_split": None,
                "eval_split": args.split,
                "applied_thresholds": {
                    "default": threshold,
                    "sample": threshold,
                },
            },
            "skipped_reason": "no_samples",
        }

    # Manual eval loop — HF Trainer.predict() truncates outputs to len(dataset),
    # discarding per-claim logits when #claims > #samples.
    workflow.stage_start(
        "supervised.predict",
        split=args.split,
        batch_size=args.batch_size,
        total_samples=len(test_payload["sample_labels"]),
        total_claims=int(len(test_payload["claim_labels"])),
    )
    claim_logits = np.asarray(test_payload["claim_logits"], dtype=np.float64)
    sample_logits = np.asarray(test_payload["sample_logits"], dtype=np.float64)
    claim_labels = np.asarray(test_payload["claim_labels"], dtype=np.int64)
    claim_type_ids = np.asarray(test_payload["claim_type_ids"], dtype=np.int64)
    claim_difficulties = np.asarray(test_payload["claim_difficulties"], dtype=np.int64)
    sample_labels = np.asarray(test_payload["sample_labels"], dtype=np.int64)
    sample_difficulties = np.asarray(test_payload["sample_difficulties"], dtype=np.int64)
    sample_rows_meta = list(test_payload["sample_rows_meta"])
    inference_runtime_s = float(test_payload["inference_runtime_s"])
    inference_cpu_peak_gb = float(test_payload["inference_cpu_peak_gb"])
    inference_gpu_peak_gb = float(test_payload["inference_gpu_peak_gb"])
    workflow.stage_end(
        "supervised.predict",
        claim_rows=len(claim_labels),
        sample_rows=len(sample_labels),
        runtime_s=inference_runtime_s,
        cpu_peak_gb=inference_cpu_peak_gb,
        gpu_peak_gb=inference_gpu_peak_gb,
    )

    fixed_overall_metrics = compute_all_metrics(sample_labels, sample_logits, threshold=threshold)

    sample_threshold = float(threshold)
    sample_temperature = 1.0
    difficulty_thresholds: dict[int, float] = {}
    threshold_protocol = build_threshold_protocol(
        mode="fixed",
        default_threshold=float(threshold),
        tune_split=None,
        eval_split=args.split,
        sample_threshold=float(sample_threshold),
        sample_temperature=float(sample_temperature),
        difficulty_thresholds=difficulty_thresholds,
    )
    tune_sample_count = 0
    tune_dataset_filter_stats = None
    train_results_threshold, train_results_meta = _load_threshold_from_train_results(
        args.head_path,
    )

    def _tune_aux_thresholds(tune_split_name: str):
        nonlocal sample_temperature, difficulty_thresholds, tune_sample_count, tune_dataset_filter_stats
        if not tune_split_name or tune_split_name == args.split:
            log.warning(
                "Tune split=%s equals eval split=%s. Skipping auxiliary threshold tuning to avoid leakage.",
                tune_split_name,
                args.split,
            )
            return None

        tune_payload = _get_split_predictions(
            tune_split_name,
            load_reasoning_sidecar=False,
            collect_prediction_rows=False,
            include_heavy_prediction_rows=False,
        )
        tune_dataset_filter_stats = tune_payload.get("dataset_filter_stats")
        if len(tune_payload["sample_labels"]) == 0:
            msg = f"Threshold tune split '{tune_split_name}' has 0 samples in {cache_dir}."
            if args.require_tune_split:
                raise RuntimeError(msg)
            log.warning("%s Leaving auxiliary thresholds disabled.", msg)
            return None

        tune_sample_logits = np.asarray(tune_payload["sample_logits"], dtype=np.float64)
        tune_sample_labels = np.asarray(tune_payload["sample_labels"], dtype=np.int64)
        tune_sample_difficulties = np.asarray(tune_payload["sample_difficulties"], dtype=np.int64)
        if args.enable_temp_scaling:
            sample_temperature = _tune_temperature(
                labels=tune_sample_labels,
                logits=tune_sample_logits,
            )
        tune_sample_probs = _to_probs(tune_sample_logits / sample_temperature)
        if args.enable_difficulty_thresholds:
            difficulty_thresholds = _tune_difficulty_thresholds(
                labels=tune_sample_labels,
                probs=tune_sample_probs,
                difficulties=tune_sample_difficulties,
                min_samples=args.difficulty_threshold_min_samples,
            )
        tune_sample_count = int(len(tune_sample_labels))
        log.info(
            "Aux thresholds tuned on %s: difficulty thresholds=%d, sample_temperature=%.2f",
            tune_split_name,
            len(difficulty_thresholds),
            sample_temperature,
        )
        return {
            "sample_labels": tune_sample_labels,
            "sample_probs": tune_sample_probs,
        }

    def _collect_split_predictions(split_name: str, collect_prediction_rows: bool = True):
        split_payload = _get_split_predictions(
            split_name,
            load_reasoning_sidecar=posthoc_needs_heavy_reasoning,
            collect_prediction_rows=collect_prediction_rows,
            include_heavy_prediction_rows=posthoc_needs_heavy_reasoning,
        )
        if len(split_payload["sample_labels"]) == 0:
            return None
        return {
            "claim_logits": np.asarray(split_payload["claim_logits"], dtype=np.float64),
            "sample_logits": np.asarray(split_payload["sample_logits"], dtype=np.float64),
            "claim_labels": np.asarray(split_payload["claim_labels"], dtype=np.int64),
            "claim_type_ids": np.asarray(split_payload["claim_type_ids"], dtype=np.int64),
            "sample_labels": np.asarray(split_payload["sample_labels"], dtype=np.int64),
            "sample_rows_meta": list(split_payload["sample_rows_meta"]),
        }

    if args.threshold_source_report:
        sample_threshold, source_meta = _load_thresholds_from_report(
            args.threshold_source_report,
            default_threshold=threshold,
        )
        source_tune_split = source_meta.get("tune_split")
        if (
            source_tune_split
            and str(source_tune_split) == str(args.split)
            and str(args.split).strip().lower() == "test"
        ):
            raise RuntimeError(
                f"Leakage guard: threshold source tune_split ({source_tune_split}) "
                f"must differ from eval split ({args.split})."
            )
        threshold_protocol = build_threshold_protocol(
            mode="source_report",
            default_threshold=float(threshold),
            tune_split=source_tune_split,
            eval_split=args.split,
            sample_threshold=float(sample_threshold),
            sample_temperature=float(source_meta.get("sample_temperature", 1.0)),
            difficulty_thresholds=source_meta.get("difficulty_thresholds", {}),
            extras={
                "source_report": args.threshold_source_report,
                "source_dataset": source_meta.get("dataset_name"),
                "source_split": source_meta.get("split"),
            },
        )
        tune_sample_count = int(source_meta.get("tune_sample_count", 0))
        sample_temperature = float(source_meta.get("sample_temperature", 1.0))
        difficulty_thresholds = source_meta.get("difficulty_thresholds", {}) or {}
        log.info(
            "Threshold source=source_report split=%s sample=%.2f temp=%.2f diff_bins=%d path=%s",
            source_meta.get("tune_split"),
            sample_threshold,
            sample_temperature,
            len(difficulty_thresholds),
            args.threshold_source_report,
        )
    elif train_results_threshold is not None:
        sample_threshold = float(train_results_threshold)
        tune_split = args.threshold_tune_split
        tuned_aux = _tune_aux_thresholds(tune_split)
        threshold_protocol = build_threshold_protocol(
            mode="train_results",
            default_threshold=float(threshold),
            tune_split=tune_split if tuned_aux is not None else None,
            eval_split=args.split,
            sample_threshold=float(sample_threshold),
            sample_temperature=float(sample_temperature),
            difficulty_thresholds=difficulty_thresholds,
            extras={"source_train_results": train_results_meta.get("path")},
        )
        log.info(
            "Threshold source=train_results split=%s sample=%.2f temp=%.2f diff_bins=%d aux_tuned=%s path=%s",
            tune_split if tuned_aux is not None else None,
            sample_threshold,
            sample_temperature,
            len(difficulty_thresholds),
            tuned_aux is not None,
            train_results_meta.get("path"),
        )
    else:
        tune_split = args.threshold_tune_split
        if tune_split and tune_split != args.split:
            tuned_aux = _tune_aux_thresholds(tune_split)
            if tuned_aux is not None:
                tune_sample_probs = tuned_aux["sample_probs"]
                tune_sample_labels = tuned_aux["sample_labels"]
                sample_threshold, _ = find_optimal_threshold(
                    tune_sample_labels.astype(int),
                    tune_sample_probs.astype(np.float64),
                )
                threshold_protocol = build_threshold_protocol(
                    mode="tuned_on_split",
                    default_threshold=float(threshold),
                    tune_split=tune_split,
                    eval_split=args.split,
                    sample_threshold=float(sample_threshold),
                    sample_temperature=float(sample_temperature),
                    difficulty_thresholds=difficulty_thresholds,
                    extras={
                        "enable_temp_scaling": bool(args.enable_temp_scaling),
                        "enable_difficulty_thresholds": bool(args.enable_difficulty_thresholds),
                        "difficulty_threshold_min_samples": int(args.difficulty_threshold_min_samples),
                    },
                )
                tune_sample_count = int(len(tune_sample_labels))
                log.info(
                    "Threshold source=tuned_on_split split=%s sample=%.2f temp=%.2f diff_bins=%d samples=%d",
                    tune_split,
                    sample_threshold,
                    sample_temperature,
                    len(difficulty_thresholds),
                    tune_sample_count,
                )
        else:
            log.warning(
                "Tune split=%s equals eval split=%s. Using fixed threshold to avoid leakage.",
                tune_split, args.split,
            )

    if args.fit_thresholds_on_eval_split:
        sample_probs_for_fit = _to_probs(sample_logits / sample_temperature)
        sample_threshold, _ = find_optimal_threshold(
            sample_labels.astype(int),
            sample_probs_for_fit.astype(np.float64),
        )
        if args.enable_difficulty_thresholds:
            difficulty_thresholds = _tune_difficulty_thresholds(
                labels=sample_labels,
                probs=sample_probs_for_fit,
                difficulties=sample_difficulties,
                min_samples=args.difficulty_threshold_min_samples,
            )
        tune_sample_count = int(len(sample_labels))
        threshold_protocol = build_threshold_protocol(
            mode="fit_on_eval_split",
            default_threshold=float(threshold),
            tune_split=args.split,
            eval_split=args.split,
            sample_threshold=float(sample_threshold),
            sample_temperature=float(sample_temperature),
            difficulty_thresholds=difficulty_thresholds,
            extras={"fit_thresholds_on_eval_split": True},
        )
        log.info(
            "Threshold source=eval_split split=%s sample=%.2f temp=%.2f diff_bins=%d samples=%d",
            args.split,
            sample_threshold,
            sample_temperature,
            len(difficulty_thresholds),
            tune_sample_count,
        )

    claim_probs = _to_probs(claim_logits)
    claim_preds = np.zeros_like(claim_labels, dtype=np.int64)

    raw_sample_probs = _to_probs(sample_logits)
    sample_probs = _to_probs(sample_logits / sample_temperature)
    pre_posthoc_sample_probs = sample_probs.copy()
    threshold_before_posthoc = float(sample_threshold)
    difficulty_thresholds_before_posthoc = dict(difficulty_thresholds)
    calibrator = None
    if posthoc_enabled:
        calibrator_path = args.posthoc_model_path or os.path.join(output_dir, "posthoc_calibrator.json")
        posthoc_info["calibrator_path"] = calibrator_path
        if args.posthoc_model_path:
            calibrator = preloaded_posthoc_calibrator or load_posthoc_calibrator(args.posthoc_model_path)
            posthoc_info["mode"] = str(calibrator.get("mode", args.posthoc_method))
            posthoc_info["applied"] = True
            feature_meta = calibrator.get("feature_meta", {}) or {}
            posthoc_info["feature_meta"] = feature_meta
            posthoc_info["feature_mode_effective"] = feature_meta.get("effective_feature_mode")
            posthoc_info["feature_mode_reason"] = feature_meta.get("feature_mode_reason")
            log.info("Loaded post-hoc calibrator from %s", args.posthoc_model_path)
        else:
            posthoc_tune_split = str(args.posthoc_tune_split or "").strip()
            if (
                posthoc_tune_split
                and posthoc_tune_split == args.split
                and str(args.split).strip().lower() == "test"
            ):
                raise RuntimeError(
                    f"Leakage guard: post-hoc tune split ({posthoc_tune_split}) "
                    f"must differ from eval split ({args.split})."
                )
            if posthoc_tune_split and posthoc_tune_split != args.split:
                tune_payload = _collect_split_predictions(posthoc_tune_split, collect_prediction_rows=True)
                if tune_payload is not None and len(tune_payload["sample_labels"]) > 0:
                    log.info(
                        "Post-hoc tuning: building feature matrix on split=%s samples=%d method=%s",
                        posthoc_tune_split,
                        len(tune_payload["sample_labels"]),
                        args.posthoc_method,
                    )
                    X_tune, feature_names, feature_meta = build_reasoning_feature_matrix_with_mode(
                        sample_rows_meta=tune_payload["sample_rows_meta"],
                        sample_logits=tune_payload["sample_logits"],
                        claim_probs=_to_probs(tune_payload["claim_logits"]),
                        claim_logits=tune_payload["claim_logits"],
                        feature_mode=args.posthoc_feature_mode,
                        min_samples_for_full=args.posthoc_min_samples_for_full,
                    )
                    y_tune_err = 1 - tune_payload["sample_labels"].astype(np.int64)
                    if len(np.unique(y_tune_err)) >= 2:
                        log.info(
                            "Post-hoc tuning: fitting calibrator method=%s effective_feature_mode=%s feature_dim=%d",
                            args.posthoc_method,
                            feature_meta.get("effective_feature_mode"),
                            int(X_tune.shape[1]) if X_tune.ndim == 2 else 0,
                        )
                        calibrator = fit_posthoc_calibrator(
                            mode=args.posthoc_method,
                            X_tune=X_tune,
                            y_tune=y_tune_err,
                            feature_names=feature_names,
                        )
                        calibrator["feature_meta"] = feature_meta
                        save_posthoc_calibrator(calibrator, calibrator_path)
                        posthoc_info["mode"] = str(calibrator.get("mode", args.posthoc_method))
                        posthoc_info["tune_split"] = posthoc_tune_split
                        posthoc_info["tune_sample_count"] = int(len(tune_payload["sample_labels"]))
                        posthoc_info["applied"] = True
                        posthoc_info["feature_meta"] = feature_meta
                        posthoc_info["feature_mode_effective"] = feature_meta.get("effective_feature_mode")
                        posthoc_info["feature_mode_reason"] = feature_meta.get("feature_mode_reason")
                        tune_conf_probs = 1.0 - apply_posthoc_calibrator(calibrator, X_tune)
                        tune_conf_probs = np.clip(tune_conf_probs.astype(np.float64), 1e-8, 1.0 - 1e-8)
                        sample_threshold, _ = find_optimal_threshold(
                            tune_payload["sample_labels"].astype(int),
                            tune_conf_probs,
                        )
                        tune_sample_count = max(tune_sample_count, int(len(tune_payload["sample_labels"])))
                        log.info(
                            "Fitted post-hoc calibrator on %s split (%d samples), tuned sample_threshold=%.4f",
                            posthoc_tune_split,
                            len(tune_payload["sample_labels"]),
                            sample_threshold,
                        )
                    else:
                        log.warning(
                            "Skipping post-hoc fit on %s: labels have only one class.",
                            posthoc_tune_split,
                        )
                else:
                    log.warning("Skipping post-hoc fit: tune split %s has no samples.", posthoc_tune_split)
            else:
                log.warning("Skipping post-hoc fit: tune split is empty.")
        if calibrator is not None:
            feature_meta = calibrator.get("feature_meta", {}) or {}
            eval_feature_mode = feature_meta.get("effective_feature_mode") or args.posthoc_feature_mode
            log.info(
                "Post-hoc eval: building feature matrix on split=%s samples=%d method=%s mode=%s",
                args.split,
                len(sample_labels),
                posthoc_info.get("mode") or args.posthoc_method,
                eval_feature_mode,
            )
            X_eval, _, eval_feature_meta = build_reasoning_feature_matrix_with_mode(
                sample_rows_meta=sample_rows_meta,
                sample_logits=sample_logits,
                claim_probs=claim_probs,
                claim_logits=claim_logits,
                feature_mode=eval_feature_mode,
                min_samples_for_full=args.posthoc_min_samples_for_full,
            )
            sample_probs = 1.0 - apply_posthoc_calibrator(calibrator, X_eval)
            sample_probs = np.clip(sample_probs.astype(np.float64), 1e-8, 1.0 - 1e-8)
            posthoc_info["applied"] = True
            if not posthoc_info.get("feature_meta"):
                posthoc_info["feature_meta"] = eval_feature_meta
                posthoc_info["feature_mode_effective"] = eval_feature_meta.get("effective_feature_mode")
                posthoc_info["feature_mode_reason"] = eval_feature_meta.get("feature_mode_reason")

    sample_preds = _predict_with_difficulty_thresholds(
        probs=sample_probs,
        difficulties=sample_difficulties,
        global_threshold=sample_threshold,
        difficulty_thresholds=difficulty_thresholds,
    )
    overall_metrics = _compute_metrics_with_fixed_predictions(
        labels=sample_labels,
        probs=sample_probs,
        preds=sample_preds,
        threshold_value=sample_threshold,
    )
    raw_sample_preds = _predict_with_difficulty_thresholds(
        probs=raw_sample_probs,
        difficulties=sample_difficulties,
        global_threshold=float(threshold),
        difficulty_thresholds={},
    )
    raw_metrics = _compute_metrics_with_fixed_predictions(
        labels=sample_labels,
        probs=raw_sample_probs,
        preds=raw_sample_preds,
        threshold_value=float(threshold),
    )
    threshold_only_preds = _predict_with_difficulty_thresholds(
        probs=pre_posthoc_sample_probs,
        difficulties=sample_difficulties,
        global_threshold=threshold_before_posthoc,
        difficulty_thresholds=difficulty_thresholds_before_posthoc,
    )
    threshold_only_metrics = _compute_metrics_with_fixed_predictions(
        labels=sample_labels,
        probs=pre_posthoc_sample_probs,
        preds=threshold_only_preds,
        threshold_value=threshold_before_posthoc,
    )
    posthoc_only_preds = _predict_with_difficulty_thresholds(
        probs=sample_probs,
        difficulties=sample_difficulties,
        global_threshold=threshold_before_posthoc,
        difficulty_thresholds=difficulty_thresholds_before_posthoc,
    )
    posthoc_only_metrics = _compute_metrics_with_fixed_predictions(
        labels=sample_labels,
        probs=sample_probs,
        preds=posthoc_only_preds,
        threshold_value=threshold_before_posthoc,
    )
    posthoc_delta_metrics = {}
    if posthoc_info.get("applied"):
        for metric_name in ("ece", "nll", "brier", "f1", "pr_auc", "roc_auc"):
            before_val = float(threshold_only_metrics.get(metric_name, float("nan")))
            after_val = float(overall_metrics.get(metric_name, float("nan")))
            posthoc_delta_metrics[f"delta_{metric_name}"] = after_val - before_val
        posthoc_info["before_metrics"] = threshold_only_metrics
        posthoc_info["delta_metrics"] = posthoc_delta_metrics
    workflow.metrics("supervised.overall_metrics", overall_metrics)

    workflow.event("supervised.per_difficulty.start", field=eval_ctx.difficulty_field)
    per_difficulty_metrics = _compute_per_difficulty_fixed(
        difficulty_arr=sample_difficulties,
        labels=sample_labels,
        probs=sample_probs,
        preds=sample_preds,
        difficulty_field=eval_ctx.difficulty_field,
        default_threshold=sample_threshold,
        difficulty_thresholds=difficulty_thresholds,
    )

    # Load training results for efficiency info
    train_results_path = Path(args.head_path).resolve().parent / "train_results.json"
    train_results = None
    if train_results_path.is_file():
        with train_results_path.open("r", encoding="utf-8") as f:
            train_results = json.load(f)

    train_eff = (train_results or {}).get("efficiency", {})
    params_m = first_number(
        train_eff.get("params_m"),
        head_cfg.get("trainable_params", 0) / 1e6,
    )

    total_samples = len(sample_labels)
    samples_per_second = 0.0
    if inference_runtime_s > 0 and total_samples > 0:
        samples_per_second = total_samples / inference_runtime_s

    efficiency = {
        "params_m": params_m,
        "flops_g": first_number(train_eff.get("flops_g")),
        "train_time_h": first_number(train_eff.get("train_time_h")),
        "epoch_s": first_number(train_eff.get("epoch_s")),
        "inference_s": inference_runtime_s,
        "samples_per_second": samples_per_second,
        "cpu_memory_gb": max(
            first_number(train_eff.get("cpu_memory_gb"), default=0),
            inference_cpu_peak_gb,
        ),
        "gpu_memory_gb": max(
            first_number(train_eff.get("gpu_memory_gb"), default=0),
            inference_gpu_peak_gb,
        ),
    }
    workflow.metrics("supervised.efficiency", efficiency)
    applied_thresholds = threshold_protocol.setdefault("applied_thresholds", {})
    applied_thresholds["sample"] = float(sample_threshold)
    applied_thresholds["sample_before_posthoc"] = float(threshold_before_posthoc)
    applied_thresholds["sample_temperature"] = float(sample_temperature)
    threshold_protocol["posthoc"] = posthoc_info

    evaluation_variants = {
        "raw_logits": {
            "metrics": raw_metrics,
            "sample_threshold": float(threshold),
            "sample_temperature": 1.0,
            "sample_difficulty_thresholds": {},
        },
        "threshold_only": {
            "metrics": threshold_only_metrics,
            "sample_threshold": float(threshold_before_posthoc),
            "sample_temperature": float(sample_temperature),
            "sample_difficulty_thresholds": {
                str(k): float(v) for k, v in difficulty_thresholds_before_posthoc.items()
            },
        },
        "posthoc_only": {
            "metrics": posthoc_only_metrics,
            "sample_threshold": float(threshold_before_posthoc),
            "sample_temperature": float(sample_temperature),
            "sample_difficulty_thresholds": {
                str(k): float(v) for k, v in difficulty_thresholds_before_posthoc.items()
            },
        },
        "threshold_plus_posthoc": {
            "metrics": overall_metrics,
            "sample_threshold": float(sample_threshold),
            "sample_temperature": float(sample_temperature),
            "sample_difficulty_thresholds": {
                str(k): float(v) for k, v in difficulty_thresholds.items()
            },
        },
    }

    report = {
        "method_type": "supervised",
        "head_type": head_type,
        "base_model_name": eval_ctx.base_model_name,
        "head_path": args.head_path,
        "dataset_name": eval_ctx.dataset_name,
        "split": args.split,
        "difficulty_field": eval_ctx.difficulty_field,
        "n_hop_values": n_hop_values or "all",
        "total_samples": int(len(sample_labels)),
        "total_claims": int(len(claim_labels)),
        "dataset_filter_stats": test_payload.get("dataset_filter_stats", {}),
        "tune_dataset_filter_stats": tune_dataset_filter_stats,
        "num_classes": num_classes,
        "trainable_params": head_cfg.get("trainable_params", 0),
        "overall_metrics": overall_metrics,
        "per_difficulty_metrics": per_difficulty_metrics,
        "threshold": threshold,
        "sample_threshold": float(sample_threshold),
        "sample_temperature": float(sample_temperature),
        "sample_difficulty_thresholds": {str(k): float(v) for k, v in difficulty_thresholds.items()},
        "threshold_protocol": threshold_protocol,
        "posthoc": posthoc_info,
        "evaluation_variants": evaluation_variants,
        "fixed_threshold_metrics": {
            "overall": fixed_overall_metrics,
        },
        "tune_sample_count": tune_sample_count,
        "efficiency": efficiency,
        "trainer_metrics": {},
    }

    diagnostics = _build_eval_diagnostics(
        labels=sample_labels,
        probs=sample_probs,
        preds=sample_preds,
        logits=sample_logits,
        difficulties=sample_difficulties,
        default_threshold=float(threshold),
        applied_threshold=float(sample_threshold),
        pre_probs=pre_posthoc_sample_probs if posthoc_info.get("applied") else None,
        sample_rows_meta=sample_rows_meta,
    )
    report["diagnostics"] = diagnostics

    if getattr(args, "save_predictions", True):
        predictions_data = _build_supervised_prediction_rows(
            sample_rows_meta=sample_rows_meta,
            sample_labels=sample_labels,
            sample_preds=sample_preds,
            sample_probs=sample_probs,
            sample_logits=sample_logits,
            claim_labels=claim_labels,
            claim_preds=claim_preds,
            claim_probs=claim_probs,
            claim_logits=claim_logits,
            include_claims=False,
        )
        if posthoc_info.get("applied"):
            for row in predictions_data:
                sid = int(row.get("sample_id", -1))
                if sid < 0 or sid >= len(pre_posthoc_sample_probs):
                    continue
                row["raw_sample_probability"] = float(raw_sample_probs[sid])
                row["threshold_only_sample_probability"] = float(pre_posthoc_sample_probs[sid])
                row["pre_posthoc_sample_probability"] = float(pre_posthoc_sample_probs[sid])
                row["posthoc_sample_probability"] = float(sample_probs[sid])
                row["posthoc_applied"] = True
        else:
            for row in predictions_data:
                sid = int(row.get("sample_id", -1))
                if sid < 0 or sid >= len(pre_posthoc_sample_probs):
                    continue
                row["raw_sample_probability"] = float(raw_sample_probs[sid])
                row["threshold_only_sample_probability"] = float(pre_posthoc_sample_probs[sid])
                row["posthoc_applied"] = False

        predictions_path = os.path.join(output_dir, "predictions.json")
        write_predictions(predictions_path, {"rows": predictions_data})
        workflow.artifact("supervised.predictions.saved", predictions_path)
        report["predictions_path"] = predictions_path

    diagnostics_path = os.path.join(output_dir, "diagnostics.json")
    write_report(diagnostics_path, diagnostics, kind="diagnostics")
    workflow.artifact("supervised.diagnostics.saved", diagnostics_path)
    report["diagnostics_path"] = diagnostics_path

    workflow.event(
        "supervised.report.ready",
        total_samples=report["total_samples"],
        total_claims=report["total_claims"],
        sample_threshold=report["sample_threshold"],
        sample_temperature=report["sample_temperature"],
        tune_sample_count=report["tune_sample_count"],
        threshold_mode=report.get("threshold_protocol", {}).get("mode"),
        posthoc_applied=bool(report.get("posthoc", {}).get("applied")),
        posthoc_mode=report.get("posthoc", {}).get("mode"),
    )

    return report


def main():
    args = parse_args()
    if args.fit_thresholds_on_eval_split and str(args.split).strip().lower() == "test":
        raise ValueError(
            "Refusing to fit thresholds on test split (data leakage risk). "
            "Use validation (or a dedicated tune split/report) for threshold fitting."
        )
    cfg = Config()
    configure_logging(cfg, force=True)

    cache_dir = args.cache_dir or cfg.generation.cache_dir
    output_dir = args.output_dir or os.path.join(cfg.output.output_dir, "evaluation")
    os.makedirs(output_dir, exist_ok=True)

    n_hop_values = None
    if args.n_hop_values:
        n_hop_values = [int(x) for x in args.n_hop_values.split(",")]

    eval_ctx = _build_eval_context(cache_dir, args.split, cfg)
    preview_threshold, preview_threshold_note = _preview_applied_sample_threshold(args)
    workflow = WorkflowLogger(log, "eval", width=cfg.output.log_banner_width)
    runner = PipelineRunner(workflow)

    def _validate_supervised_output(report):
        if not isinstance(report, dict):
            raise TypeError("supervised stage output must be a dict")
        required = ("head_type", "overall_metrics", "threshold_protocol")
        missing = [k for k in required if k not in report]
        if missing:
            raise ValueError(f"supervised stage output missing keys: {missing}")

    runner.register_stage(
        StageSpec(
            name="supervised",
            fn=lambda payload: evaluate_supervised(
                args, cache_dir, n_hop_values, output_dir, eval_ctx, cfg
            ),
            output_contract=_validate_supervised_output,
            retries=0,
            start_fields=lambda payload: {
                "head_path": args.head_path,
                "output_dir": output_dir,
            },
            result_fields=lambda report: {
                "total_samples": report.get("total_samples", 0),
                "total_claims": report.get("total_claims", 0),
            },
        )
    )

    workflow.header(
        "ChainUQ: Evaluation",
        dataset=eval_ctx.dataset_name,
        cache_dir=cache_dir,
        split=args.split,
        difficulty=n_hop_values or "all",
        difficulty_field=eval_ctx.difficulty_field,
        head_path=args.head_path or "none",
        threshold=preview_threshold,
        threshold_source=preview_threshold_note,
        cli_threshold=float(args.threshold),
        tune_split=args.threshold_tune_split,
        require_tune_split=args.require_tune_split,
        tune_source=args.threshold_source_report or "none",
        output_dir=output_dir,
    )

    # Check if combined report already exists (skip unless --force)
    combined_path = os.path.join(output_dir, "combined_evaluation.json")
    if not args.force:
        existing = _load_existing_report(combined_path, "combined")
        if existing is not None:
            workflow.event("combined_report.skip", path=combined_path, reason="exists")
            return

    all_reports = {}

    # Evaluate supervised head
    if args.head_path:
        head_report_path = os.path.join(output_dir, "evaluation_report.json")
        if not args.force:
            existing_head = _load_existing_report(head_report_path, "head")
            if existing_head is not None:
                head_type = existing_head.get("head_type", "unknown")
                all_reports[head_type] = existing_head
                _write_posthoc_status(existing_head, head_report_path)
                workflow.event("supervised.skip", reason="existing_report", path=head_report_path, head_type=head_type)

        if args.head_path and not all_reports:
            head_report = runner.run("supervised")
            head_type = head_report["head_type"]
            all_reports[head_type] = head_report

            head_report_path = os.path.join(output_dir, "evaluation_report.json")
            write_report(head_report_path, head_report, kind="evaluation")
            status_path = _write_posthoc_status(head_report, head_report_path)
            workflow.artifact("supervised.report.saved", head_report_path, head_type=head_type)
            workflow.artifact("supervised.posthoc_status.saved", status_path, head_type=head_type)

    # Print summary
    workflow.header("Evaluation Summary", reports=len(all_reports), output_dir=output_dir)
    for method_name, report in all_reports.items():
        m = report["overall_metrics"]
        method_type = report.get("method_type", "unknown")
        workflow.event(
            "summary.row",
            method=method_name,
            method_type=method_type,
            accuracy=m.get("accuracy", 0),
            precision=m.get("precision", 0),
            recall=m.get("recall", 0),
            f1=m.get("f1", 0),
            roc_auc=m.get("roc_auc", 0),
            pr_auc=m.get("pr_auc", 0),
            ece=m.get("ece", 0),
        )

    # Save combined report
    combined_path = os.path.join(output_dir, "combined_evaluation.json")
    write_report(combined_path, all_reports, kind="evaluation_combined")
    workflow.artifact("combined_report.saved", combined_path)


if __name__ == "__main__":
    main()
