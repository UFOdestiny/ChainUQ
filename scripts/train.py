#!/usr/bin/env python3
"""
train.py - Phase 2: Train one supervised head on cached features.

Loads pre-extracted features from Phase 1 (no LLM needed), trains a
lightweight binary classification head with BCEWithLogitsLoss.

Usage:
    python scripts/train.py --head_type chainuq --cache_dir /path/to/cached_features
    python scripts/train.py --head_type uq_abl_v1 --num_epochs 50 --batch_size 32
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from transformers import EarlyStoppingCallback, TrainerCallback

from config import Config
from data.cached_features import CachedFeatureDataset
from models.heads import build_head
from models.wrapper import CachedFeatureModel
from utils.common import collate_claim_cached_features, make_binary_compute_metrics
from scripts.trainer import HeadTrainer, build_train_training_arguments
from utils.efficiency import (
    get_cpu_peak_memory_gb,
    get_gpu_peak_memory_gb,
    reset_gpu_peak_memory,
)
from utils.log import WorkflowLogger, configure_logging, get_logger, set_logger_level
from utils.pipeline import PipelineRunner, StageSpec
from utils.reporting import write_report, write_training_args
from utils.contracts import TrainResultReport

log = get_logger(__name__)

# Suppress noisy Accelerate INFO about dataset length
set_logger_level("accelerate.accelerator", "WARNING")


class StepHeartbeatCallback(TrainerCallback):
    """Emit trainer metrics into standard logs at each logging event."""

    def __init__(self):
        self._pending_train_log = None

    @staticmethod
    def _compact_value(value):
        if isinstance(value, float):
            return round(float(value), 6)
        return value

    def _flush_pending(self):
        if not self._pending_train_log:
            return
        step, epoch_text, payload = self._pending_train_log
        self._pending_train_log = None
        self._emit(step, epoch_text, payload)

    def _emit(self, step: int, epoch_text: str, payload: dict):
        parts = []
        for name in ("train", "eval", "counts"):
            section = payload.get(name, {})
            if section:
                parts.append(f"{name}={section}")
        if not parts:
            return
        log.info(
            "trainer_epoch_log step=%d epoch=%s %s",
            step,
            epoch_text,
            " ".join(parts),
        )

    def _build_payload(self, metrics: dict) -> dict:
        train_metrics = {}
        eval_metrics = {}
        counts = {}

        for key, value in metrics.items():
            if key in {"epoch", "total_flos"}:
                continue
            value = self._compact_value(value)
            if key.startswith("eval_sample_"):
                continue  # eval_* already exposes sample-level metrics as the primary surface
            if key.startswith("eval_"):
                short = key[len("eval_") :]
                if short in {"runtime", "samples_per_second", "steps_per_second"}:
                    continue
                if short in {"optimal_threshold", "optimal_f1"}:
                    continue
                if short in {"samples", "claims"}:
                    counts[short] = value
                else:
                    eval_metrics[short] = value
                continue

            short = "lr" if key == "learning_rate" else key
            train_metrics[short] = value

        return {
            "train": train_metrics,
            "eval": eval_metrics,
            "counts": counts,
        }

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = args, control, kwargs
        if not logs:
            return

        step = int(getattr(state, "global_step", 0))
        epoch = logs.get("epoch", state.epoch)
        if isinstance(epoch, (int, float)):
            epoch_text = f"{float(epoch):.4f}"
        else:
            epoch_text = str(epoch)

        metrics = {k: v for k, v in logs.items() if k != "total_flos"}
        payload = self._build_payload(metrics)
        has_eval = bool(payload["eval"] or payload["counts"])
        has_train = bool(payload["train"])

        if has_eval:
            if self._pending_train_log is not None:
                pending_step, pending_epoch_text, pending_payload = self._pending_train_log
                self._pending_train_log = None
                if pending_step == step and pending_epoch_text == epoch_text:
                    merged_train = dict(pending_payload.get("train", {}))
                    merged_train.update(payload.get("train", {}))
                    payload["train"] = merged_train
                else:
                    self._emit(pending_step, pending_epoch_text, pending_payload)
            self._emit(step, epoch_text, payload)
            return

        if has_train:
            self._flush_pending()
            self._pending_train_log = (step, epoch_text, payload)

    def on_train_end(self, args, state, control, **kwargs):
        _ = args, state, control, kwargs
        self._flush_pending()


def _count_trainable_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def _collect_dataset_log_fields(dataset: "CachedFeatureDataset") -> dict:
    sample_correct, sample_hall = dataset.label_distribution()
    sample_total = len(dataset)
    claim_stats = dataset.claim_distribution()
    filter_stats = dataset.filter_stats()
    total_claims = int(claim_stats.get("total_claims", 0) or 0)
    verified_claims = int(claim_stats.get("verified_claims", 0) or 0)
    correct_claims = int(claim_stats.get("correct_claims", 0) or 0)
    incorrect_claims = int(claim_stats.get("incorrect_claims", 0) or 0)
    reasoning_claims = int(claim_stats.get("reasoning_claims", 0) or 0)
    conclusion_claims = int(claim_stats.get("conclusion_claims", 0) or 0)

    return {
        "samples": sample_total,
        "sample_correct": sample_correct,
        "sample_hallucination": sample_hall,
        "sample_correct_rate": sample_correct / max(sample_total, 1),
        "sample_hallucination_rate": sample_hall / max(sample_total, 1),
        "claim_total": total_claims,
        "verified_claims": verified_claims,
        "pending_claims": int(claim_stats.get("pending_claims", 0) or 0),
        "claim_correct": correct_claims,
        "claim_hallucination": incorrect_claims,
        "claim_correct_rate": correct_claims / max(verified_claims, 1),
        "claim_hallucination_rate": incorrect_claims / max(verified_claims, 1),
        "reasoning_claims": reasoning_claims,
        "reasoning_correct_rate": int(claim_stats.get("reasoning_correct", 0) or 0) / max(reasoning_claims, 1),
        "conclusion_claims": conclusion_claims,
        "conclusion_correct_rate": int(claim_stats.get("conclusion_correct", 0) or 0) / max(conclusion_claims, 1),
        "removed_n_hops": int(filter_stats.get("removed_n_hops", 0) or 0),
        "removed_pending": int(filter_stats.get("removed_pending", 0) or 0),
        "removed_no_verified_claims": int(filter_stats.get("removed_no_usable_verified_claims", 0) or 0),
        "mem_budget_gb": getattr(dataset, "_mem_budget_gb", None),
    }


def _resolve_head_dim_for_target(
    *,
    head_type: str,
    feature_dim: int,
    num_classes: int,
    n_layers: int,
    n_heads: int,
    dropout: float,
    default_dim: int,
    target_params: int,
    min_dim: int = 96,
    max_dim: int = 640,
    step: int = 8,
) -> int:
    best_dim = int(default_dim)
    best_gap = None
    best_params = None

    for dim in range(min_dim, max_dim + 1, step):
        candidate = build_head(
            head_type=head_type,
            feature_dim=feature_dim,
            num_classes=num_classes,
            head_dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
        )
        params = _count_trainable_params(candidate)
        gap = abs(params - target_params)
        if best_gap is None or gap < best_gap:
            best_gap = gap
            best_dim = dim
            best_params = params
        del candidate

    log.info(
        "Auto-sized %s for ~%d params: head_dim=%d -> %s params",
        head_type,
        target_params,
        best_dim,
        f"{int(best_params):,}" if best_params is not None else "unknown",
    )
    return int(best_dim)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 2: Train a single supervised head")
    parser.add_argument("--head_type", type=str, default=None,
                        help="Single head type to train")
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output dir for this head")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--warmup_steps", type=int, default=None)
    parser.add_argument("--n_hop_values", type=str, default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--loss_type",
        type=str,
        default=None,
        choices=["bce", "balanced_bce", "focal"],
        help="Binary loss type for training.",
    )
    parser.add_argument(
        "--loss_pos_weight",
        type=float,
        default=None,
        help="Positive class weight for balanced_bce.",
    )
    parser.add_argument(
        "--focal_gamma",
        type=float,
        default=None,
        help="Focal loss gamma when --loss_type focal.",
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default=None,
        help="Experiment tracking: none, tensorboard, or all",
    )
    parser.add_argument("--sample_pos_weight", type=float, default=None)
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint dir, or 'auto' to find latest checkpoint in output_dir",
    )
    return parser.parse_args()


def _parse_csv_items(raw: str) -> List[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_int_csv(raw: str) -> List[int]:
    return [int(item) for item in _parse_csv_items(raw)]


def _coerce_like(raw_value: str, current_value):
    if isinstance(current_value, bool):
        return raw_value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(current_value, int):
        return int(raw_value)
    if isinstance(current_value, float):
        return float(raw_value)
    return raw_value


def _set_cfg_value(cfg: Config, section: str, field: str, raw_value: str):
    sub_cfg = getattr(cfg, section)
    current = getattr(sub_cfg, field)
    setattr(sub_cfg, field, _coerce_like(raw_value, current))


def apply_overrides(cfg: Config, args):
    """Apply CLI overrides to config."""

    if args.head_type:
        cfg.head.head_type = args.head_type
    if args.cache_dir:
        cfg.generation.cache_dir = args.cache_dir
    if args.output_dir:
        cfg.output.output_dir = args.output_dir
    if args.num_epochs:
        cfg.training.num_epochs = args.num_epochs
    if args.batch_size:
        cfg.training.per_device_train_batch_size = args.batch_size
        cfg.training.per_device_eval_batch_size = args.batch_size
    if args.learning_rate:
        cfg.training.learning_rate = args.learning_rate
    if args.warmup_steps is not None:
        cfg.training.warmup_steps = args.warmup_steps
    if args.n_hop_values:
        cfg.dataset.n_hop_values = _parse_int_csv(args.n_hop_values)
    if args.max_train_samples is not None:
        cfg.dataset.max_train_samples = args.max_train_samples
    if args.seed is not None:
        cfg.training.seed = args.seed
    if args.loss_type:
        cfg.training.loss_type = args.loss_type
    if args.loss_pos_weight is not None:
        cfg.training.loss_pos_weight = args.loss_pos_weight
    if args.focal_gamma is not None:
        cfg.training.focal_gamma = args.focal_gamma
    if args.report_to:
        cfg.training.report_to = args.report_to
    if args.sample_pos_weight is not None:
        cfg.training.sample_pos_weight = args.sample_pos_weight
    return cfg


def list_checkpoints(output_dir: str) -> list[str]:
    """List checkpoint directories sorted by step (descending)."""
    if not os.path.isdir(output_dir):
        return []

    checkpoint_dirs = []
    for name in os.listdir(output_dir):
        if not name.startswith("checkpoint-"):
            continue
        try:
            step = int(name.split("-", 1)[1])
        except (IndexError, ValueError):
            continue
        path = os.path.join(output_dir, name)
        if os.path.isdir(path):
            checkpoint_dirs.append((step, path))

    checkpoint_dirs.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in checkpoint_dirs]


def ensure_hf_checkpoint_compatible(checkpoint_dir: str) -> bool:
    """Check if checkpoint contains HF-required model file for resume."""
    weights_file = os.path.join(checkpoint_dir, "pytorch_model.bin")
    safe_weights_file = os.path.join(checkpoint_dir, "model.safetensors")
    return os.path.isfile(weights_file) or os.path.isfile(safe_weights_file)


def find_latest_resumable_checkpoint(output_dir: str) -> Optional[str]:
    """Find latest checkpoint that can be resumed by HF Trainer."""
    checkpoints = list_checkpoints(output_dir)
    if not checkpoints:
        log.info("No checkpoints found in %s", output_dir)
        return None

    log.info("Found %d checkpoint(s) in %s", len(checkpoints), output_dir)
    for checkpoint_dir in checkpoints:
        if ensure_hf_checkpoint_compatible(checkpoint_dir):
            trainer_state = os.path.join(checkpoint_dir, "trainer_state.json")
            if os.path.isfile(trainer_state):
                with open(trainer_state, "r") as f:
                    state = json.load(f)
                log.info(
                    "  Valid checkpoint: %s (step=%d, epoch=%.2f)",
                    os.path.basename(checkpoint_dir),
                    state.get("global_step", -1),
                    state.get("epoch", -1),
                )
            else:
                log.info("  Valid checkpoint: %s (no trainer_state.json)", os.path.basename(checkpoint_dir))
            return checkpoint_dir
        else:
            log.warning("  Skipping incompatible checkpoint: %s", os.path.basename(checkpoint_dir))

    return None


def train_single_head(
    cfg: Config,
    train_ds: "CachedFeatureDataset",
    val_ds: "CachedFeatureDataset",
    feature_dim: int,
    resume_from_checkpoint: Optional[str] = None,
) -> TrainResultReport:
    """Train a single head using pre-loaded datasets."""
    workflow = WorkflowLogger(log, "train", width=cfg.output.log_banner_width)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = cfg.output.output_dir
    os.makedirs(output_dir, exist_ok=True)
    cache_dir = cfg.generation.cache_dir

    workflow.stage_start(
        "fit",
        device=device.type,
        output_dir=output_dir,
        cache_dir=cache_dir,
        feature_dim=feature_dim,
        resume_from_checkpoint=resume_from_checkpoint or "none",
    )

    train_summary = _collect_dataset_log_fields(train_ds)
    val_summary = _collect_dataset_log_fields(val_ds)
    workflow.event("train_split.summary", split="train", **train_summary)
    workflow.event("validation_split.summary", split="validation", **val_summary)

    n_correct_sample = int(train_summary["sample_correct"])
    n_hall_sample = int(train_summary["sample_hallucination"])
    n_correct = n_correct_sample
    n_hall = n_hall_sample
    total_claims = int(train_summary.get("claim_total", 0) or 0)
    verified_total = int(train_summary.get("verified_claims", 0) or 0)
    verified_correct = int(train_summary.get("claim_correct", 0) or 0)
    verified_incorrect = int(train_summary.get("claim_hallucination", 0) or 0)
    if total_claims > 0:
        if verified_total > 0:
            n_correct = verified_correct
            n_hall = verified_incorrect
        else:
            n_correct = verified_correct or n_correct_sample
            n_hall = total_claims - n_correct

    # Build head
    num_classes = cfg.head.num_classes
    workflow.stage_start(
        "model_build",
        head_type=cfg.head.head_type,
        num_classes=num_classes,
        target_params_m=cfg.head.target_params_m,
    )
    target_params = int(round(cfg.head.target_params_m * 1_000_000))
    resolved_head_dim = _resolve_head_dim_for_target(
        head_type=cfg.head.head_type,
        feature_dim=feature_dim,
        num_classes=num_classes,
        n_layers=cfg.head.n_layers,
        n_heads=cfg.head.n_heads,
        dropout=cfg.head.dropout,
        default_dim=cfg.head.head_dim,
        target_params=target_params,
        min_dim=cfg.head.head_dim_search_min,
        max_dim=cfg.head.head_dim_search_max,
        step=cfg.head.head_dim_search_step,
    )
    head = build_head(
        head_type=cfg.head.head_type,
        feature_dim=feature_dim,
        num_classes=num_classes,
        head_dim=resolved_head_dim,
        n_layers=cfg.head.n_layers,
        n_heads=cfg.head.n_heads,
        dropout=cfg.head.dropout,
    ).to(device)

    head_trainable_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    log.info("Head trainable parameters: %s", f"{head_trainable_params:,}")

    # Build model
    loss_pos_weight = float(cfg.training.loss_pos_weight)
    if loss_pos_weight <= 0 and cfg.training.loss_type in ("balanced_bce", "focal"):
        # Auto-calculate from label distribution
        raw_loss_pos_weight = float(n_hall / max(n_correct, 1))
        loss_pos_weight = float(min(raw_loss_pos_weight, cfg.training.max_auto_pos_weight))
        log.info(
            "Auto-calculated loss_pos_weight=%.3f (raw=%.3f, cap=%.3f) from label ratio",
            loss_pos_weight,
            raw_loss_pos_weight,
            cfg.training.max_auto_pos_weight,
        )

    sample_pos_weight = float(cfg.training.sample_pos_weight)
    if sample_pos_weight <= 0 and cfg.training.loss_type in ("balanced_bce", "focal"):
        raw_sample_pos_weight = float(n_hall_sample / max(n_correct_sample, 1))
        sample_pos_weight = float(min(raw_sample_pos_weight, cfg.training.max_auto_pos_weight))
        log.info(
            "Auto-calculated sample_pos_weight=%.3f (raw=%.3f, cap=%.3f) from sample label ratio",
            sample_pos_weight,
            raw_sample_pos_weight,
            cfg.training.max_auto_pos_weight,
        )

    model = CachedFeatureModel(
        head=head,
        num_classes=num_classes,
        loss_type=cfg.training.loss_type,
        pos_weight=loss_pos_weight,
        focal_gamma=cfg.training.focal_gamma,
        label_smoothing=cfg.training.label_smoothing,
        sample_pos_weight=sample_pos_weight,
    )
    trainable_params = model.count_trainable_params()
    log.info("Total trainable parameters (head only): %s", f"{trainable_params:,}")

    head_config = {
        "head_type": cfg.head.head_type,
        "feature_dim": feature_dim,
        "num_classes": num_classes,
        "head_dim": resolved_head_dim,
        "n_layers": cfg.head.n_layers,
        "n_heads": cfg.head.n_heads,
        "dropout": cfg.head.dropout,
        "target_head_params": target_params,
        "cache_dir": cache_dir,
        "trainable_params": trainable_params,
        "supervision_mode": "conclusion_only",
        "loss_type": cfg.training.loss_type,
        "loss_pos_weight": loss_pos_weight,
        "focal_gamma": cfg.training.focal_gamma,
        "sample_pos_weight": float(sample_pos_weight),
    }

    # Training arguments
    # Let HuggingFace Trainer handle epoch/step accounting via num_train_epochs.
    # Previously we computed max_steps manually using ceil(), which disagreed with
    # the Trainer's dataloader length when drop_last=True (floor), doubling epochs.
    warmup_steps = cfg.training.warmup_steps
    if warmup_steps <= 0 and cfg.training.warmup_ratio > 0:
        world_size = max(1, int(os.environ.get("WORLD_SIZE", "1")))
        effective_batch = (
            cfg.training.per_device_train_batch_size
            * cfg.training.gradient_accumulation_steps
            * world_size
        )
        steps_per_epoch = max(1, len(train_ds) // max(1, effective_batch))
        est_total_steps = int(cfg.training.num_epochs * steps_per_epoch)
        warmup_steps = int(est_total_steps * cfg.training.warmup_ratio)
    log.info(
        "Scheduler setup: num_epochs=%d, warmup_steps=%d",
        cfg.training.num_epochs,
        warmup_steps,
    )

    training_args = build_train_training_arguments(
        cfg=cfg,
        output_dir=output_dir,
        warmup_steps=warmup_steps,
    )
    log.info(
        "Trainer logging config: strategy=%s, steps=%s, disable_tqdm=%s",
        training_args.logging_strategy,
        training_args.logging_steps,
        training_args.disable_tqdm,
    )

    # Create Trainer
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=cfg.training.early_stopping_patience
        ),
        StepHeartbeatCallback(),
    ]
    trainer = HeadTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collate_claim_cached_features,
        compute_metrics=make_binary_compute_metrics(),
        callbacks=callbacks,
    )
    trainer._head_config = head_config

    # Remove default PrinterCallback to avoid duplicate dict output on stdout
    from transformers import PrinterCallback
    trainer.remove_callback(PrinterCallback)

    training_args_path = os.path.join(output_dir, "training_args.json")
    write_training_args(training_args_path, training_args.to_dict())
    workflow.stage_end(
        "model_build",
        head_dim=resolved_head_dim,
        head_trainable_params=head_trainable_params,
        total_trainable_params=trainable_params,
        warmup_steps=warmup_steps,
    )
    workflow.artifact("training_args.saved", training_args_path)

    # Resolve checkpoint for resume
    actual_checkpoint = None
    if resume_from_checkpoint:
        if resume_from_checkpoint.lower() == "auto":
            actual_checkpoint = find_latest_resumable_checkpoint(output_dir)
            if actual_checkpoint:
                log.info("Auto-detected checkpoint: %s", actual_checkpoint)
            else:
                log.info("No checkpoint found in %s, starting from scratch", output_dir)
        elif os.path.isdir(resume_from_checkpoint):
            if ensure_hf_checkpoint_compatible(resume_from_checkpoint):
                actual_checkpoint = resume_from_checkpoint
                log.info("Resuming from specified checkpoint: %s", actual_checkpoint)
            else:
                log.warning(
                    "Specified checkpoint is not HF-compatible: %s. Starting from scratch.",
                    resume_from_checkpoint,
                )
        else:
            log.warning("Checkpoint path not found: %s, starting from scratch", resume_from_checkpoint)

    # Train
    if actual_checkpoint:
        workflow.event("fit.resume", checkpoint=actual_checkpoint)
    else:
        workflow.event("fit.resume", checkpoint="scratch")
    reset_gpu_peak_memory(device)
    train_result = trainer.train(resume_from_checkpoint=actual_checkpoint)
    train_cpu_peak_gb = get_cpu_peak_memory_gb()
    train_gpu_peak_gb = get_gpu_peak_memory_gb(device)

    train_metrics = train_result.metrics
    if "total_flos" not in train_metrics:
        train_metrics["total_flos"] = trainer.state.total_flos
    workflow.metrics("fit.train_metrics", train_metrics, prefixes_to_strip=("train_",))

    log_history = trainer.state.log_history

    # Final evaluation
    eval_metrics = trainer.evaluate()
    workflow.metrics(
        "fit.eval_metrics",
        eval_metrics,
        include=(
            "eval_accuracy",
            "eval_precision",
            "eval_recall",
            "eval_f1",
            "eval_pr_auc",
            "eval_roc_auc",
            "eval_ece",
            "eval_threshold",
            "eval_samples",
            "eval_loss",
        ),
        prefixes_to_strip=("eval_",),
    )

    train_runtime_s = float(train_metrics.get("train_runtime", 0.0) or 0.0)
    epochs_completed = float(train_metrics.get("epoch", cfg.training.num_epochs) or 0.0)
    total_flos = float(train_metrics.get("total_flos", 0.0) or 0.0)
    flops_source = "trainer"
    if total_flos <= 0.0:
        estimated_flos = (
            6.0
            * float(trainable_params)
            * float(len(train_ds))
            * max(epochs_completed, 0.0)
        )
        if estimated_flos > 0.0:
            total_flos = estimated_flos
            flops_source = "estimated_head_only"

    efficiency = {
        "params_m": trainable_params / 1e6,
        "flops_g": total_flos / 1e9 if total_flos else 0.0,
        "flops_source": flops_source,
        "train_time_h": train_runtime_s / 3600 if train_runtime_s else 0.0,
        "epoch_s": (train_runtime_s / epochs_completed) if epochs_completed else 0.0,
        "inference_s": float(eval_metrics.get("eval_runtime", 0.0) or 0.0),
        "cpu_memory_gb": train_cpu_peak_gb,
        "gpu_memory_gb": train_gpu_peak_gb,
        "epochs_completed": epochs_completed,
    }
    workflow.metrics("fit.efficiency", efficiency)

    # Save best model
    if cfg.output.save_final_model:
        final_dir = os.path.join(output_dir, cfg.output.final_model_subdir)
        trainer._save(output_dir=final_dir)
        workflow.artifact("model.saved", final_dir, artifact="best_model")

    # Save consolidated results
    results = {
        "head_type": cfg.head.head_type,
        "num_classes": num_classes,
        "cache_dir": cache_dir,
        "trainable_params": trainable_params,
        "n_hop_values": cfg.dataset.n_hop_values,
        "train_samples": len(train_ds),
        "eval_samples": len(val_ds),
        "train_dataset_filter_stats": train_ds.filter_stats(),
        "eval_dataset_filter_stats": val_ds.filter_stats(),
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "efficiency": efficiency,
        "log_history": log_history,
        "head_config": head_config,
    }
    results_path = os.path.join(output_dir, "train_results.json")
    write_report(results_path, results, kind="train_results")
    results["results_path"] = results_path
    if cfg.output.save_final_model:
        results["final_model_dir"] = final_dir
    workflow.artifact("results.saved", results_path)
    workflow.stage_end(
        "fit",
        best_checkpoint=trainer.state.best_model_checkpoint or "none",
        train_runtime_s=train_runtime_s,
        eval_runtime_s=float(eval_metrics.get("eval_runtime", 0.0) or 0.0),
    )

    # Clean up model/trainer before exit
    del trainer, model, head
    torch.cuda.empty_cache()

    return results


def load_datasets(cfg: Config):
    """Load training and validation datasets for a single head training run."""
    cache_dir = cfg.generation.cache_dir
    n_hop_values = cfg.dataset.n_hop_values if cfg.dataset.n_hop_values else None

    # Avoid DataLoader worker memory duplication with large chunk caches
    if cfg.training.dataloader_num_workers > 0:
        log.info("Forcing dataloader_num_workers=0 for cached-feature training to save memory")
        cfg.training.dataloader_num_workers = 0
        cfg.training.dataloader_persistent_workers = False

    log.info("Loading cached feature datasets from %s", cache_dir)

    train_ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        split="train",
        n_hop_values=n_hop_values,
        max_samples=cfg.dataset.max_train_samples,
        preload_all=False,
        mem_budget_gb=cfg.cache.train_mem_budget_gb,
        load_reasoning_sidecar=False,
    )
    val_ds = CachedFeatureDataset(
        cache_dir=cache_dir,
        split="validation",
        n_hop_values=n_hop_values,
        max_samples=cfg.dataset.max_eval_samples,
        preload_all=False,
        mem_budget_gb=cfg.cache.val_mem_budget_gb,
        load_reasoning_sidecar=False,
    )

    if len(train_ds) == 0:
        raise ValueError(
            "No training samples remain after pending/incomplete-claim filtering. "
            "Likely labels are still pending or some samples are only partially judged. "
            "Run scripts/judge.py on the cache before training."
        )
    if len(val_ds) == 0:
        raise ValueError(
            "No validation samples remain after pending/incomplete-claim filtering. "
            "Likely labels are still pending or some samples are only partially judged. "
            "Run scripts/judge.py on the cache before training."
        )

    feature_dim = int(train_ds._manifest.get("feature_dim", 0))
    if feature_dim <= 0:
        raise ValueError(
            "feature_dim not found in train split manifest.json. "
            "Re-run scripts/generate.py to regenerate the cache."
        )

    return train_ds, val_ds, feature_dim


def main():
    args = parse_args()
    cfg = Config()
    cfg = apply_overrides(cfg, args)
    configure_logging(
        cfg,
        force=True,
        logger_levels={"accelerate.accelerator": "WARNING"},
    )
    workflow = WorkflowLogger(log, "train", width=cfg.output.log_banner_width)
    runner = PipelineRunner(workflow)

    # Single head mode
    workflow.header(
        "ChainUQ Phase 2: Single-Head Training",
        head_type=cfg.head.head_type,
        dataset=cfg.dataset.dataset_name,
        cache_dir=cfg.generation.cache_dir,
        n_hop_values=cfg.dataset.n_hop_values or "all",
        epochs=cfg.training.num_epochs,
        train_batch_size=cfg.training.per_device_train_batch_size,
        eval_batch_size=cfg.training.per_device_eval_batch_size,
        grad_accum=cfg.training.gradient_accumulation_steps,
        learning_rate=cfg.training.learning_rate,
        loss=cfg.training.loss_type,
        auto_pos_cap=cfg.training.max_auto_pos_weight,
        output_dir=cfg.output.output_dir,
        resume_from_checkpoint=args.resume_from_checkpoint or "none",
    )

    runner.register_stage(
        StageSpec(
            name="load_datasets",
            fn=lambda payload: load_datasets(cfg),
            start_fields=lambda payload: {
                "train_split": "train",
                "eval_split": "validation",
                "train_mem_budget_gb": cfg.cache.train_mem_budget_gb,
                "val_mem_budget_gb": cfg.cache.val_mem_budget_gb,
            },
            result_fields=lambda result: {
                "train_samples": len(result[0]),
                "val_samples": len(result[1]),
                "feature_dim": result[2],
            },
        )
    )
    runner.register_stage(
        StageSpec(
            name="run",
            fn=lambda payload: train_single_head(
                cfg=cfg,
                train_ds=payload["train_ds"],
                val_ds=payload["val_ds"],
                feature_dim=payload["feature_dim"],
                resume_from_checkpoint=args.resume_from_checkpoint,
            ),
            start_fields=lambda payload: {"head_type": cfg.head.head_type},
            result_fields=lambda payload: {
                "accuracy": payload["eval_metrics"].get("eval_accuracy", 0),
                "f1": payload["eval_metrics"].get("eval_f1", 0),
                "pr_auc": payload["eval_metrics"].get("eval_pr_auc", 0),
                "roc_auc": payload["eval_metrics"].get("eval_roc_auc", 0),
                "ece": payload["eval_metrics"].get("eval_ece", 0),
                "results_path": payload.get("results_path"),
            },
        )
    )

    train_ds, val_ds, feature_dim = runner.run("load_datasets")
    runner.run(
        "run",
        {
            "train_ds": train_ds,
            "val_ds": val_ds,
            "feature_dim": feature_dim,
        },
    )


if __name__ == "__main__":
    main()
