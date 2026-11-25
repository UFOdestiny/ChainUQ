#!/usr/bin/env python3
"""Trainer and TrainingArguments builders for ChainUQ scripts."""

import json
import os
import random
from collections import defaultdict
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Sampler
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput

from config import Config
from scripts.metrics import compute_all_metrics
from utils.log import get_logger

log = get_logger(__name__)


class ChunkGroupedSampler(Sampler):
    """Yields indices grouped by chunk file to preserve LRU cache locality.

    Within each epoch: shuffles chunk order + shuffles indices within each chunk.
    This prevents cache thrashing when chunks are very large (tens of GB each).
    """

    def __init__(self, dataset, seed: int = 42):
        self._chunks = defaultdict(list)
        for i, entry in enumerate(dataset._index):
            self._chunks[entry.get("chunk_file", "")].append(i)
        self._total = len(dataset)
        self._seed = seed
        self._epoch = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self._seed + self._epoch)
        chunk_keys = list(self._chunks.keys())
        rng.shuffle(chunk_keys)
        for key in chunk_keys:
            indices = self._chunks[key][:]
            rng.shuffle(indices)
            yield from indices

    def __len__(self):
        return self._total


def resolve_greater_is_better(
    metric_for_best_model: str,
    configured_greater_is_better: Optional[bool],
) -> bool:
    """Resolve whether larger metric values are better."""
    if configured_greater_is_better is not None:
        return configured_greater_is_better

    metric_name = (metric_for_best_model or "").lower()
    if metric_name.endswith("ece") or metric_name.endswith("loss"):
        return False
    return True


def build_train_training_arguments(
    cfg: Config,
    output_dir: str,
    warmup_steps: int,
) -> TrainingArguments:
    """Build TrainingArguments for training."""
    use_mixed_precision = torch.cuda.is_available()
    greater_is_better = resolve_greater_is_better(
        cfg.training.metric_for_best_model,
        cfg.training.greater_is_better,
    )
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg.training.num_epochs,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        learning_rate=cfg.training.learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=cfg.training.weight_decay,
        max_grad_norm=cfg.training.max_grad_norm,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        fp16=cfg.training.fp16 and use_mixed_precision,
        bf16=cfg.training.bf16 and use_mixed_precision,
        eval_strategy=cfg.training.eval_strategy,
        save_strategy=cfg.training.save_strategy,
        save_total_limit=cfg.training.save_total_limit,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=greater_is_better,
        seed=cfg.training.seed,
        report_to=cfg.training.report_to,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        dataloader_pin_memory=cfg.training.dataloader_pin_memory,
        dataloader_prefetch_factor=cfg.training.dataloader_prefetch_factor if cfg.training.dataloader_num_workers > 0 else None,
        dataloader_persistent_workers=cfg.training.dataloader_persistent_workers if cfg.training.dataloader_num_workers > 0 else False,
        dataloader_drop_last=cfg.training.dataloader_drop_last,
        logging_steps=cfg.training.logging_steps,
        logging_first_step=True,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        remove_unused_columns=False,
        label_names=["labels", "claim_labels"],
        logging_strategy=cfg.training.logging_strategy,
        disable_tqdm=cfg.training.disable_tqdm,
        accelerator_config={"dispatch_batches": False},
    )


class HeadTrainer(Trainer):
    """Custom trainer for conclusion-only supervised heads."""

    def _with_drop_last_disabled(self, loader_builder, *args, **kwargs):
        old_drop_last = self.args.dataloader_drop_last
        try:
            self.args.dataloader_drop_last = False
            return loader_builder(*args, **kwargs)
        finally:
            self.args.dataloader_drop_last = old_drop_last

    def _get_train_sampler(self, train_dataset=None):
        """Use ChunkGroupedSampler when the train dataset has chunk-based indices."""
        train_ds = train_dataset if train_dataset is not None else self.train_dataset
        if hasattr(train_ds, "_index") and train_ds._index:
            sampler = ChunkGroupedSampler(train_ds, seed=self.args.seed)
            epoch = int(self.state.epoch) if self.state and self.state.epoch is not None else 0
            sampler.set_epoch(epoch)
            return sampler
        return super()._get_train_sampler(train_ds)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """Override to handle variable-length claim labels.

        HuggingFace Trainer expects labels to be a fixed-size tensor per batch.
        Our claim_labels is a list of tensors (one per sample, variable claims).
        We flatten claim_labels to match the flattened logits from wrapper.py.
        """
        claim_labels = inputs.get("claim_labels")

        effective_prediction_loss_only = prediction_loss_only
        if claim_labels is not None and self.compute_metrics is not None:
            effective_prediction_loss_only = False

        loss, logits, _ = super().prediction_step(
            model, inputs, effective_prediction_loss_only, ignore_keys
        )

        if isinstance(logits, (tuple, list)):
            logits = next((x for x in logits if isinstance(x, torch.Tensor)), None)

        if logits is None:
            return loss, logits, None

        if claim_labels is not None and isinstance(claim_labels, (list, tuple)):
            label_parts = [cl.to(logits.device) for cl in claim_labels if isinstance(cl, torch.Tensor)]
            labels = torch.cat(label_parts, dim=0) if label_parts else None
        elif isinstance(claim_labels, torch.Tensor):
            labels = claim_labels.to(logits.device)
        else:
            labels = None

        return loss, logits, labels

    def evaluation_loop(self, dataloader, description, prediction_loss_only=None,
                        ignore_keys=None, metric_key_prefix="eval"):
        """Eval loop with both claim-level and sample-level metrics.

        We keep claim-level metrics for diagnostics, but expose sample-level metrics
        as the primary ``eval_*`` keys so model selection matches the final objective.
        """
        _ = description, prediction_loss_only
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()

        all_claim_logits, all_claim_labels = [], []
        all_sample_logits, all_sample_labels = [], []
        all_consistency_gap, all_consistency_gate = [], []
        all_consistency_losses = []
        all_gate_spread_losses = []
        all_losses = []
        observed_num_examples = 0

        for step, inputs in enumerate(dataloader):
            _ = step, ignore_keys
            inputs = self._prepare_inputs(inputs)

            with torch.no_grad():
                outputs = model(**inputs)

            batch_size = inputs["features"].shape[0]
            observed_num_examples += batch_size

            loss = getattr(outputs, "loss", None)
            if loss is not None:
                all_losses.append((loss.detach().float().item(), batch_size))

            consistency_loss = getattr(outputs, "consistency_loss", None)
            if isinstance(consistency_loss, torch.Tensor):
                all_consistency_losses.append((consistency_loss.detach().float().item(), batch_size))
            gate_spread_loss = getattr(outputs, "gate_spread_loss", None)
            if isinstance(gate_spread_loss, torch.Tensor):
                all_gate_spread_losses.append((gate_spread_loss.detach().float().item(), batch_size))

            claim_logits = getattr(outputs, "logits", None)
            if isinstance(claim_logits, (tuple, list)):
                claim_logits = next((x for x in claim_logits if isinstance(x, torch.Tensor)), None)
            if isinstance(claim_logits, torch.Tensor):
                if claim_logits.ndim > 1:
                    claim_logits = claim_logits.squeeze(-1)
                all_claim_logits.append(claim_logits.detach().view(-1).cpu())

            claim_labels = inputs.get("claim_labels")
            if isinstance(claim_labels, (list, tuple)):
                parts = [x.detach().view(-1).cpu() for x in claim_labels if isinstance(x, torch.Tensor)]
                if parts:
                    all_claim_labels.append(torch.cat(parts, dim=0))
            elif isinstance(claim_labels, torch.Tensor):
                all_claim_labels.append(claim_labels.detach().view(-1).cpu())

            sample_logits = getattr(outputs, "sample_logits", None)
            if isinstance(sample_logits, torch.Tensor):
                if sample_logits.ndim > 1:
                    sample_logits = sample_logits.squeeze(-1)
                all_sample_logits.append(sample_logits.detach().view(-1).cpu())

            consistency_gap = getattr(outputs, "consistency_gap", None)
            if isinstance(consistency_gap, torch.Tensor):
                all_consistency_gap.append(consistency_gap.detach().view(-1).cpu())

            consistency_gate = getattr(outputs, "consistency_gate", None)
            if isinstance(consistency_gate, torch.Tensor):
                all_consistency_gate.append(consistency_gate.detach().view(-1).cpu())

            labels = inputs.get("labels")
            if isinstance(labels, torch.Tensor):
                all_sample_labels.append(labels.detach().view(-1).cpu())

        cat_claim_logits = torch.cat(all_claim_logits, dim=0).numpy() if all_claim_logits else np.array([])
        cat_claim_labels = torch.cat(all_claim_labels, dim=0).numpy() if all_claim_labels else np.array([])
        cat_sample_logits = torch.cat(all_sample_logits, dim=0).numpy() if all_sample_logits else np.array([])
        cat_sample_labels = torch.cat(all_sample_labels, dim=0).numpy() if all_sample_labels else np.array([])

        sample_metrics = {}
        if cat_sample_logits.size > 0 and cat_sample_labels.size > 0:
            sample_metrics = compute_all_metrics(cat_sample_labels, cat_sample_logits)
            selected_threshold = float(sample_metrics.get("optimal_threshold", 0.5) or 0.5)
            sample_metrics = compute_all_metrics(
                cat_sample_labels,
                cat_sample_logits,
                threshold=selected_threshold,
            )
            sample_metrics.pop("optimal_threshold", None)
            sample_metrics.pop("optimal_f1", None)

        primary_metrics = sample_metrics
        metrics = {
            f"{metric_key_prefix}_{k}": v
            for k, v in primary_metrics.items()
        }

        if sample_metrics:
            for k, v in sample_metrics.items():
                metrics[f"{metric_key_prefix}_sample_{k}"] = v

        if all_losses:
            total_loss = sum(l * n for l, n in all_losses)
            total_n = sum(n for _, n in all_losses)
            metrics[f"{metric_key_prefix}_loss"] = total_loss / max(total_n, 1)
        if all_consistency_losses:
            total_consistency_loss = sum(l * n for l, n in all_consistency_losses)
            total_consistency_n = sum(n for _, n in all_consistency_losses)
            metrics[f"{metric_key_prefix}_consistency_loss"] = (
                total_consistency_loss / max(total_consistency_n, 1)
            )
        if all_gate_spread_losses:
            total_gate_spread_loss = sum(l * n for l, n in all_gate_spread_losses)
            total_gate_spread_n = sum(n for _, n in all_gate_spread_losses)
            metrics[f"{metric_key_prefix}_gate_spread_loss"] = (
                total_gate_spread_loss / max(total_gate_spread_n, 1)
            )
        if all_consistency_gap:
            cat_consistency_gap = torch.cat(all_consistency_gap, dim=0).numpy()
            metrics[f"{metric_key_prefix}_consistency_gap_mean"] = float(np.mean(cat_consistency_gap))
            metrics[f"{metric_key_prefix}_consistency_gap_std"] = float(np.std(cat_consistency_gap))
        if all_consistency_gate:
            cat_consistency_gate = torch.cat(all_consistency_gate, dim=0).numpy()
            metrics[f"{metric_key_prefix}_consistency_gate_mean"] = float(np.mean(cat_consistency_gate))
            metrics[f"{metric_key_prefix}_consistency_gate_std"] = float(np.std(cat_consistency_gate))
        metrics[f"{metric_key_prefix}_runtime"] = 0.0
        metrics[f"{metric_key_prefix}_samples_per_second"] = 0.0
        metrics[f"{metric_key_prefix}_steps_per_second"] = 0.0
        metrics[f"{metric_key_prefix}_samples"] = int(cat_sample_logits.shape[0]) if cat_sample_logits.size > 0 else observed_num_examples

        return EvalLoopOutput(
            predictions=cat_claim_logits,
            label_ids=cat_claim_labels,
            metrics=metrics,
            num_samples=observed_num_examples,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        """Never drop the last batch for eval; otherwise small eval sets can become empty."""
        return self._with_drop_last_disabled(super().get_eval_dataloader, eval_dataset)

    def get_test_dataloader(self, test_dataset):
        """Never drop the last batch for test-time metrics."""
        return self._with_drop_last_disabled(super().get_test_dataloader, test_dataset)

    def _determine_best_metric(self, metrics, trial):
        """Robust best-metric selection with fallback when configured metric is absent."""
        try:
            return super()._determine_best_metric(metrics=metrics, trial=trial)
        except KeyError:
            metric_name = self.args.metric_for_best_model or ""
            metric_key = metric_name if metric_name.startswith("eval_") else f"eval_{metric_name}"
            fallback_order = [
                "eval_pr_auc",
                "eval_roc_auc",
                "eval_f1",
                "eval_accuracy",
                "eval_loss",
                "eval_runtime",
            ]
            fallback_key = next((k for k in fallback_order if k in metrics), None)
            if fallback_key is None:
                log.warning(
                    "Best-metric key '%s' missing and no fallback metric found. "
                    "Skipping best-model update this evaluation step.",
                    metric_key,
                )
                return False

            old_metric = self.args.metric_for_best_model
            old_greater = self.args.greater_is_better
            self.args.metric_for_best_model = fallback_key
            if fallback_key.endswith("loss") or fallback_key.endswith("runtime"):
                self.args.greater_is_better = False
            elif old_greater is None:
                self.args.greater_is_better = True

            log.warning(
                "Best-metric key '%s' missing; temporarily using '%s'.",
                metric_key,
                fallback_key,
            )
            try:
                return super()._determine_best_metric(metrics=metrics, trial=trial)
            finally:
                self.args.metric_for_best_model = old_metric
                self.args.greater_is_better = old_greater

    def create_optimizer(self):
        """Only optimize trainable (head + feature extractor) parameters."""
        if self.optimizer is None:
            trainable = self.model.get_trainable_params()
            self.optimizer = torch.optim.AdamW(
                trainable,
                lr=self.args.learning_rate,
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def _save(self, output_dir=None, state_dict=None):
        """Save head (and feature extractor if present) artifacts."""
        _ = state_dict
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Keep HuggingFace checkpoint contract for resume_from_checkpoint.
        torch.save(self.model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        torch.save(self.model.head.state_dict(), os.path.join(output_dir, "head_weights.pth"))

        if hasattr(self.model, "feature_extractor"):
            torch.save(
                self.model.feature_extractor.state_dict(),
                os.path.join(output_dir, "fe_weights.pth"),
            )

        if hasattr(self, "_head_config"):
            with open(os.path.join(output_dir, "head_config.json"), "w", encoding="utf-8") as f:
                json.dump(self._head_config, f, indent=2)

        with open(os.path.join(output_dir, "training_args.json"), "w", encoding="utf-8") as f:
            f.write(self.args.to_json_string())

    def _load_best_model(self):
        """Load best checkpoint weights for head (+ feature extractor if present)."""
        best_path = self.state.best_model_checkpoint
        if best_path is None:
            if self.args.load_best_model_at_end:
                raise RuntimeError(
                    "load_best_model_at_end=True, but no best checkpoint was recorded."
                )
            log.warning("No best model checkpoint found, skipping load_best_model_at_end.")
            return

        log.info("Loading best model from %s", best_path)

        head_path = os.path.join(best_path, "head_weights.pth")
        if not os.path.isfile(head_path):
            raise FileNotFoundError(f"Expected head checkpoint not found: {head_path}")
        self.model.head.load_state_dict(torch.load(head_path, map_location="cpu"))
        log.info("  Loaded head weights from %s", head_path)

        fe_path = os.path.join(best_path, "fe_weights.pth")
        if hasattr(self.model, "feature_extractor") and os.path.isfile(fe_path):
            self.model.feature_extractor.load_state_dict(torch.load(fe_path, map_location="cpu"))
            log.info("  Loaded feature extractor weights from %s", fe_path)
