"""Shared utilities for data collation, metric computation, and model loading."""

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from scripts.metrics import compute_all_metrics
from utils.efficiency import load_model_with_dtype
from utils.log import get_logger

log = get_logger(__name__)

def _flatten_to_1d_numeric(x):
    """Recursively flatten nested arrays / tensors into a 1-D numpy array."""
    if isinstance(x, np.ndarray):
        if x.dtype == object:
            out = []
            for v in x:
                out.extend(_flatten_to_1d_numeric(v).tolist())
            return np.asarray(out)
        return x.reshape(-1)
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().reshape(-1)
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_flatten_to_1d_numeric(v).tolist())
        return np.asarray(out)
    return np.asarray([x])

def collate_cached_features(batch):
    """Collate pre-extracted feature dicts, padding variable-length sequences."""
    seq_lens = [b["features"].shape[0] for b in batch]
    all_same_len = len(set(seq_lens)) == 1

    if all_same_len:
        return {
            "features": torch.stack([b["features"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch]),
        }

    padded_features = pad_sequence(
        [b["features"] for b in batch], batch_first=True, padding_value=0.0,
    )
    padded_masks = pad_sequence(
        [b["attention_mask"] for b in batch], batch_first=True, padding_value=0,
    )
    return {
        "features": padded_features,
        "attention_mask": padded_masks,
        "labels": torch.stack([b["labels"] for b in batch]),
    }


def collate_claim_cached_features(batch):
    """Collate cached features for conclusion-only supervision."""
    seq_lens = [b["features"].shape[0] for b in batch]
    all_same_len = len(set(seq_lens)) == 1

    if all_same_len:
        features = torch.stack([b["features"] for b in batch])
        attention_mask = torch.stack([b["attention_mask"] for b in batch])
    else:
        features = pad_sequence(
            [b["features"] for b in batch], batch_first=True, padding_value=0.0,
        )
        attention_mask = pad_sequence(
            [b["attention_mask"] for b in batch], batch_first=True, padding_value=0,
        )

    base = {
        "features": features,
        "attention_mask": attention_mask,
        "labels": torch.stack([b["labels"] for b in batch]),
    }
    seq_len = features.shape[1]

    sample_questions = []
    sample_generated_texts = []
    sample_n_hops = []
    sample_ids = []
    sample_uids = []
    sample_reasoning_claim_counts = []
    sample_reasoning_verifieds = []
    sample_reasoning_feature_stats = []
    sample_reasoning_feature_means = []
    sample_token_probs = []
    sample_log_likelihoods = []

    def _dummy_unlabeled_claim(item):
        mask = torch.zeros(seq_len, dtype=torch.float32)
        valid_len = int(item["attention_mask"].sum().item())
        if valid_len > 0:
            mask[:valid_len] = 1.0
        return (
            mask.unsqueeze(0),
            torch.tensor([1], dtype=torch.long),
            torch.tensor([-1.0], dtype=torch.float32),
        )

    claim_masks, claim_types, claim_labels = [], [], []
    claim_records = []

    for item in batch:
        sample_questions.append(item.get("question", ""))
        sample_generated_texts.append(item.get("generated_text", ""))
        sample_n_hops.append(int(item.get("n_hops", 0)))
        sample_ids.append(int(item.get("sample_id", -1)))
        sample_uids.append(str(item.get("sample_uid", "")))
        sample_reasoning_claim_counts.append(len(item.get("reasoning_claims", []) or []))
        sample_reasoning_verifieds.append(list(item.get("reasoning_verified", []) or []))
        sample_reasoning_feature_stats.append(item.get("reasoning_feature_stats"))
        sample_reasoning_feature_means.append(item.get("reasoning_feature_mean"))
        sample_token_probs.append(item.get("token_probs"))
        sample_log_likelihoods.append(item.get("log_likelihoods"))

        conclusion_entry = None
        claim = item.get("conclusion_claim")
        label_val = item.get("conclusion_verified", -1)
        if isinstance(claim, dict):
            mask = torch.zeros(seq_len, dtype=torch.float32)
            for tid in claim.get("aligned_token_ids", []):
                if 0 <= int(tid) < seq_len:
                    mask[int(tid)] = 1.0
            if mask.sum() > 0:
                safe_label = float(label_val) if label_val in (0, 1) else -1.0
                claim_record = {
                    "claim_text": str(claim.get("claim", "") or claim.get("text", "")),
                    "claim_type": "conclusion",
                    "claim_type_id": 1,
                    "n_hops": int(item.get("n_hops", 0)),
                }
                conclusion_entry = (mask, 1, safe_label, claim_record)
            else:
                log.warning(
                    "Dropping conclusion claim with empty token alignment (dataset=%s, n_hops=%s).",
                    item.get("dataset_name", ""),
                    item.get("n_hops", ""),
                )

        if conclusion_entry is None:
            mask, types, labels = _dummy_unlabeled_claim(item)
            conclusion_entry = (mask[0], int(types[0].item()), float(labels[0].item()), None)

        ordered_entries = [conclusion_entry]
        masks = [entry[0] for entry in ordered_entries]
        types = [entry[1] for entry in ordered_entries]
        labels = [entry[2] for entry in ordered_entries]
        records = [entry[3] for entry in ordered_entries]

        if masks:
            claim_masks.append(torch.stack(masks))
            claim_types.append(torch.tensor(types, dtype=torch.long))
            claim_labels.append(torch.tensor(labels, dtype=torch.float32))
            claim_records.append(records)
        else:
            mask, types, labels = _dummy_unlabeled_claim(item)
            claim_masks.append(mask)
            claim_types.append(types)
            claim_labels.append(labels)
            claim_records.append([None])

    max_claims = max((cm.shape[0] for cm in claim_masks), default=1)
    for idx in range(len(claim_masks)):
        n_claims = claim_masks[idx].shape[0]
        if n_claims >= max_claims:
            continue

        pad_n = max_claims - n_claims
        pad_masks = torch.zeros((pad_n, seq_len), dtype=claim_masks[idx].dtype)
        pad_types = torch.full((pad_n,), 2, dtype=claim_types[idx].dtype)
        pad_labels = torch.full((pad_n,), -1.0, dtype=claim_labels[idx].dtype)

        claim_masks[idx] = torch.cat([claim_masks[idx], pad_masks], dim=0)
        claim_types[idx] = torch.cat([claim_types[idx], pad_types], dim=0)
        claim_labels[idx] = torch.cat([claim_labels[idx], pad_labels], dim=0)
        claim_records[idx].extend([None] * pad_n)

    base["claim_masks"] = torch.stack(claim_masks)
    base["claim_types"] = torch.stack(claim_types)
    base["claim_labels"] = torch.stack(claim_labels)
    base["claim_records"] = claim_records
    base["sample_questions"] = sample_questions
    base["sample_generated_texts"] = sample_generated_texts
    base["sample_n_hops"] = sample_n_hops
    base["sample_ids"] = sample_ids
    base["sample_uids"] = sample_uids
    base["sample_reasoning_claim_counts"] = sample_reasoning_claim_counts
    base["sample_reasoning_verifieds"] = sample_reasoning_verifieds
    base["sample_reasoning_feature_stats"] = sample_reasoning_feature_stats
    base["sample_reasoning_feature_means"] = sample_reasoning_feature_means
    base["sample_token_probs"] = sample_token_probs
    base["sample_log_likelihoods"] = sample_log_likelihoods
    return base

def make_binary_compute_metrics():
    """Return a ``compute_metrics`` callable compatible with HF Trainer."""

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = _flatten_to_1d_numeric(logits)

        label_candidates = []
        if isinstance(labels, (tuple, list)):
            for part in labels:
                flat = _flatten_to_1d_numeric(part)
                if flat.size > 0:
                    label_candidates.append(flat)
        else:
            label_candidates.append(_flatten_to_1d_numeric(labels))

        if not label_candidates:
            return {
                k: float("nan")
                for k in [
                    "accuracy", "precision", "recall", "f1",
                    "roc_auc", "pr_auc", "ece",
                ]
            }

        # Pick the candidate whose length best matches logits
        labels = min(label_candidates, key=lambda arr: abs(arr.size - logits.size))
        if labels.size != logits.size:
            log.error(
                "Metric length mismatch (labels=%d, logits=%d); this indicates a "
                "pipeline bug — check claim-level vs sample-level alignment.",
                labels.size, logits.size,
            )
            min_len = min(labels.size, logits.size)
            labels = labels[:min_len]
            logits = logits[:min_len]

        return compute_all_metrics(labels, logits)

    return compute_metrics

def resolve_torch_dtype(dtype_name: str) -> torch.dtype:
    """Map a string dtype name to its ``torch.dtype``."""
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return dtype_map.get(dtype_name, torch.float16)


def load_tokenizer_from_path(
    tokenizer_cls, model_path, trust_remote_code, cache_dir, padding_side,
):
    """Load a HF tokenizer and configure padding."""
    tokenizer = tokenizer_cls.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, cache_dir=cache_dir,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def load_llm_from_path(
    model_cls,
    model_path,
    torch_dtype_name,
    device_map,
    trust_remote_code,
    cache_dir,
    attn_implementation="eager",
):
    """Load a HF causal-LM, resolving the dtype string first."""
    torch_dtype = resolve_torch_dtype(torch_dtype_name)
    return load_model_with_dtype(
        model_cls.from_pretrained,
        model_path,
        torch_dtype,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
        cache_dir=cache_dir,
        attn_implementation=attn_implementation,
    )
