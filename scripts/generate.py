#!/usr/bin/env python3
"""Phase 1: Generate LLM responses, extract features, cache to disk.

For each dataset sample:
  1. Build chat prompt via dataset's build_chat_messages()
  2. Generate tokens via engine backend (vLLM hybrid or pure HF)
  3. Parse final answer via dataset's parse_answer()
  4. Extract/align claims from generated chain-of-thought
  5. Build claim-level labels (verified) + sample-level summary label
  6. Extract hidden states, token probs, attention weights
  7. Save claim-level cached chunks to disk

Usage:
    python scripts/generate.py --split train --backend vllm
    python scripts/generate.py --split train,validation,test --max_samples 100,50,50
    python scripts/generate.py --dataset hotpotqa --split test
"""

import sys
import gc
import argparse
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from config import Config, GLOBAL_SEED
from data.claims import (
    CLAIM_TYPE_IDS,
    extract_claims_from_generation,
    make_claim_stable_key,
    trim_reasoning_claims,
)
from data.datasets import get_dataset
from models.features.hidden_states import HiddenStateExtractor
from models.features.token_probs import TokenProbExtractor
from models.features.attention import AttentionExtractor
from models.features.combined import CombinedExtractor
from engine import get_engine
from utils.cache_io import load_torch_with_tmp_recovery, save_torch_atomic
from utils.log import WorkflowLogger, configure_logging, get_logger
from utils.pipeline import PipelineRunner, StageSpec
from utils.prompting import (
    build_chat_prompt_input,
    prompt_to_token_ids,
    truncate_token_ids,
)
from utils.reporting import write_index, write_manifest
from utils.contracts import GenerateManifest

log = get_logger(__name__)
SCHEMA_VERSION = "split-storage.v1"

GLOBAL_CLAIM_FORMAT_CONTRACT = (
    "STRICT OUTPUT CONTRACT:\n"
    "Objective: solve the task using minimal evidence-based reasoning.\n"
    "Output schema (exact):\n"
    "Reasoning:\n"
    "Step 1: <one atomic fact>\n"
    "Step 2: <one atomic fact>\n"
    "...\n"
    "Conclusion: <dataset-specific final answer>\n"
    "Rules:\n"
    "- Plain text only. No markdown, bullets, or code fences.\n"
    "- 1 to 8 Step lines; use only as many steps as the evidence requires.\n"
    "- The Conclusion line must contain only the final answer format requested by the dataset.\n"
    "- No text before 'Reasoning:' or after 'Conclusion:'.\n"
    "Format example only:\n"
    "Reasoning:\n"
    "Step 1: <one evidence statement>\n"
    "Conclusion: <final answer>"
)


def _inject_claim_contract(messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Apply global claim-format constraints to every dataset prompt."""
    safe_messages = []
    for msg in (messages or []):
        if isinstance(msg, dict):
            safe_messages.append({"role": msg.get("role", "user"), "content": str(msg.get("content", ""))})
    if not safe_messages:
        return [{"role": "system", "content": GLOBAL_CLAIM_FORMAT_CONTRACT}]

    for msg in safe_messages:
        if msg.get("role") == "system":
            msg["content"] = f'{msg.get("content", "").rstrip()}\n\n{GLOBAL_CLAIM_FORMAT_CONTRACT}'
            return safe_messages
    safe_messages.insert(0, {"role": "system", "content": GLOBAL_CLAIM_FORMAT_CONTRACT})
    return safe_messages


def _extract_sample_uid(raw_item: dict, split: str, sample_id: int) -> str:
    """Extract a stable sample UID from raw item fields, with deterministic fallback."""
    if isinstance(raw_item, dict):
        for key in ("qid", "id", "_id", "question_id", "uid", "guid"):
            value = raw_item.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
    return f"{split}:{sample_id}"


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    log.info("Random seed set to %d", seed)


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1: Generate + cache features")
    parser.add_argument(
        "--split", type=str, default="train",
        help="Split(s) to process. Comma-separated for multi-split: 'train,validation,test'",
    )
    parser.add_argument("--dataset", type=str, default=None,
                        help="Dataset name (e.g. 'hotpotqa', 'musique'). Default from config.")
    parser.add_argument("--backend", type=str, default=None,
                        choices=["vllm", "hf"],
                        help="Generation backend: 'vllm' (fast, hybrid) or 'hf' (pure HuggingFace)")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument(
        "--max_samples", type=str, default="0",
        help="Max samples per split. Single int (applied to all) or comma-sep per split: '100,50,50'",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Max new tokens for backbone generation outputs (default from config).",
    )
    parser.add_argument(
        "--prompt_max_tokens",
        type=int,
        default=int(os.environ.get("GEN_PROMPT_MAX_TOKENS", "0")),
        help="Max prompt tokens before generation (0 disables truncation).",
    )
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--n_hop_values", type=str, default=None,
                        help="Difficulty/n_hops filter. Comma-sep ints, or empty string for all.")
    parser.add_argument("--skip_existing", action="store_true", default=True)
    parser.add_argument("--no_skip_existing", dest="skip_existing", action="store_false")
    return parser.parse_args()


def _parse_max_samples(value: str, n_splits: int) -> list:
    """Parse --max_samples into a per-split list."""
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) == 1:
        return [int(parts[0])] * n_splits
    if len(parts) != n_splits:
        raise ValueError(
            f"--max_samples has {len(parts)} value(s) but {n_splits} split(s) were given"
        )
    return [int(p) for p in parts]


def _parse_field_list(value: str, n_items: int, default_value: str, field_name: str) -> list:
    """Parse comma-separated field values with broadcast support."""
    if value is None:
        return [default_value] * n_items
    parts = [p.strip() for p in str(value).split(",")]
    if len(parts) == 1:
        return [parts[0]] * n_items
    if len(parts) != n_items:
        raise ValueError(
            f"{field_name} has {len(parts)} value(s) but {n_items} split(s) were given"
        )
    return parts


def build_feature_extractor(cfg: Config, hidden_size: int, num_layers: int, num_heads: int, device: torch.device):
    """Build the CombinedExtractor from config.

    Returns:
        (feature_extractor, use_attention, attn_layer_indices)
    """
    hs_layers_str = str(cfg.features.hidden_state_layers).strip().lower()
    if hs_layers_str == "all":
        hs_layer_nums = None
    else:
        hs_layer_nums = [int(x.strip()) for x in hs_layers_str.split(",") if x.strip()]
    hs_weights_raw = str(cfg.features.hidden_state_weights or "").strip()
    hs_layer_weights = (
        [float(x.strip()) for x in hs_weights_raw.split(",") if x.strip()]
        if hs_weights_raw
        else None
    )

    extractors = [
        HiddenStateExtractor(
            layer_nums=hs_layer_nums,
            hidden_size=hidden_size,
            fusion=cfg.features.hidden_state_fusion,
            layer_weights=hs_layer_weights,
            num_hidden_layers=num_layers,
        ),
        TokenProbExtractor(
            top_n=cfg.features.top_n_probs,
            temperature=cfg.features.temperature,
            append_stats=cfg.features.token_append_stats,
        ),
    ]

    use_attention = bool(cfg.features.attention_layers)
    attn_layer_indices = None
    if use_attention:
        if str(cfg.features.attention_layers).strip().lower() == "all":
            attn_layer_nums = list(range(num_layers))
        else:
            attn_layer_nums = [int(x.strip()) for x in cfg.features.attention_layers.split(",")]
        attn_layer_indices = attn_layer_nums
        extractors.append(
            AttentionExtractor(
                layer_nums=attn_layer_nums,
                head_nums=cfg.features.attention_heads,
                attn_history_sz=cfg.features.attn_history_sz,
                pool=cfg.features.pool_attention_layers,
                num_layers=num_layers,
                num_heads=num_heads,
            )
        )

    feature_extractor = CombinedExtractor(extractors).to(device)
    return feature_extractor, use_attention, attn_layer_indices


def _release_cuda_cache(device: torch.device, reason: str) -> None:
    """Release reclaimable CUDA memory after persistent chunk writes."""
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    collected = gc.collect()
    torch.cuda.empty_cache()
    if hasattr(torch.cuda, "ipc_collect"):
        torch.cuda.ipc_collect()
    log.debug("  CUDA cache cleanup after %s (gc_collected=%d)", reason, collected)


def _normalize_claims_and_labels(
    claim_dicts,
    sample_label: int,
    n_hops: int | None = None,
    dataset_name: str | None = None,
    stable_key: str | None = None,
):
    """Normalize claims structure and assign provisional labels.

    Strict mode: accept only explicit reasoning+conclusion claims already
    extracted from the generated text. Missing/ambiguous conclusion structure is
    treated as extraction failure and the sample is dropped from claim-level
    supervision rather than repaired with fallback claims.
    """
    if not isinstance(claim_dicts, list):
        return [], [], False

    filtered = []
    for c in claim_dicts:
        if not isinstance(c, dict):
            continue
        ctype = str(c.get("claim_type", "")).strip().lower()
        if ctype not in {"reasoning", "conclusion"}:
            continue
        c = dict(c)
        c["claim_type"] = ctype
        c["claim_type_id"] = CLAIM_TYPE_IDS.get(ctype, 2)
        filtered.append(c)
    reasoning_claims = [c for c in filtered if c["claim_type"] == "reasoning"]
    conclusion_claims = [c for c in filtered if c["claim_type"] == "conclusion"]
    reasoning_claims = trim_reasoning_claims(
        reasoning_claims,
        n_hops=n_hops,
        dataset_name=dataset_name,
        stable_key=stable_key,
    )
    if not reasoning_claims or len(conclusion_claims) != 1:
        return [], [], False
    conclusion_claim = conclusion_claims[0]
    claim_dicts = reasoning_claims + [conclusion_claim]

    label = int(sample_label) if sample_label is not None else -1
    verified = []
    for c in claim_dicts:
        if c["claim_type"] == "conclusion":
            verified.append(label if label in (0, 1) else -1)
        else:
            verified.append(-1)

    assert len(verified) == len(claim_dicts)
    return claim_dicts, verified, False


def _split_claim_storage(claim_dicts, verified):
    """Split claim storage into reasoning/conclusion buckets with aligned labels."""
    reasoning_claims = []
    reasoning_verified = []
    conclusion_claim = None
    conclusion_verified = -1

    for ci, claim in enumerate(claim_dicts):
        v = int(verified[ci]) if ci < len(verified) else -1
        ctype = str(claim.get("claim_type", "unknown")).strip().lower() if isinstance(claim, dict) else "unknown"
        if ctype == "conclusion" and conclusion_claim is None:
            conclusion_claim = claim
            conclusion_verified = v
        elif ctype == "reasoning":
            reasoning_claims.append(claim)
            reasoning_verified.append(v)

    return {
        "reasoning_claims": reasoning_claims,
        "reasoning_verified": reasoning_verified,
        "conclusion_claim": conclusion_claim,
        "conclusion_verified": conclusion_verified,
    }


def _combined_claims_and_verified_from_split(sample):
    """Return claim list in reasoning...+conclusion order from split storage."""
    claims = []
    verified = []
    reasoning_claims = sample.get("reasoning_claims", []) or []
    reasoning_verified = sample.get("reasoning_verified", []) or []
    for ci, claim in enumerate(reasoning_claims):
        if not isinstance(claim, dict):
            continue
        claims.append(claim)
        try:
            verified.append(int(reasoning_verified[ci]) if ci < len(reasoning_verified) else -1)
        except Exception:
            verified.append(-1)
    conclusion_claim = sample.get("conclusion_claim")
    if isinstance(conclusion_claim, dict):
        claims.append(conclusion_claim)
        try:
            verified.append(int(sample.get("conclusion_verified", -1)))
        except Exception:
            verified.append(-1)
    return claims, verified


def _claims_have_valid_alignment(claim_dicts, gen_len: int) -> bool:
    """Strict alignment contract for per-claim token slices."""
    if not isinstance(claim_dicts, list) or int(gen_len) <= 0:
        return False
    n_reasoning = 0
    n_conclusion = 0
    for claim in claim_dicts:
        if not isinstance(claim, dict):
            return False
        ctype = str(claim.get("claim_type", "")).strip().lower()
        if ctype == "reasoning":
            n_reasoning += 1
        elif ctype == "conclusion":
            n_conclusion += 1
        else:
            return False

        aligned = claim.get("aligned_token_ids", []) or []
        if not isinstance(aligned, list) or not aligned:
            return False
        for tid in aligned:
            try:
                it = int(tid)
            except Exception:
                return False
            if it < 0 or it >= int(gen_len):
                return False
    return n_reasoning >= 1 and n_conclusion == 1


def generate_split(
    cfg: Config,
    split: str,
    args,
    engine,
    feature_extractor,
    use_attention,
    max_samples: int,
    dataset_name: str,
    dataset_path: str,
    cache_dir: str,
    attn_layer_indices=None,
) -> GenerateManifest:
    """Process a single split using a pre-built engine."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configured_batch_size = args.batch_size or cfg.generation.batch_size
    batch_size = configured_batch_size
    max_new_tokens = args.max_new_tokens or cfg.generation.max_new_tokens
    prompt_max_tokens = max(int(args.prompt_max_tokens or 0), 0)
    chunk_size = args.chunk_size or cfg.generation.chunk_size
    skip_existing = args.skip_existing and cfg.generation.skip_existing
    backend = args.backend or cfg.generation.backend
    model_path = args.model_path or cfg.model.pretrained_model_name_or_path
    feature_dim = feature_extractor.feature_dim()

    split_dir = Path(cache_dir) / split
    split_dir.mkdir(parents=True, exist_ok=True)

    # Parse difficulty filter
    n_hop_values = None
    if args.n_hop_values is not None:
        if args.n_hop_values.strip():
            n_hop_values = [int(x) for x in args.n_hop_values.split(",")]

    ds = get_dataset(
        name=dataset_name,
        dataset_path=dataset_path,
        split=split,
        n_hop_values=n_hop_values,
        max_samples=max_samples,
    )

    workflow = WorkflowLogger(log, "generate", width=cfg.output.log_banner_width)
    workflow.header(
        "ChainUQ Phase 1: Generate + Cache Features",
        dataset=dataset_name,
        split=split,
        backend=backend,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        chunk_size=chunk_size,
        prompt_max_tokens=prompt_max_tokens if prompt_max_tokens > 0 else "disabled",
        n_hop_values=n_hop_values or "all",
        cache_dir=cache_dir,
    )
    workflow.stage_start(
        "split",
        split=split,
        total_samples=len(ds),
        batch_size=batch_size,
        chunk_size=chunk_size,
        backend=backend,
    )

    if ds.needs_judge:
        log.info(
            "Dataset '%s' is free-form — skipping cheap auto-labeling; "
            "all sample labels start pending until judge.",
            dataset_name,
        )

    generation_start = time.time()
    current_chunk = []
    chunk_idx = 0
    total_correct = 0
    total_processed = 0
    total_pending = 0
    total_claim_pending = 0
    consistency_overrides = 0
    claim_regex_ok = 0
    claim_empty = 0
    dropped_claim_parse = 0
    dropped_claim_contract = 0
    prompt_truncated_samples = 0
    prompt_token_sum = 0
    prompt_token_max = 0
    counted_existing_chunks = set()
    total_samples = len(ds)
    last_progress_log = 0

    for start_idx in range(0, len(ds), batch_size):
        end_idx = min(start_idx + batch_size, len(ds))

        progress_pct = int((start_idx / total_samples) * 100)
        if progress_pct >= last_progress_log + 10:
            log.info("  Progress: %d%% (%d/%d samples)", progress_pct, start_idx, total_samples)
            last_progress_log = progress_pct

        # Check if this batch's chunk already exists
        chunk_for_start = start_idx // chunk_size
        chunk_path = split_dir / f"chunk_{chunk_for_start}.pt"
        if skip_existing and chunk_path.exists():
            if chunk_for_start in counted_existing_chunks:
                continue
            try:
                # Light validation: load only metadata without materializing
                # full feature tensors (which can use 50GB+ RAM per chunk).
                file_size = chunk_path.stat().st_size
                if file_size < 1024:
                    log.warning("  Chunk %d is suspiciously small (%d bytes), deleting", chunk_for_start, file_size)
                    chunk_path.unlink(missing_ok=True)
                else:
                    # Estimate sample count from expected chunk position
                    expected_samples = min(chunk_size, len(ds) - chunk_for_start * chunk_size)
                    counted_existing_chunks.add(chunk_for_start)
                    log.info("  Skipping chunk %d (exists, ~%d samples, %.1f MB)",
                             chunk_for_start, expected_samples, file_size / 1e6)
                    chunk_idx = max(chunk_idx, chunk_for_start + 1)
                    current_chunk = []
                    continue
            except Exception as e:
                log.warning("  Chunk %d corrupt (%s), deleting and regenerating", chunk_for_start, str(e))
                chunk_path.unlink(missing_ok=True)

        # Build prompts via dataset interface
        raw_items = [ds.data[i] for i in range(start_idx, min(end_idx, len(ds.data)))]
        prompts = []
        gt_strings = []
        batch_n_hops = []
        batch_questions = []

        for raw_item in raw_items:
            messages = _inject_claim_contract(ds.build_chat_messages(raw_item))
            prompt_input = build_chat_prompt_input(engine.tokenizer, messages)
            prompt_ids = prompt_to_token_ids(engine.tokenizer, prompt_input)
            original_len = len(prompt_ids)
            if prompt_max_tokens > 0:
                prompt_ids = truncate_token_ids(engine.tokenizer, prompt_ids, prompt_max_tokens)
            if len(prompt_ids) < original_len:
                prompt_truncated_samples += 1
            prompt_token_sum += len(prompt_ids)
            prompt_token_max = max(prompt_token_max, len(prompt_ids))
            prompts.append(prompt_ids)
            gt_strings.append(ds.get_ground_truth(raw_item))
            batch_n_hops.append(ds.get_difficulty(raw_item))
            batch_questions.append(raw_item.get("question", ""))

        # Generate + extract features via engine
        result = engine.generate_batch(
            prompts=prompts,
            feature_extractor=feature_extractor,
            max_new_tokens=max_new_tokens,
            use_attention=use_attention,
            attn_layer_indices=attn_layer_indices,
        )

        # Process each sample in batch
        per_sample_rows = []
        per_sample_claims = []
        per_sample_claim_errors = []
        for bi in range(len(prompts)):
            gen_text = result.generated_texts[bi]
            gen_token_ids = []
            gen_len = 0
            if isinstance(getattr(result, "generated_lengths", None), list) and bi < len(result.generated_lengths):
                try:
                    gen_len = max(0, int(result.generated_lengths[bi]))
                except Exception:
                    gen_len = 0
            if result.generated_token_ids is not None:
                gen_token_ids = result.generated_token_ids[bi].tolist()
                if gen_len > 0:
                    gen_token_ids = gen_token_ids[:gen_len]
                elif gen_token_ids:
                    gen_len = len(gen_token_ids)
            if gen_len <= 0 and result.features is not None:
                gen_len = int(result.features.shape[1])

            predicted_answer = ds.parse_answer(gen_text)
            gt_answer = gt_strings[bi]
            label = -1 if ds.needs_judge else ds.check_correctness(predicted_answer, gt_answer)
            answer_correct = (label == 1)

            extraction_method = "regex"
            extraction_error = ""
            try:
                sample_claim_key = make_claim_stable_key(
                    dataset_name=dataset_name,
                    question=batch_questions[bi],
                    n_hops=batch_n_hops[bi],
                    generated_text=gen_text,
                )
                claims = extract_claims_from_generation(
                    generated_text=gen_text,
                    generated_token_ids=gen_token_ids if gen_token_ids else None,
                    tokenizer=engine.tokenizer,
                    n_hops=batch_n_hops[bi],
                    dataset_name=dataset_name,
                    stable_key=sample_claim_key,
                )
                claim_regex_ok += 1
            except Exception as e:
                claims = []
                extraction_method = "regex_failed"
                extraction_error = str(e)
                claim_empty += 1

            claim_dicts = []
            for c in claims:
                if isinstance(c, dict):
                    cd = dict(c)
                    cd["extracted_by"] = extraction_method
                    claim_dicts.append(cd)
                else:
                    claim_dicts.append({
                        "text": str(c),
                        "claim_type": "reasoning",
                        "claim_type_id": CLAIM_TYPE_IDS.get("reasoning", 0),
                        "aligned_token_ids": [],
                        "extracted_by": extraction_method,
                    })
            per_sample_claim_errors.append(extraction_error)

            per_sample_rows.append(
                {
                    "sample_id": int(start_idx + bi),
                    "sample_uid": _extract_sample_uid(raw_items[bi], split=split, sample_id=int(start_idx + bi)),
                    "gen_text": gen_text,
                    "gen_len": gen_len,
                    "predicted_answer": predicted_answer,
                    "ground_truth": gt_answer,
                    "label": label,
                    "answer_correct": answer_correct,
                    "n_hops": batch_n_hops[bi],
                    "question": batch_questions[bi],
                    "features": (
                        result.features[bi].cpu()
                        if result.features is not None
                        else torch.zeros(1, feature_dim, dtype=torch.bfloat16)
                    ),
                    "token_probs": (
                        result.top_probs[bi].cpu()
                        if result.top_probs is not None
                        else torch.zeros(1, max(1, int(cfg.features.top_n_probs)), dtype=torch.bfloat16)
                    ),
                    "log_likelihoods": (
                        result.log_likelihoods[bi].cpu()
                        if result.log_likelihoods is not None
                        else torch.zeros(1, dtype=torch.bfloat16)
                    ),
                }
            )
            per_sample_claims.append(claim_dicts)

        # Normalize claims + assign provisional labels, then persist
        chunk_saved_in_batch = False
        for bi in range(len(prompts)):
            row = per_sample_rows[bi]
            claim_dicts = per_sample_claims[bi]

            claim_dicts, verified, overridden = _normalize_claims_and_labels(
                claim_dicts,
                sample_label=row["label"],
                n_hops=row.get("n_hops"),
                dataset_name=dataset_name,
                stable_key=make_claim_stable_key(
                    dataset_name=dataset_name,
                    question=row.get("question", ""),
                    n_hops=row.get("n_hops"),
                    generated_text=row.get("gen_text", ""),
                ),
            )
            if overridden:
                consistency_overrides += 1
            if not claim_dicts:
                err = (per_sample_claim_errors[bi] or "").strip()
                if err:
                    dropped_claim_parse += 1
                    # log.warning(
                    #     "Dropping sample due to claim parse failure "
                    #     "(split=%s idx=%d question=%r err=%s)",
                    #     split,
                    #     start_idx + bi,
                    #     row.get("question", "")[:120],
                    #     err[:200],
                    # )
                    log.warning(
                        "Dropping sample due to claim parse failure "
                        "(split=%s idx=%d)",
                        split,
                        start_idx + bi,
                    )
                else:
                    dropped_claim_contract += 1
                    log.warning(
                        "Dropping sample due to claim contract failure "
                        "(split=%s idx=%d question=%r)",
                        split,
                        start_idx + bi,
                        row.get("question", "")[:120],
                    )
                continue
            if not _claims_have_valid_alignment(claim_dicts, gen_len=row.get("gen_len", 0)):
                dropped_claim_contract += 1
                log.warning(
                    "Dropping sample due to invalid claim-token alignment "
                    "(split=%s idx=%d question=%r gen_len=%s)",
                    split,
                    start_idx + bi,
                    row.get("question", "")[:120],
                    row.get("gen_len", 0),
                )
                continue

            if row["label"] == 1:
                total_correct += 1
            elif row["label"] == -1:
                total_pending += 1
            total_claim_pending += sum(1 for v in verified if v == -1)

            attn_mask = torch.zeros(result.features.shape[1], dtype=torch.long) if result.features is not None else torch.ones(1, dtype=torch.long)
            if result.features is not None:
                eff_len = max(1, min(int(row["gen_len"]), int(result.features.shape[1])))
                attn_mask[:eff_len] = 1

            n_reasoning_claims = sum(
                1 for c in claim_dicts
                if isinstance(c, dict) and c.get("claim_type") == "reasoning"
            )
            split_claims = _split_claim_storage(claim_dicts, verified)
            sample_dict = {
                "sample_id": int(row["sample_id"]),
                "sample_uid": str(row["sample_uid"]),
                "features": row["features"],
                "attention_mask": attn_mask,
                "label": row["label"],
                "answer_correct": row["answer_correct"],
                "n_hops": row["n_hops"],
                "n_reasoning_claims": n_reasoning_claims,
                "predicted_answer": row["predicted_answer"],
                "ground_truth": row["ground_truth"],
                "question": row["question"],
                "generated_text": row["gen_text"],
                "dataset_name": dataset_name,
                "claim_extraction_failed": False,
                "claim_extraction_error": per_sample_claim_errors[bi],
                "claim_label_override": bool(overridden),
                "token_probs": row["token_probs"],
                "log_likelihoods": row["log_likelihoods"],
                "reasoning_claims": split_claims["reasoning_claims"],
                "reasoning_verified": split_claims["reasoning_verified"],
                "conclusion_claim": split_claims["conclusion_claim"],
                "conclusion_verified": split_claims["conclusion_verified"],
            }
            current_chunk.append(sample_dict)
            total_processed += 1

            if len(current_chunk) >= chunk_size:
                chunk_path = split_dir / f"chunk_{chunk_idx}.pt"
                save_torch_atomic(current_chunk, chunk_path)

                labeled = [s for s in current_chunk if s["label"] >= 0]
                incorrect_rate = sum(1 for s in labeled if int(s.get("label", 1)) == 0) / max(len(labeled), 1)
                pending_count = sum(1 for s in current_chunk if s["label"] == -1)

                claim_type_counts = {"reasoning": 0, "conclusion": 0}
                type_label_counter = defaultdict(Counter)
                total_claims = 0
                for s in current_chunk:
                    sc, sv = _combined_claims_and_verified_from_split(s)
                    total_claims += len(sc)
                    for ci, c in enumerate(sc):
                        ctype = c.get("claim_type", "reasoning") if isinstance(c, dict) else "reasoning"
                        claim_type_counts[ctype] = claim_type_counts.get(ctype, 0) + 1
                        label_str = "invalid"
                        if ci < len(sv):
                            try:
                                iv = int(sv[ci])
                                if iv in (-1, 0, 1):
                                    label_str = str(iv)
                            except Exception:
                                pass
                        type_label_counter[ctype][label_str] += 1
                avg_claims = total_claims / max(len(current_chunk), 1)
                log.info(
                    "  Chunk %d: %d samples, incorrect_rate=%.2f, pending=%d, avg_claims=%.1f, types=%s",
                    chunk_idx, len(current_chunk), incorrect_rate, pending_count, avg_claims,
                    {k: v for k, v in claim_type_counts.items() if v > 0},
                )

                current_chunk = []
                chunk_idx += 1
                chunk_saved_in_batch = True

        del result, raw_items, prompts, gt_strings, batch_n_hops, batch_questions
        if chunk_saved_in_batch:
            _release_cuda_cache(device, reason=f"chunk {chunk_idx - 1}")

    # Save remaining samples
    if current_chunk:
        chunk_path = split_dir / f"chunk_{chunk_idx}.pt"
        save_torch_atomic(current_chunk, chunk_path)

        labeled = [s for s in current_chunk if s["label"] >= 0]
        incorrect_rate = sum(1 for s in labeled if int(s.get("label", 1)) == 0) / max(len(labeled), 1)
        total_claims = sum(
            len(_combined_claims_and_verified_from_split(s)[0]) for s in current_chunk
        )
        avg_claims = total_claims / max(len(current_chunk), 1)
        log.info(
            "  Final chunk %d: %d samples, incorrect_rate=%.2f, avg_claims=%.1f",
            chunk_idx, len(current_chunk), incorrect_rate, avg_claims
        )
        chunk_idx += 1
        _release_cuda_cache(device, reason=f"final chunk {chunk_idx - 1}")

    # Write manifest
    hidden_size = engine.model_config.hidden_size
    max_new_tokens_val = args.max_new_tokens or cfg.generation.max_new_tokens
    total_labeled = total_processed - total_pending
    total_incorrect = total_labeled - total_correct
    correct_rate = total_correct / max(total_labeled, 1)
    incorrect_rate = total_incorrect / max(total_labeled, 1)

    chunk_file_sizes = {}
    for ci in range(chunk_idx):
        cp = split_dir / f"chunk_{ci}.pt"
        if cp.exists():
            chunk_file_sizes[cp.name] = cp.stat().st_size

    # chunk_sample_counts is built incrementally while constructing the index
    # (keyed by chunk name, converted to ordered list after the loop).
    chunk_sample_counts_map: dict = defaultdict(int)

    # Build pre-computed index for fast dataset loading
    index = []
    sample_correct = 0
    sample_incorrect = 0
    sample_pending = 0
    total_claims_count = 0
    claims_correct = 0
    claims_incorrect = 0
    claims_pending = 0

    # Recompute claim extraction stats from stored data (handles resume case)
    scan_regex_ok = 0
    scan_llm_ok = 0
    scan_empty = 0

    # Per claim-type stats
    type_stats = defaultdict(lambda: {"count": 0, "correct": 0, "incorrect": 0, "pending": 0, "char_total": 0})
    # Per n_hops stats
    nhop_stats = defaultdict(lambda: {
        "samples": 0, "sample_correct": 0,
        "claims": 0, "claims_correct": 0, "claims_incorrect": 0, "claims_pending": 0,
        "char_total": 0,
        "type_stats": defaultdict(lambda: {"count": 0, "correct": 0, "incorrect": 0, "char_total": 0}),
    })

    for ci in range(chunk_idx):
        cp = split_dir / f"chunk_{ci}.pt"
        if not cp.exists():
            continue
        chunk_data = load_torch_with_tmp_recovery(cp, logger=log, delete_corrupt=False)
        if chunk_data is None:
            continue
        for local_idx, sample in enumerate(chunk_data):
            claims, verified = _combined_claims_and_verified_from_split(sample)
            n_hops = sample.get("n_hops", 0)
            sample_label = sample.get("label", -1)
            if sample_label == 1:
                sample_correct += 1
            elif sample_label == 0:
                sample_incorrect += 1
            else:
                sample_pending += 1

            n_reasoning = 0
            n_conclusion = 0
            n_reasoning_correct = 0
            n_conclusion_correct = 0
            n_claims = len(verified)
            n_pending = 0
            n_correct = 0
            n_incorrect = 0

            # Count extraction methods from stored claims
            if claims:
                methods = set()
                for c in claims:
                    if isinstance(c, dict):
                        methods.add(c.get("extracted_by", ""))
                if "llm" in methods:
                    scan_llm_ok += 1
                elif "regex" in methods:
                    scan_regex_ok += 1
                else:
                    scan_empty += 1
            else:
                scan_empty += 1

            nh = nhop_stats[n_hops]
            nh["samples"] += 1
            if sample_label == 1:
                nh["sample_correct"] += 1

            for c_idx, v in enumerate(verified):
                iv = int(v)
                if iv == 1:
                    n_correct += 1
                elif iv == 0:
                    n_incorrect += 1
                else:
                    n_pending += 1

                # Claim type and text
                ctype = "unknown"
                char_len = 0
                if c_idx < len(claims) and isinstance(claims[c_idx], dict):
                    ctype = claims[c_idx].get("claim_type", "unknown") or "unknown"
                    text = claims[c_idx].get("text", "") or ""
                    char_len = len(text)

                if ctype == "reasoning":
                    n_reasoning += 1
                    if iv == 1:
                        n_reasoning_correct += 1
                elif ctype == "conclusion":
                    n_conclusion += 1
                    if iv == 1:
                        n_conclusion_correct += 1

                ts = type_stats[ctype]
                ts["count"] += 1
                ts["char_total"] += char_len
                if iv == 1:
                    ts["correct"] += 1
                elif iv == 0:
                    ts["incorrect"] += 1
                else:
                    ts["pending"] += 1

                # Per n_hops type stats
                nht = nh["type_stats"][ctype]
                nht["count"] += 1
                nht["char_total"] += char_len
                if iv == 1:
                    nht["correct"] += 1
                elif iv == 0:
                    nht["incorrect"] += 1

                nh["claims"] += 1
                nh["char_total"] += char_len
                if iv == 1:
                    nh["claims_correct"] += 1
                elif iv == 0:
                    nh["claims_incorrect"] += 1
                else:
                    nh["claims_pending"] += 1

            total_claims_count += n_claims
            claims_correct += n_correct
            claims_incorrect += n_incorrect
            claims_pending += n_pending
            index.append({
                "chunk_file": f"chunk_{ci}.pt",
                "local_idx": local_idx,
                "sample_id": int(sample.get("sample_id", -1)),
                "sample_uid": str(sample.get("sample_uid", "")),
                "label": sample_label,
                "n_hops": n_hops,
                "n_reasoning_claims": n_reasoning,
                "n_conclusion_claims": n_conclusion,
                "n_reasoning_correct": n_reasoning_correct,
                "n_conclusion_correct": n_conclusion_correct,
                "claims_correct": n_correct,
                "claims_incorrect": n_incorrect,
                "dataset_name": sample.get("dataset_name", ""),
                "claims_total": n_claims,
                "claims_pending": n_pending,
                "claim_label_override": bool(sample.get("claim_label_override", False)),
            })
            chunk_sample_counts_map[f"chunk_{ci}.pt"] += 1
        del chunk_data
        gc.collect()

    # Convert map to ordered list matching chunk_0 … chunk_N order.
    chunk_sample_counts = [
        chunk_sample_counts_map.get(f"chunk_{ci}.pt", 0)
        for ci in range(chunk_idx)
    ]

    override_samples = sum(1 for e in index if e.get("claim_label_override", False))
    no_fully_verified_claim_samples = sum(
        1
        for e in index
        if int(e.get("claims_total", 0) or 0) <= 0
        or int(e.get("claims_pending", 0) or 0) > 0
    )

    # Build claim_type_stats summary
    claim_type_stats = {}
    for ctype, ts in type_stats.items():
        labeled = ts["correct"] + ts["incorrect"]
        claim_type_stats[ctype] = {
            "count": ts["count"],
            "avg_char_length": round(ts["char_total"] / max(ts["count"], 1), 1),
            "correct": ts["correct"],
            "incorrect": ts["incorrect"],
            "pending": ts["pending"],
            "correct_rate": round(ts["correct"] / max(labeled, 1), 4) if labeled > 0 else None,
        }

    # Overall avg claim char length
    total_char = sum(ts["char_total"] for ts in type_stats.values())
    avg_claim_char_length = round(total_char / max(total_claims_count, 1), 1)

    # Build per_nhop_stats summary
    per_nhop_stats = {}
    for nh_val in sorted(nhop_stats.keys()):
        nh = nhop_stats[nh_val]
        labeled_claims = nh["claims_correct"] + nh["claims_incorrect"]
        nh_type = {}
        for ctype, nht in nh["type_stats"].items():
            nht_labeled = nht["correct"] + nht.get("incorrect", 0)
            nh_type[ctype] = {
                "count": nht["count"],
                "avg_char_length": round(nht["char_total"] / max(nht["count"], 1), 1),
                "correct_rate": round(nht["correct"] / max(nht_labeled, 1), 4) if nht_labeled > 0 else None,
            }
        per_nhop_stats[str(nh_val)] = {
            "samples": nh["samples"],
            "sample_correct_rate": round(nh["sample_correct"] / max(nh["samples"], 1), 4),
            "claims": nh["claims"],
            "avg_claim_char_length": round(nh["char_total"] / max(nh["claims"], 1), 1),
            "claims_correct_rate": round(nh["claims_correct"] / max(labeled_claims, 1), 4) if labeled_claims > 0 else None,
            "claim_types": nh_type,
        }

    generation_duration = time.time() - generation_start
    has_claims = bool(scan_regex_ok)
    total_processed = len(index)
    total_correct = sample_correct
    total_incorrect = sample_incorrect
    total_pending = sample_pending
    total_labeled = total_correct + total_incorrect
    correct_rate = total_correct / max(total_labeled, 1)
    incorrect_rate = total_incorrect / max(total_labeled, 1)

    manifest: GenerateManifest = {
        "schema_version": SCHEMA_VERSION,
        "split": split,
        "dataset_name": dataset_name,
        "granularity": "claim",
        "total_samples": total_processed,
        "total_claims": total_claims_count,
        "chunk_size": chunk_size,
        "num_chunks": chunk_idx,
        "chunk_sample_counts": chunk_sample_counts,
        "chunk_file_sizes": chunk_file_sizes,
        "feature_dim": feature_dim,
        "model_path": model_path,
        "claim_extractor_model": "",
        "claim_extractor_applied": has_claims,
        "hidden_size": hidden_size,
        "max_new_tokens": max_new_tokens_val,
        "prompt_max_tokens": prompt_max_tokens,
        "prompt_truncated_samples": int(prompt_truncated_samples),
        "avg_prompt_tokens": round(prompt_token_sum / max(total_processed, 1), 2),
        "max_prompt_tokens": int(prompt_token_max),
        "backend": backend,
        "sample_correct": total_correct,
        "sample_incorrect": total_incorrect,
        "sample_pending": total_pending,
        "correct_rate": correct_rate,
        "incorrect_rate": incorrect_rate,
        "claims_correct": claims_correct,
        "claims_incorrect": claims_incorrect,
        "claims_pending": claims_pending,
        "avg_claim_char_length": avg_claim_char_length,
        "total_claim_pending": total_claim_pending,
        "claim_type_stats": claim_type_stats,
        "per_nhop_stats": per_nhop_stats,
        "claim_label_consistency_overrides": int(consistency_overrides),
        "claim_label_override_samples": int(override_samples),
        "dropped_samples_generation": int(dropped_claim_parse + dropped_claim_contract),
        "drop_reason_counts": {
            "claim_parse_failed": int(dropped_claim_parse),
            "claim_contract_failed": int(dropped_claim_contract),
        },
        "no_fully_verified_claim_samples": int(no_fully_verified_claim_samples),
        "no_usable_verified_claim_samples": int(no_fully_verified_claim_samples),
        "claim_extraction_stats": {
            "regex_ok": scan_regex_ok,
            "llm_ok": scan_llm_ok,
            "empty": scan_empty,
        },
        "phase_status": {
            "generation": {
                "status": "complete",
                "duration_seconds": round(generation_duration, 1),
            },
        },
    }
    manifest_path = split_dir / "manifest.json"
    write_manifest(manifest_path, manifest)

    # Write index to separate file (keeps manifest small)
    index_path = split_dir / "index.json"
    write_index(index_path, index)

    workflow.stage_end(
        "split",
        split=split,
        samples=total_processed,
        sample_correct=total_correct,
        sample_incorrect=total_incorrect,
        sample_pending=total_pending,
        claims=total_claims_count,
        claim_correct=claims_correct,
        claim_incorrect=claims_incorrect,
        claim_pending=claims_pending,
        consistency_overrides=consistency_overrides,
        regex_claim_matches=scan_regex_ok,
        empty_claim_outputs=scan_empty,
    )
    workflow.artifact("manifest.saved", manifest_path, split=split)
    workflow.artifact("index.saved", index_path, split=split)

    if total_pending > 0:
        workflow.warning(
            "split.pending_samples",
            split=split,
            pending_samples=total_pending,
            next_step="run scripts/judge.py before training",
        )
    if total_claim_pending > 0:
        workflow.event(
            "split.pending_claims",
            split=split,
            pending_claims=total_claim_pending,
        )

    return manifest


def main():
    args = parse_args()
    cfg = Config()
    configure_logging(cfg, force=True)
    workflow = WorkflowLogger(log, "generate", width=cfg.output.log_banner_width)
    runner = PipelineRunner(workflow)

    set_seed(GLOBAL_SEED)

    splits = [s.strip() for s in args.split.split(",") if s.strip()]
    max_samples_list = _parse_max_samples(args.max_samples, len(splits))
    backend = args.backend or cfg.generation.backend
    model_path = args.model_path or cfg.model.pretrained_model_name_or_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_names = _parse_field_list(
        args.dataset,
        len(splits),
        cfg.dataset.dataset_name,
        "--dataset",
    )
    dataset_paths = _parse_field_list(
        args.dataset_path,
        len(splits),
        cfg.dataset.dataset_path,
        "--dataset_path",
    )
    cache_dirs = _parse_field_list(
        args.cache_dir,
        len(splits),
        cfg.generation.cache_dir,
        "--cache_dir",
    )

    def _init_engine_and_extractor():
        initialized_engine = get_engine(backend=backend, cfg=cfg, model_path=model_path, device=device)
        extractor, enabled_attention, layer_indices = build_feature_extractor(
            cfg,
            initialized_engine.model_config.hidden_size,
            initialized_engine.model_config.num_hidden_layers,
            initialized_engine.model_config.num_attention_heads,
            device,
        )
        return initialized_engine, extractor, enabled_attention, layer_indices

    runner.register_stage(
        StageSpec(
            name="engine_init",
            fn=lambda payload: _init_engine_and_extractor(),
            start_fields=lambda payload: {
                "backend": backend,
                "model_path": model_path,
                "device": device.type,
            },
            result_fields=lambda result: {
                "feature_dim": result[1].feature_dim(),
                "use_attention": result[2],
                "attention_layers": result[3],
            },
        )
    )

    runner.register_stage(
        StageSpec(
            name="task",
            fn=lambda payload: generate_split(
                cfg=cfg,
                split=payload["split"],
                args=args,
                engine=payload["engine"],
                feature_extractor=payload["feature_extractor"],
                use_attention=payload["use_attention"],
                max_samples=payload["max_samples"],
                dataset_name=payload["dataset_name"],
                dataset_path=payload["dataset_path"],
                cache_dir=payload["cache_dir"],
                attn_layer_indices=payload["attn_layer_indices"],
            ),
            start_fields=lambda payload: {
                "task_index": payload["task_index"],
                "total_tasks": payload["total_tasks"],
                "dataset": payload["dataset_name"],
                "split": payload["split"],
                "cache_dir": payload["cache_dir"],
                "max_samples": payload["max_samples"],
            },
            result_fields=lambda manifest: {
                "split": manifest.get("split"),
                "samples": int(manifest.get("total_samples", 0)),
                "chunks": int(manifest.get("num_chunks", 0)),
                "sample_pending": int(manifest.get("sample_pending", 0)),
                "dropped_generation": int(manifest.get("dropped_samples_generation", 0)),
                "claims_pending": int(manifest.get("claims_pending", 0)),
            },
        )
    )

    runner.register_stage(
        StageSpec(
            name="engine_unload",
            fn=lambda payload: payload.unload(),
            start_fields=lambda payload: {"backend": backend},
            include_duration=False,
        )
    )

    engine, feature_extractor, use_attention, attn_layer_indices = runner.run("engine_init")

    n_tasks = len(splits)
    for i, (split, max_samples, dataset_name, dataset_path, cache_dir) in enumerate(
        zip(splits, max_samples_list, dataset_names, dataset_paths, cache_dirs),
        start=1,
    ):
        runner.run(
            "task",
            {
                "task_index": i,
                "total_tasks": n_tasks,
                "split": split,
                "max_samples": max_samples,
                "dataset_name": dataset_name,
                "dataset_path": dataset_path,
                "cache_dir": cache_dir,
                "engine": engine,
                "feature_extractor": feature_extractor,
                "use_attention": use_attention,
                "attn_layer_indices": attn_layer_indices,
            },
        )

    runner.run("engine_unload", engine)


if __name__ == "__main__":
    main()
