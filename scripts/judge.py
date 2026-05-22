#!/usr/bin/env python3
"""
judge.py - Phase 1.5: LLM-as-Judge for free-form answer evaluation.

For datasets where check_correctness() returns -1, this script uses a judge LLM
to evaluate whether the generated answer is correct and verify individual claims.

Two-stage chain judge:
  Stage 1: Answer correctness + conclusion verification
  Stage 2: Reasoning chain consistency (per-claim verification)

Usage:
    python scripts/judge.py --cache_dir /path/to/cached_features --split test \
        --judge_model /path/to/judge_model --judge_backend vllm

    python scripts/judge.py --cache_dir /path/to/cached_features --split test \
        --judge_model /path/to/judge_model --force
"""

import sys
import gc
import time
import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from config import Config
from data.claims import CLAIM_TYPE_IDS, make_claim_stable_key, trim_reasoning_claims
from engine import get_engine
from scripts.chain_judge import (
    parse_judge_response_strict,
)
from utils.cache_io import load_torch_with_tmp_recovery, read_json, save_torch_atomic
from utils.log import WorkflowLogger, configure_logging, get_logger
from utils.pipeline import PipelineRunner, StageSpec
from utils.prompting import build_chat_prompt_input, prompt_to_token_ids, truncate_token_ids
from utils.reporting import write_index, write_manifest
from utils.contracts import JudgeRunSummary

log = get_logger(__name__)
SCHEMA_VERSION = "split-storage.v1"

STEPGAME_ALLOWED_LABELS = [
    "upper-right", "lower-right", "upper-left", "lower-left",
    "above", "left", "below", "right", "overlap",
]


def _sample_claims_and_verified(sample):
    """Read canonical split-storage claims in reasoning...+conclusion order."""
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


def _write_sample_verified(sample, claims, verified):
    """Persist verified labels back into split-storage fields."""
    if not isinstance(claims, list) or not isinstance(verified, list):
        sample["reasoning_verified"] = []
        sample["conclusion_verified"] = -1
        return sample
    reasoning_verified = []
    conclusion_verified = -1
    for ci, claim in enumerate(claims):
        ctype = str(claim.get("claim_type", "")).strip().lower() if isinstance(claim, dict) else ""
        try:
            v = int(verified[ci]) if ci < len(verified) else -1
        except Exception:
            v = -1
        if ctype == "conclusion":
            conclusion_verified = v
        elif ctype == "reasoning":
            reasoning_verified.append(v)
    sample["reasoning_verified"] = reasoning_verified
    sample["conclusion_verified"] = conclusion_verified
    return sample


def _compute_claim_type_label_ratio(samples):
    """Compute per-claim-type label ratios from claim verified labels."""
    type_label_counter = defaultdict(Counter)
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        claims, verified = _sample_claims_and_verified(sample)
        if not isinstance(claims, list):
            continue
        if not isinstance(verified, list):
            verified = []
        for ci, claim in enumerate(claims):
            ctype = "unknown"
            if isinstance(claim, dict):
                ctype = str(claim.get("claim_type", "unknown")).strip().lower() or "unknown"
            label = "invalid"
            if ci < len(verified):
                try:
                    iv = int(verified[ci])
                    if iv in (-1, 0, 1):
                        label = str(iv)
                except Exception:
                    pass
            type_label_counter[ctype][label] += 1

    ratio = {}
    for ctype, counter in type_label_counter.items():
        denom = float(sum(counter.values()))
        if denom <= 0:
            continue
        ordered = {}
        for key in ("-1", "0", "1", "invalid"):
            if key in counter:
                ordered[key] = round(float(counter[key]) / denom, 6)
        for key, count in counter.items():
            if key not in ordered:
                ordered[key] = round(float(count) / denom, 6)
        ratio[ctype] = ordered
    return ratio


def _get_dataset_label_space(dataset_name: str):
    name = (dataset_name or "").strip().lower()
    if name == "stepgame":
        return STEPGAME_ALLOWED_LABELS
    if name == "spartqa":
        # SpartQA ground truth is stored as 0-indexed option id.
        return ["0", "1", "2", "3"]
    return None


def _build_stage1_prompt(
    tokenizer,
    ground_truth,
    conclusion_text,
    allowed_labels=None,
    include_analysis: bool = False,
    prompt_max_tokens: int = 0,
):
    """Build stage-1 prompt using only conclusion claim + ground truth."""
    conclusion = (conclusion_text or "").strip() or "unknown"
    prompt = (
        "You are an exact answer verifier.\n"
        "Objective: decide whether the model's conclusion matches the ground-truth answer.\n"
        "Input fields:\n"
        f"Ground Truth Answer: {ground_truth}\n"
        f"Conclusion Claim: {conclusion}\n"
        "Output schema (strict JSON): {\"correct\": true/false}\n"
    )
    if allowed_labels:
        labels = ", ".join(str(x) for x in allowed_labels)
        prompt += (
            f"Allowed labels: [{labels}]\n"
            "Normalize the conclusion to one allowed label if possible.\n"
            "Do not treat partial direction matches as correct.\n"
        )
    prompt += (
        "Few-shot example:\n"
        "Input:\n"
        "Ground Truth Answer: Paris\n"
        "Conclusion Claim: The answer is Paris.\n"
        "Output:\n"
        "{\"correct\": true}\n"
        "Strict rules:\n"
        "1. Output exactly one JSON object on one line.\n"
        "2. No markdown, code fences, prefixes, or suffixes.\n"
    )
    if include_analysis:
        prompt += (
            '3. Use schema: {"analysis": "...", "correct": true/false}.\n'
            "4. Keep analysis concise and evidence-based.\n"
            '5. Start with "{" and end immediately after "}".\n'
        )
    else:
        prompt += (
            '3. Use schema: {"correct": true/false}.\n'
            '4. Start with "{" and end immediately after "}".\n'
        )
    messages = [{"role": "user", "content": prompt}]
    prompt_input = build_chat_prompt_input(tokenizer, messages)
    prompt_ids = prompt_to_token_ids(tokenizer, prompt_input)
    original_len = len(prompt_ids)
    if prompt_max_tokens > 0:
        prompt_ids = truncate_token_ids(tokenizer, prompt_ids, prompt_max_tokens)
    return prompt_ids, original_len, len(prompt_ids)


def _build_stage1_schema(include_analysis: bool):
    properties = {
        "correct": {"type": "boolean"},
    }
    required = ["correct"]
    if include_analysis:
        properties["analysis"] = {"type": "string"}
        required = ["analysis", "correct"]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _coerce_json_bool(value):
    """Normalize judge ``correct`` field to bool; unsupported types -> None."""
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        s = value.strip().lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
    return None


def _parse_stage1_output(response):
    """Parse stage-1 judge output into a single sample/conclusion label."""
    parsed = parse_judge_response_strict(response)
    if not isinstance(parsed, dict):
        return None, "failed"
    if "correct" not in parsed:
        return None, "failed"
    correct = _coerce_json_bool(parsed["correct"])
    if correct is None:
        return None, "failed"
    return (1 if correct else 0), "exact"


def _build_stage2_prompt(
    tokenizer,
    question,
    generated_text,
    reasoning_claims,
    conclusion_text,
    include_analysis: bool = False,
    prompt_max_tokens: int = 0,
):
    """Build stage-2 judge prompt: reasoning chain consistency.

    Uses a concise prompt requesting ONLY a short JSON array to stay within
    token limits. Ground truth is intentionally excluded to avoid leakage.
    """
    n = len(reasoning_claims)
    steps = "\n".join(f"Step {i+1}: {c}" for i, c in enumerate(reasoning_claims))
    final = conclusion_text or generated_text[:200]
    prompt = (
        "You are an exact reasoning-step verifier.\n"
        "Objective: evaluate each reasoning step for logical consistency toward the conclusion.\n"
        "Input fields:\n"
        f"Question: {question}\n"
        f"Reasoning:\n{steps}\n"
        f"Conclusion: {final}\n\n"
        "Output schema (strict JSON): {\"reasoning_verified\": [0,1,...]}\n"
        "Few-shot example:\n"
        "Input:\n"
        "Question: Is water wet?\n"
        "Reasoning:\nStep 1: Water is a liquid.\nStep 2: Liquids can make surfaces wet.\n"
        "Conclusion: yes\n"
        "Output:\n"
        "{\"reasoning_verified\": [1, 1]}\n"
        "Strict rules:\n"
        "1. Output exactly one JSON object on one line.\n"
        "2. No markdown, code fences, prefixes, or suffixes.\n"
        f"3. Array length must be exactly {n}.\n"
        "4. Array values must be only 0 or 1.\n"
    )
    if include_analysis:
        prompt += (
            '5. Use schema: {"analysis": "...", "reasoning_verified": [0, 1, ...]}.\n'
            "6. Keep analysis concise and evidence-based.\n"
            '7. Start with "{" and end immediately after "}".\n'
        )
    else:
        prompt += (
            '5. Use schema: {"reasoning_verified": [0, 1, ...]}.\n'
            '6. Start with "{" and end immediately after "}".\n'
        )
    messages = [{"role": "user", "content": prompt}]
    prompt_input = build_chat_prompt_input(tokenizer, messages)
    prompt_ids = prompt_to_token_ids(tokenizer, prompt_input)
    original_len = len(prompt_ids)
    if prompt_max_tokens > 0:
        prompt_ids = truncate_token_ids(tokenizer, prompt_ids, prompt_max_tokens)
    return prompt_ids, original_len, len(prompt_ids)


def _build_stage2_schema(expected_len: int, include_analysis: bool):
    properties = {
        "reasoning_verified": {
            "type": "array",
            "items": {"type": "integer", "enum": [0, 1]},
            "minItems": int(expected_len),
            "maxItems": int(expected_len),
        },
    }
    required = ["reasoning_verified"]
    if include_analysis:
        properties["analysis"] = {"type": "string"}
        required = ["analysis", "reasoning_verified"]
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _parse_stage2_output(response, expected_len):
    """Parse stage-2 judge output: per-reasoning-claim verification.

    Accept only exact JSON responses with the requested array length.
    """
    parsed = parse_judge_response_strict(response)
    if not isinstance(parsed, dict):
        return [-1] * expected_len, "failed"
    verified = parsed.get("reasoning_verified", None)
    if not isinstance(verified, list) or len(verified) != expected_len:
        return [-1] * expected_len, "failed"

    result = []
    for v in verified:
        if isinstance(v, bool):
            result.append(1 if v else 0)
        elif isinstance(v, (int, float)) and int(v) in (0, 1):
            result.append(int(v))
        else:
            return [-1] * expected_len, "failed"
    return result, "exact"


def parse_args():
    parser = argparse.ArgumentParser(description="Phase 1.5: LLM-as-Judge")
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="Cache directory containing chunk files (comma-separated supported)")
    parser.add_argument("--split", type=str, default="test",
                        help="Split to judge (default: test)")
    parser.add_argument("--judge_model", type=str, default=None,
                        help="Path to judge model (overrides config)")
    parser.add_argument("--judge_backend", type=str, default=None,
                        choices=["vllm", "hf"],
                        help="Judge backend (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for judge generation (overrides config)")
    parser.add_argument("--max_new_tokens", type=int, default=None,
                        help="Max new tokens for judge (overrides config)")
    parser.add_argument(
        "--prompt_max_tokens",
        type=int,
        default=int(os.environ.get("JUDGE_PROMPT_MAX_TOKENS", "0")),
        help="Max prompt tokens for stage prompts (0 disables truncation).",
    )
    parser.add_argument("--force", action="store_true",
                        help="Re-judge all samples, not just label=-1")
    return parser.parse_args()


def _parse_csv_items(raw: str):
    return [item.strip() for item in raw.split(",") if item.strip()]


def _discover_targets(cache_dirs, splits):
    targets = []
    for cache_dir in cache_dirs:
        for split in splits:
            split_dir = Path(cache_dir) / split
            if not split_dir.exists():
                log.info("Split directory not found: %s — skipping.", split_dir)
                continue
            if not any(split_dir.glob("chunk_*.pt")):
                log.info("No chunk files in %s — skipping.", split_dir)
                continue
            targets.append((cache_dir, split, split_dir))
    return targets


def _sample_has_pending_judgment(sample) -> bool:
    if bool(sample.get("_drop_sample", False)):
        return True
    claims, _ = _sample_claims_and_verified(sample)
    if not claims:
        return True
    if sample.get("label", -1) == -1:
        return True
    _, verified = _sample_claims_and_verified(sample)
    if isinstance(verified, list):
        for value in verified:
            try:
                if int(value) == -1:
                    return True
            except Exception:
                continue
    return False


def _enforce_sample_claim_contract(sample):
    claims, old_verified = _sample_claims_and_verified(sample)
    normalized = []
    for c in claims:
        if not isinstance(c, dict):
            continue
        text = str(c.get("text", "")).strip()
        if not text:
            continue
        ctype = str(c.get("claim_type", "")).strip().lower()
        if ctype not in ("reasoning", "conclusion"):
            ctype = "reasoning"
        row = dict(c)
        row["text"] = text
        row["claim_type"] = ctype
        row["claim_type_id"] = CLAIM_TYPE_IDS.get(ctype, CLAIM_TYPE_IDS["reasoning"])
        normalized.append(row)

    reasoning = [c for c in normalized if c["claim_type"] == "reasoning"]
    reasoning = trim_reasoning_claims(
        reasoning,
        n_hops=sample.get("n_hops"),
        dataset_name=sample.get("dataset_name"),
        stable_key=make_claim_stable_key(
            dataset_name=sample.get("dataset_name"),
            question=sample.get("question", ""),
            n_hops=sample.get("n_hops"),
            generated_text=sample.get("generated_text", ""),
        ),
    )
    conclusion_candidates = [c for c in normalized if c["claim_type"] == "conclusion"]
    if not reasoning or len(conclusion_candidates) != 1:
        sample["reasoning_claims"] = []
        sample["reasoning_verified"] = []
        sample["conclusion_claim"] = None
        sample["conclusion_verified"] = -1
        return sample
    if any(not c.get("aligned_token_ids") for c in (reasoning + conclusion_candidates)):
        sample["reasoning_claims"] = []
        sample["reasoning_verified"] = []
        sample["conclusion_claim"] = None
        sample["conclusion_verified"] = -1
        return sample

    new_claims = reasoning + [conclusion_candidates[0]]
    if isinstance(old_verified, list) and len(old_verified) == len(new_claims):
        new_verified = old_verified
    else:
        new_verified = [-1] * len(new_claims)
    sample["reasoning_claims"] = [c for c in new_claims if c.get("claim_type") == "reasoning"]
    sample["conclusion_claim"] = next((c for c in new_claims if c.get("claim_type") == "conclusion"), None)
    _write_sample_verified(sample, new_claims, new_verified)
    return sample


def _mark_sample_dropped(sample, reason: str):
    """Flag sample for removal at chunk flush time."""
    sample["_drop_sample"] = True
    sample["drop_reason"] = str(reason)
    return sample


def _target_has_pending_judgment(split_dir, args) -> bool:
    if args.force:
        return True

    for chunk_path in sorted(split_dir.glob("chunk_*.pt")):
        chunk_data = load_torch_with_tmp_recovery(chunk_path, logger=None, delete_corrupt=False)
        if chunk_data is None:
            return True
        try:
            if any(_sample_has_pending_judgment(sample) for sample in chunk_data):
                return True
        finally:
            del chunk_data
    return False


def process_split(cache_dir, split, split_dir, args, cfg, engine, judge_model,
                  judge_batch_size, judge_max_new_tokens):
    """Run judge on a single split. Memory-efficient: processes one chunk at a time."""
    chunk_files = sorted(split_dir.glob("chunk_*.pt"))
    if not chunk_files:
        log.info("No chunk files found in %s — skipping.", split_dir)
        return

    log.info("Scanning %d chunks for pending samples...", len(chunk_files))
    pending_by_chunk = {}
    total_samples = 0

    for chunk_idx, chunk_path in enumerate(chunk_files):
        chunk_data = load_torch_with_tmp_recovery(chunk_path, logger=log)
        if chunk_data is None:
            continue
        total_samples += len(chunk_data)
        pending_indices = []
        for local_idx, sample in enumerate(chunk_data):
            if args.force or _sample_has_pending_judgment(sample):
                pending_indices.append(local_idx)
        if pending_indices:
            pending_by_chunk[chunk_idx] = pending_indices
        del chunk_data

    total_pending = sum(len(v) for v in pending_by_chunk.values())
    log.info("Split '%s': %d total samples, %d need judgment across %d chunks.",
             split, total_samples, total_pending, len(pending_by_chunk))

    return _process_pending(
        cache_dir, split, split_dir, args, cfg, engine, judge_model,
        judge_batch_size, judge_max_new_tokens,
        chunk_files, pending_by_chunk, total_pending,
    )


def _process_pending(cache_dir, split, split_dir, args, cfg, engine, judge_model,
                     judge_batch_size, judge_max_new_tokens,
                     chunk_files, pending_by_chunk, total_pending) -> JudgeRunSummary:
    workflow = WorkflowLogger(log, "judge", width=cfg.output.log_banner_width)

    if not pending_by_chunk:
        log.info("  No samples need judgment. Refreshing manifest statistics.")

    prompt_max_tokens = max(int(args.prompt_max_tokens or 0), 0)
    judge_start = time.time()
    workflow.header(
        "ChainUQ Phase 1.5: LLM-as-Judge",
        cache_dir=cache_dir,
        split_dir=split_dir,
        judge_model=judge_model,
        batch_size=judge_batch_size,
        prompt_max_tokens=prompt_max_tokens if prompt_max_tokens > 0 else "disabled",
        force=args.force,
    )

    total_correct = 0
    total_incorrect = 0
    total_unparseable = 0
    total_processed = 0
    stage2_ok = 0
    stage2_partial = 0
    stage2_failed = 0
    stage2_discarded = 0
    dropped_samples = 0
    drop_reason_counter = Counter()
    active_backend = args.judge_backend or cfg.judge.judge_backend
    guided_enabled = bool(cfg.judge.guided_decoding_enabled and active_backend == "vllm")
    include_analysis = bool(cfg.judge.guided_json_with_analysis)
    thinking_budget = cfg.judge.thinking_token_budget
    stage1_prompt_truncated = 0
    stage2_prompt_truncated = 0
    stage1_prompt_token_sum = 0
    stage2_prompt_token_sum = 0
    stage1_prompt_count = 0
    stage2_prompt_count = 0

    for chunk_idx in sorted(pending_by_chunk.keys()):
        chunk_path = chunk_files[chunk_idx]
        pending_indices = pending_by_chunk[chunk_idx]
        workflow.event(
            "chunk.start",
            chunk_index=chunk_idx + 1,
            total_chunks=len(chunk_files),
            pending_samples=len(pending_indices),
            chunk_path=chunk_path,
        )

        chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
        pending_samples = [(chunk_idx, idx, chunk_data[idx]) for idx in pending_indices]
        dropped_indices = set()

        for batch_start in range(0, len(pending_samples), judge_batch_size):
            batch = pending_samples[batch_start:batch_start + judge_batch_size]

            # Stage 1: judge conclusion against ground truth, then copy that
            # verdict to both sample label and conclusion label.
            stage1_prompts = []
            stage1_meta = []
            for ci, local_idx, sample in batch:
                if local_idx in dropped_indices:
                    continue
                if bool(sample.get("_drop_sample", False)):
                    dropped_indices.add(local_idx)
                    continue
                sample = _enforce_sample_claim_contract(sample)
                chunk_data[local_idx] = sample
                ground_truth = sample.get("ground_truth", "")
                claims, _ = _sample_claims_and_verified(sample)
                if not claims:
                    total_unparseable += 1
                    dropped_samples += 1
                    drop_reason_counter["stage1_claim_contract_failed"] += 1
                    _mark_sample_dropped(sample, "stage1_claim_contract_failed")
                    dropped_indices.add(local_idx)
                    chunk_data[local_idx] = sample
                    continue
                conclusion_pos = -1
                conclusion_text = ""
                for c_idx, c in enumerate(claims):
                    if not isinstance(c, dict):
                        continue
                    if str(c.get("claim_type", "")).strip().lower() == "conclusion":
                        conclusion_pos = c_idx
                        conclusion_text = str(c.get("text", "")).strip()

                dataset_name = sample.get("dataset_name", "")
                label_space = _get_dataset_label_space(dataset_name)

                stage1_prompt, stage1_original_len, stage1_final_len = _build_stage1_prompt(
                    tokenizer=engine.tokenizer,
                    ground_truth=ground_truth,
                    conclusion_text=conclusion_text,
                    allowed_labels=label_space,
                    include_analysis=include_analysis,
                    prompt_max_tokens=prompt_max_tokens,
                )
                stage1_prompt_token_sum += stage1_final_len
                if stage1_final_len < stage1_original_len:
                    stage1_prompt_truncated += 1
                stage1_prompt_count += 1
                stage1_prompts.append(stage1_prompt)
                stage1_meta.append((ci, local_idx, conclusion_pos))

            if stage1_prompts:
                # Without analysis, keep the budget tiny to limit run-on decoding.
                # With ``guided_json_with_analysis``, a long ``analysis`` string needs
                # the full ``judge_max_new_tokens`` budget; capping at 64 truncates
                # mid-JSON and breaks strict parsing.
                if include_analysis:
                    stage1_max_tokens = min(judge_max_new_tokens, 512)
                else:
                    stage1_max_tokens = max(24, min(judge_max_new_tokens, 64))
                stage1_schema = _build_stage1_schema(include_analysis=include_analysis)
                stage1_outputs = engine.generate_text_only(
                    stage1_prompts,
                    stage1_max_tokens,
                    structured_json_schema=stage1_schema if guided_enabled else None,
                    thinking_token_budget=thinking_budget if guided_enabled else None,
                )

                for response, (ci, local_idx, conclusion_pos) in zip(stage1_outputs, stage1_meta):
                    answer_label, parse_status = _parse_stage1_output(response)
                    sample = chunk_data[local_idx]
                    if parse_status != "exact" or answer_label not in (0, 1):
                        total_unparseable += 1
                        log.warning(
                            "  Stage-1 parse failed (chunk=%d, idx=%d): %r",
                            ci, local_idx, response[:200],
                        )
                        dropped_samples += 1
                        drop_reason_counter["stage1_parse_failed"] += 1
                        _mark_sample_dropped(sample, "stage1_parse_failed")
                        dropped_indices.add(local_idx)
                        continue

                    sample["label"] = int(answer_label)
                    sample["answer_correct"] = bool(int(answer_label) == 1)
                    if int(answer_label) == 1:
                        total_correct += 1
                    else:
                        total_incorrect += 1
                    claims, verified = _sample_claims_and_verified(sample)
                    if claims:
                        if not isinstance(verified, list) or len(verified) != len(claims):
                            verified = [-1] * len(claims)
                        if conclusion_pos >= 0 and conclusion_pos < len(verified):
                            verified[conclusion_pos] = int(answer_label)
                        _write_sample_verified(sample, claims, verified)

            # Stage 2: reasoning chain consistency
            stage2_prompts = []
            stage2_meta = []
            for ci, local_idx, sample in batch:
                if local_idx in dropped_indices:
                    continue
                sample = chunk_data[local_idx]
                if int(sample.get("label", -1)) == -1:
                    continue
                claims, _ = _sample_claims_and_verified(sample)
                reasoning_positions = []
                reasoning_texts = []
                conclusion_text = ""
                for c_idx, c in enumerate(claims):
                    if not isinstance(c, dict):
                        continue
                    ctype = str(c.get("claim_type", "")).strip().lower()
                    if ctype == "reasoning":
                        reasoning_positions.append(c_idx)
                        reasoning_texts.append(str(c.get("text", "")))
                    elif ctype == "conclusion":
                        conclusion_text = str(c.get("text", ""))
                if not reasoning_texts:
                    continue
                stage2_prompt, stage2_original_len, stage2_final_len = _build_stage2_prompt(
                    tokenizer=engine.tokenizer,
                    question=sample.get("question", ""),
                    generated_text=sample.get("generated_text", ""),
                    reasoning_claims=reasoning_texts,
                    conclusion_text=conclusion_text,
                    include_analysis=include_analysis,
                    prompt_max_tokens=prompt_max_tokens,
                )
                stage2_prompt_token_sum += stage2_final_len
                if stage2_final_len < stage2_original_len:
                    stage2_prompt_truncated += 1
                stage2_prompt_count += 1
                stage2_prompts.append(stage2_prompt)
                stage2_meta.append((ci, local_idx, reasoning_positions))

            if stage2_prompts:
                # Same tradeoff as Stage-1: analysis + long step lists need headroom.
                max_reasoning = max(len(pos) for _, _, pos in stage2_meta)
                min_for_array = max(48, 32 + 8 * max_reasoning)
                if include_analysis:
                    stage2_max_tokens = max(
                        min_for_array,
                        min(judge_max_new_tokens, 512),
                    )
                else:
                    stage2_max_tokens = max(
                        48,
                        min(judge_max_new_tokens, 32 + 8 * max_reasoning),
                    )
                stage2_outputs = [None] * len(stage2_prompts)
                grouped_indices = defaultdict(list)
                for i, (_, _, reasoning_positions) in enumerate(stage2_meta):
                    grouped_indices[len(reasoning_positions)].append(i)

                for expected_len, indices in grouped_indices.items():
                    for start in range(0, len(indices), judge_batch_size):
                        sub_indices = indices[start:start + judge_batch_size]
                        sub_prompts = [stage2_prompts[i] for i in sub_indices]
                        sub_outputs = engine.generate_text_only(
                            sub_prompts,
                            stage2_max_tokens,
                            structured_json_schema=(
                                _build_stage2_schema(
                                    expected_len=expected_len,
                                    include_analysis=include_analysis,
                                )
                                if guided_enabled
                                else None
                            ),
                            thinking_token_budget=thinking_budget if guided_enabled else None,
                        )
                        for local_out_idx, out_text in enumerate(sub_outputs):
                            stage2_outputs[sub_indices[local_out_idx]] = out_text
                for response, (ci, local_idx, reasoning_positions) in zip(stage2_outputs, stage2_meta):
                    parsed_reasoning, parse_status = _parse_stage2_output(
                        response,
                        expected_len=len(reasoning_positions),
                    )
                    discard_sample = parse_status != "exact" or any(v == -1 for v in parsed_reasoning)
                    if parse_status == "failed":
                        stage2_failed += 1
                        if stage2_failed <= 3:
                            log.warning("  Stage-2 parse failed (chunk=%d idx=%d, %d claims): %r",
                                        ci, local_idx, len(reasoning_positions), response[:200])
                    elif discard_sample:
                        stage2_partial += 1
                    else:
                        stage2_ok += 1
                    if -1 in parsed_reasoning:
                        log.warning("  Stage-2 has -1 values (chunk=%d idx=%d, status=%s, parsed=%s): %r",
                                    ci, local_idx, parse_status, parsed_reasoning, response[:300])
                    sample = chunk_data[local_idx]
                    claims, verified = _sample_claims_and_verified(sample)
                    if not claims:
                        continue
                    if discard_sample:
                        stage2_discarded += 1
                        dropped_samples += 1
                        drop_reason = "stage2_parse_failed" if parse_status == "failed" else "stage2_schema_failed"
                        drop_reason_counter[drop_reason] += 1
                        _mark_sample_dropped(sample, drop_reason)
                        dropped_indices.add(local_idx)
                        continue
                    if not isinstance(verified, list) or len(verified) != len(claims):
                        verified = [-1] * len(claims)
                    for ri, claim_pos in enumerate(reasoning_positions):
                        if 0 <= claim_pos < len(verified):
                            verified[claim_pos] = int(parsed_reasoning[ri])
                    _write_sample_verified(sample, claims, verified)

            total_processed += len(batch)
            log.info(
                "  Progress: %d/%d samples (s1: correct=%d incorrect=%d unparseable=%d | "
                "s2: ok=%d partial=%d failed=%d discarded=%d | dropped=%d)",
                total_processed, total_pending,
                total_correct, total_incorrect, total_unparseable,
                stage2_ok, stage2_partial, stage2_failed, stage2_discarded, dropped_samples,
            )

            # Save chunk after each batch for crash recovery
            save_torch_atomic(chunk_data, chunk_path)
            log.debug("  Intermediate save after batch %d-%d", batch_start, batch_start + len(batch))

        # Final chunk save with diagnostics
        if dropped_indices:
            before = len(chunk_data)
            chunk_data = [s for i, s in enumerate(chunk_data) if i not in dropped_indices]
            log.warning(
                "  Dropped %d sample(s) from chunk %d due to strict judge failures.",
                before - len(chunk_data),
                chunk_idx,
            )
        chunk_claim_type_label_ratio = _compute_claim_type_label_ratio(chunk_data)
        log.info("  Chunk %d claim_type_label_ratio=%s", chunk_idx, chunk_claim_type_label_ratio)
        save_torch_atomic(chunk_data, chunk_path)
        log.info("  Saved updated chunk: %s", chunk_path.name)
        del chunk_data

    # Update manifest (re-scan chunks for final stats)
    log.info("Updating manifest with final statistics...")
    manifest_path = split_dir / "manifest.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
    else:
        manifest = {}
    manifest.setdefault("schema_version", SCHEMA_VERSION)

    all_labels = []
    unparseable_samples = []
    total_claim_pending = 0
    type_label_counter = defaultdict(Counter)
    index = []

    # Per claim-type and per n_hops accumulators
    type_stats = defaultdict(lambda: {"count": 0, "correct": 0, "incorrect": 0, "pending": 0, "char_total": 0})
    nhop_stats = defaultdict(lambda: {
        "samples": 0, "sample_correct": 0,
        "claims": 0, "claims_correct": 0, "claims_incorrect": 0, "claims_pending": 0,
        "char_total": 0,
        "type_stats": defaultdict(lambda: {"count": 0, "correct": 0, "incorrect": 0, "char_total": 0}),
    })

    for chunk_idx, chunk_path in enumerate(chunk_files):
        if not chunk_path.exists():
            continue
        chunk_data = load_torch_with_tmp_recovery(chunk_path, logger=log)
        if chunk_data is None:
            continue
        for sample_idx, sample in enumerate(chunk_data):
            label = sample.get("label", 1)
            all_labels.append(label)
            if label == -1:
                unparseable_samples.append({"chunk": chunk_idx, "idx": sample_idx})
            claims, verified = _sample_claims_and_verified(sample)
            n_hops = sample.get("n_hops", 0)

            # Per n_hops sample-level
            nh = nhop_stats[n_hops]
            nh["samples"] += 1
            if label == 1:
                nh["sample_correct"] += 1

            # Per-sample claim stats for index
            sample_claims_total = len(verified) if isinstance(verified, list) else 0
            sample_claims_pending = 0
            sample_claims_correct = 0
            sample_claims_incorrect = 0
            n_reasoning = 0
            n_conclusion = 0
            n_reasoning_correct = 0
            n_conclusion_correct = 0

            if isinstance(claims, list) and isinstance(verified, list):
                for ci, claim in enumerate(claims):
                    ctype = "unknown"
                    char_len = 0
                    if isinstance(claim, dict):
                        ctype = str(claim.get("claim_type", "unknown")).strip().lower() or "unknown"
                        text = claim.get("text", "") or ""
                        char_len = len(text)

                    iv = None
                    lbl = "invalid"
                    if ci < len(verified):
                        try:
                            iv = int(verified[ci])
                            if iv in (-1, 0, 1):
                                lbl = str(iv)
                        except Exception:
                            pass

                    if ctype == "reasoning":
                        n_reasoning += 1
                        if iv == 1:
                            n_reasoning_correct += 1
                    elif ctype == "conclusion":
                        n_conclusion += 1
                        if iv == 1:
                            n_conclusion_correct += 1
                    type_label_counter[ctype][lbl] += 1

                    if iv == -1:
                        total_claim_pending += 1
                        sample_claims_pending += 1
                    elif iv == 1:
                        sample_claims_correct += 1
                    elif iv == 0:
                        sample_claims_incorrect += 1

                    # Per claim-type stats
                    ts = type_stats[ctype]
                    ts["count"] += 1
                    ts["char_total"] += char_len
                    if iv == 1:
                        ts["correct"] += 1
                    elif iv == 0:
                        ts["incorrect"] += 1
                    elif iv == -1:
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
                    elif iv == -1:
                        nh["claims_pending"] += 1
            elif isinstance(verified, list):
                for v in verified:
                    try:
                        if int(v) == -1:
                            total_claim_pending += 1
                            sample_claims_pending += 1
                    except Exception:
                        continue

            # Build index entry
            index.append({
                "chunk_file": chunk_path.name,
                "local_idx": sample_idx,
                "sample_id": int(sample.get("sample_id", -1)),
                "sample_uid": str(sample.get("sample_uid", "")),
                "label": label,
                "n_hops": n_hops,
                "n_reasoning_claims": n_reasoning,
                "n_conclusion_claims": n_conclusion,
                "n_reasoning_correct": n_reasoning_correct,
                "n_conclusion_correct": n_conclusion_correct,
                "claims_correct": sample_claims_correct,
                "claims_incorrect": sample_claims_incorrect,
                "dataset_name": sample.get("dataset_name", ""),
                "claims_total": sample_claims_total,
                "claims_pending": sample_claims_pending,
            })
        del chunk_data
    gc.collect()

    total_count = len(all_labels)
    n_correct = sum(1 for l in all_labels if l == 1)
    n_incorrect = sum(1 for l in all_labels if l == 0)
    n_pending = sum(1 for l in all_labels if l == -1)
    total_labeled = total_count - n_pending
    correct_rate = n_correct / max(total_labeled, 1)
    incorrect_rate = n_incorrect / max(total_labeled, 1)

    total_claims = sum(sum(counter.values()) for counter in type_label_counter.values())
    claims_correct = sum(counter.get("1", 0) for counter in type_label_counter.values())
    claims_incorrect = sum(counter.get("0", 0) for counter in type_label_counter.values())
    claims_pending_count = sum(counter.get("-1", 0) for counter in type_label_counter.values())
    no_fully_verified_claim_samples = sum(
        1
        for entry in index
        if int(entry.get("claims_total", 0) or 0) <= 0
        or int(entry.get("claims_pending", 0) or 0) > 0
    )

    overall_claim_type_label_ratio = {}
    for ctype, counter in type_label_counter.items():
        denom = float(sum(counter.values()))
        if denom <= 0:
            continue
        ordered = {}
        for key in ("-1", "0", "1", "invalid"):
            if key in counter:
                ordered[key] = round(float(counter[key]) / denom, 6)
        for key, count in counter.items():
            if key not in ordered:
                ordered[key] = round(float(count) / denom, 6)
        overall_claim_type_label_ratio[ctype] = ordered

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
    avg_claim_char_length = round(total_char / max(total_claims, 1), 1)

    # Build per_nhop_stats summary
    per_nhop_stats = {}
    for nh_val in sorted(nhop_stats.keys()):
        nh = nhop_stats[nh_val]
        labeled_claims = nh["claims_correct"] + nh["claims_incorrect"]
        nh_type = {}
        for ctype, nht in nh["type_stats"].items():
            nht_labeled = nht["correct"] + nht["incorrect"]
            nh_type[ctype] = {
                "count": nht["count"],
                "avg_char_length": round(nht["char_total"] / max(nht["count"], 1), 1),
                "correct_rate": round(nht["correct"] / max(nht_labeled, 1), 4) if nht_labeled > 0 else None,
            }
        per_nhop_stats[str(nh_val)] = {
            "samples": nh["samples"],
            "sample_correct_rate": round(nh["sample_correct"] / max(nh["samples"], 1), 4),
            "claims": nh["claims"],
            "claims_pending": nh["claims_pending"],
            "avg_claim_char_length": round(nh["char_total"] / max(nh["claims"], 1), 1),
            "claims_correct_rate": round(nh["claims_correct"] / max(labeled_claims, 1), 4) if labeled_claims > 0 else None,
            "claim_types": nh_type,
        }

    judge_duration = time.time() - judge_start

    # Preserve judge processing stats from prior runs when no new work was done
    if total_pending > 0:
        judge_stats = {
            "judge_samples_processed": total_pending,
            "judge_unparseable": total_unparseable,
            "judge_stage2_ok": stage2_ok,
            "judge_stage2_partial": stage2_partial,
            "judge_stage2_failed": stage2_failed,
            "judge_stage2_discarded": stage2_discarded,
            "judge_dropped_samples": dropped_samples,
            "judge_drop_reason_counts": dict(drop_reason_counter),
            "judge_unparseable_samples": unparseable_samples[:100],
            "judge_prompt_max_tokens": prompt_max_tokens,
            "judge_stage1_prompt_truncated": int(stage1_prompt_truncated),
            "judge_stage2_prompt_truncated": int(stage2_prompt_truncated),
            "judge_stage1_avg_prompt_tokens": round(stage1_prompt_token_sum / max(stage1_prompt_count, 1), 2),
            "judge_stage2_avg_prompt_tokens": round(stage2_prompt_token_sum / max(stage2_prompt_count, 1), 2),
        }
    else:
        judge_stats = {
            k: manifest.get(k, 0)
            for k in ("judge_samples_processed", "judge_unparseable",
                       "judge_stage2_ok", "judge_stage2_partial", "judge_stage2_failed",
                       "judge_stage2_discarded", "judge_dropped_samples")
        }
        judge_stats["judge_drop_reason_counts"] = manifest.get("judge_drop_reason_counts", {})
        judge_stats["judge_unparseable_samples"] = manifest.get("judge_unparseable_samples", [])
        judge_stats["judge_prompt_max_tokens"] = manifest.get("judge_prompt_max_tokens", prompt_max_tokens)
        judge_stats["judge_stage1_prompt_truncated"] = manifest.get("judge_stage1_prompt_truncated", 0)
        judge_stats["judge_stage2_prompt_truncated"] = manifest.get("judge_stage2_prompt_truncated", 0)
        judge_stats["judge_stage1_avg_prompt_tokens"] = manifest.get("judge_stage1_avg_prompt_tokens", 0)
        judge_stats["judge_stage2_avg_prompt_tokens"] = manifest.get("judge_stage2_avg_prompt_tokens", 0)

    phase_status = manifest.get("phase_status", {})
    phase_status["judge"] = {
        "status": "complete",
        "duration_seconds": round(judge_duration, 1),
        "samples_judged": total_pending,
    }

    manifest.update({
        "total_samples": total_count,
        "total_claims": total_claims,
        "sample_correct": n_correct,
        "sample_incorrect": n_incorrect,
        "sample_pending": n_pending,
        "correct_rate": correct_rate,
        "incorrect_rate": incorrect_rate,
        "claims_correct": claims_correct,
        "claims_incorrect": claims_incorrect,
        "claims_pending": claims_pending_count,
        "avg_claim_char_length": avg_claim_char_length,
        "no_fully_verified_claim_samples": int(no_fully_verified_claim_samples),
        "no_usable_verified_claim_samples": int(no_fully_verified_claim_samples),
        "claim_type_stats": claim_type_stats,
        "per_nhop_stats": per_nhop_stats,
        "judge_model": judge_model,
        "total_claim_pending": total_claim_pending,
        "claim_type_label_ratio": overall_claim_type_label_ratio,
        "phase_status": phase_status,
    })
    manifest.update(judge_stats)

    write_manifest(manifest_path, manifest)

    # Write index to separate file (keeps manifest small)
    index_path = Path(manifest_path).parent / "index.json"
    write_index(index_path, index)

    workflow.stage_end(
        "run",
        total_samples=total_count,
        sample_correct=n_correct,
        sample_incorrect=n_incorrect,
        sample_pending=n_pending,
        total_claims=total_claims,
        claim_correct=claims_correct,
        claim_incorrect=claims_incorrect,
        claim_pending=claims_pending_count,
        unparseable=total_unparseable,
        stage2_ok=stage2_ok,
        stage2_partial=stage2_partial,
        stage2_failed=stage2_failed,
        stage2_discarded=stage2_discarded,
        dropped_samples=dropped_samples,
        duration_s=judge_duration,
    )
    workflow.event(
        "label_quality",
        cache=Path(cache_dir).name or cache_dir,
        split=split,
        labeled_samples=total_labeled,
        labeled_rate=total_labeled / max(total_count, 1),
        pending_samples=n_pending,
        pending_rate=n_pending / max(total_count, 1),
        pending_claims=total_claim_pending,
        pending_claim_rate=total_claim_pending / max(total_claims, 1),
        parse_failures=total_unparseable,
        parse_failure_rate=total_unparseable / max(total_pending, 1),
    )
    if total_labeled > 0:
        imbalance_ratio = max(n_correct, n_incorrect) / max(min(n_correct, n_incorrect), 1)
        workflow.event(
            "class_balance",
            correct=n_correct,
            correct_rate=n_correct / max(total_labeled, 1),
            incorrect=n_incorrect,
            incorrect_rate=n_incorrect / max(total_labeled, 1),
            imbalance_ratio=imbalance_ratio,
            severity="balanced" if imbalance_ratio < 3 else "moderate" if imbalance_ratio < 5 else "severe",
        )
    for ctype in sorted(overall_claim_type_label_ratio.keys()):
        ratios = overall_claim_type_label_ratio[ctype]
        workflow.event("claim_type_ratio", claim_type=ctype, ratios=ratios)

    workflow.artifact("manifest.saved", manifest_path, split=split)
    workflow.artifact("index.saved", index_path, split=split)
    return {
        "split": split,
        "total_samples": total_count,
        "sample_pending": n_pending,
        "total_claims": total_claims,
        "claim_pending": claims_pending_count,
        "samples_judged": total_pending,
        "dropped_samples": dropped_samples,
    }


def main():
    args = parse_args()
    cfg = Config()
    configure_logging(cfg, force=True)
    workflow = WorkflowLogger(log, "judge", width=cfg.output.log_banner_width)
    runner = PipelineRunner(workflow)

    judge_model = args.judge_model or cfg.judge.judge_model_path
    judge_backend = args.judge_backend or cfg.judge.judge_backend
    judge_batch_size = args.batch_size or cfg.judge.judge_batch_size
    judge_max_new_tokens = args.max_new_tokens or cfg.judge.judge_max_new_tokens
    guided_enabled = bool(cfg.judge.guided_decoding_enabled)

    cache_dirs = _parse_csv_items(args.cache_dir)
    splits = _parse_csv_items(args.split)
    pending_targets = _discover_targets(cache_dirs, splits)

    if not pending_targets:
        log.info("No cache/split target needs judgment. Done.")
        return

    # Pre-scan: check if any target actually has pending samples before loading
    # the expensive vLLM engine. Prefer chunk scans over potentially stale index metadata.
    log.info("Pre-scan: checking %d target(s) for pending samples...", len(pending_targets))
    has_pending_work = False
    for cache_dir, split, split_dir in pending_targets:
        if _target_has_pending_judgment(split_dir, args):
            has_pending_work = True
            break

    if not has_pending_work:
        log.info("Pre-scan: 0 pending samples across all targets. Skipping engine load.")
        return

    if not judge_model:
        log.error("Samples need judgment but no judge model specified. "
                  "Use --judge_model or set judge.judge_model_path in config.")
        return

    if guided_enabled and judge_backend != "vllm":
        log.error("judge.guided_decoding_enabled=true requires judge_backend=vllm.")
        return
    if guided_enabled:
        try:
            __import__("vllm.sampling_params")
        except Exception as exc:
            log.error(
                "Guided decoding requires vLLM structured outputs support. "
                "Please upgrade vLLM. Error: %s: %s",
                type(exc).__name__,
                exc,
            )
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workflow.header(
        "Judge Pipeline",
        backend=judge_backend,
        judge_model=judge_model,
        targets=len(pending_targets),
        force=args.force,
    )

    def _load_engine():
        engine_kwargs = {
            "backend": judge_backend,
            "cfg": cfg,
            "model_path": judge_model,
            "device": device,
        }
        if judge_backend == "vllm":
            engine_kwargs["text_only"] = True
        return get_engine(**engine_kwargs)

    def _validate_target_result(result: JudgeRunSummary) -> None:
        if not isinstance(result, dict):
            raise TypeError("judge target result must be dict")

    runner.register_stage(
        StageSpec(
            name="engine_load",
            fn=lambda payload: _load_engine(),
            start_fields=lambda payload: {
                "backend": judge_backend,
                "judge_model": judge_model,
                "device": device.type,
            },
            include_duration=True,
        )
    )

    runner.register_stage(
        StageSpec(
            name="target",
            fn=lambda payload: process_split(
                payload["cache_dir"],
                payload["split"],
                payload["split_dir"],
                args,
                cfg,
                payload["engine"],
                judge_model,
                judge_batch_size,
                judge_max_new_tokens,
            ) or {"split": payload["split"], "status": "skipped"},
            start_fields=lambda payload: {
                "cache_dir": payload["cache_dir"],
                "split": payload["split"],
            },
            output_contract=_validate_target_result,
            result_fields=lambda result: result,
        )
    )

    runner.register_stage(
        StageSpec(
            name="engine_unload",
            fn=lambda payload: payload.unload(),
            start_fields=lambda payload: {"backend": judge_backend},
            include_duration=False,
        )
    )

    engine = runner.run("engine_load")

    for cache_dir, split, split_dir in pending_targets:
        runner.run(
            "target",
            {
                "cache_dir": cache_dir,
                "split": split,
                "split_dir": split_dir,
                "engine": engine,
            },
        )

    runner.run("engine_unload", engine)


if __name__ == "__main__":
    main()
