#!/usr/bin/env python3
"""
Cleanup script: Remove samples with pending claims from cached chunks.

When a sample has any claims with verification status = -1 (pending),
we remove the entire sample (and all its claims) to ensure clean training data.

Usage:
    python scripts/cleanup_pending_claims.py --cache_dir <dir> --split train
    python scripts/cleanup_pending_claims.py --cache_dir <dir> --split train,validation,test
"""

import sys
import json
import os
import gc
import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict

import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.cache_io import write_json_atomic, save_torch_atomic
from utils.log import WorkflowLogger, configure_logging, get_logger

log = get_logger(__name__)
SCHEMA_VERSION = "split-storage.v1"


def load_manifest(split_dir: str) -> Dict:
    """Load manifest.json if exists."""
    manifest_path = Path(split_dir) / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def has_pending_claims(sample: dict) -> bool:
    """Check if sample has any pending claims (verified == -1)."""
    if "claims_pending" in sample:
        try:
            return int(sample.get("claims_pending", 0)) > 0
        except Exception:
            pass
    verified = list(sample.get("reasoning_verified", []) or [])
    if "conclusion_verified" in sample:
        verified.append(sample.get("conclusion_verified", -1))
    if not verified:
        return False
    return any(str(v).strip() == "-1" for v in verified)


def _convert_sample_to_compact_and_sidecar(sample: dict) -> tuple[dict, dict]:
    """Split one sample into compact (train-fast) and reasoning sidecar payload."""
    reasoning_claims_raw = list(sample.get("reasoning_claims", []) or [])
    reasoning_verified_raw = list(sample.get("reasoning_verified", []) or [])
    conclusion_claim_raw = sample.get("conclusion_claim")
    conclusion_verified_raw = sample.get("conclusion_verified", -1)
    claims = list(reasoning_claims_raw)
    verified = list(reasoning_verified_raw)
    if isinstance(conclusion_claim_raw, dict):
        claims.append(conclusion_claim_raw)
        verified.append(conclusion_verified_raw)
    features = sample.get("features")
    if not isinstance(features, torch.Tensor):
        raise ValueError("sample.features must be a torch.Tensor during cleanup")
    seq_len = int(features.shape[0])
    reasoning_claims = []
    reasoning_verified = []
    conclusion_claim = None
    conclusion_verified = -1
    n_reasoning_correct = 0
    n_conclusion_correct = 0
    claims_correct = 0
    claims_incorrect = 0
    claims_pending = 0
    reasoning_token_ids = []
    conclusion_token_ids = []

    def _claim_token_ids(claim: dict) -> list[int]:
        ids = []
        seen = set()
        for tid in claim.get("aligned_token_ids", []) if isinstance(claim, dict) else []:
            try:
                t = int(tid)
            except Exception:
                continue
            if 0 <= t < seq_len and t not in seen:
                ids.append(t)
                seen.add(t)
        return ids

    for i, claim in enumerate(claims):
        v = -1
        if i < len(verified):
            try:
                v = int(verified[i])
            except Exception:
                v = -1
        ctype = "unknown"
        if isinstance(claim, dict):
            ctype = str(claim.get("claim_type", "unknown")).strip().lower()
        if v == 1:
            claims_correct += 1
        elif v == 0:
            claims_incorrect += 1
        else:
            claims_pending += 1
        if ctype == "conclusion" and conclusion_claim is None:
            conclusion_claim = claim
            conclusion_verified = v
            conclusion_token_ids = _claim_token_ids(claim if isinstance(claim, dict) else {})
            if v == 1:
                n_conclusion_correct += 1
        else:
            reasoning_claims.append(claim)
            reasoning_verified.append(v)
            reasoning_token_ids.extend(_claim_token_ids(claim if isinstance(claim, dict) else {}))
            if v == 1:
                n_reasoning_correct += 1

    # Deduplicate while preserving order.
    if reasoning_token_ids:
        seen = set()
        reasoning_token_ids = [t for t in reasoning_token_ids if not (t in seen or seen.add(t))]
    if not isinstance(conclusion_claim, dict):
        raise ValueError("Missing conclusion claim during cleanup remap.")
    if not conclusion_token_ids:
        raise ValueError("Conclusion claim has no aligned token ids; refusing fallback remap.")

    # Build feature partitions.
    conclusion_features = features[conclusion_token_ids]
    conclusion_attention = torch.ones(conclusion_features.shape[0], dtype=torch.long)
    if reasoning_token_ids:
        reasoning_features = features[reasoning_token_ids]
    else:
        reasoning_features = torch.empty((0, features.shape[1]), dtype=features.dtype)
    reasoning_attention = torch.ones(reasoning_features.shape[0], dtype=torch.long)

    # Remap claim token ids into partition-local coordinates.
    conclusion_id_map = {tid: idx for idx, tid in enumerate(conclusion_token_ids)}
    reasoning_id_map = {tid: idx for idx, tid in enumerate(reasoning_token_ids)}

    def _remap_claim(claim: dict, id_map: dict[int, int]) -> dict:
        if not isinstance(claim, dict):
            return {"claim": str(claim), "claim_type": "unknown", "claim_type_id": 2, "aligned_token_ids": []}
        out = dict(claim)
        new_ids = []
        for t in claim.get("aligned_token_ids", []):
            try:
                tid = int(t)
            except Exception:
                continue
            if tid in id_map:
                new_ids.append(id_map[tid])
        out["aligned_token_ids"] = new_ids
        return out

    conclusion_claim_mapped = (
        _remap_claim(conclusion_claim, conclusion_id_map)
        if isinstance(conclusion_claim, dict)
        else None
    )
    if not isinstance(conclusion_claim_mapped, dict) or not (conclusion_claim_mapped.get("aligned_token_ids") or []):
        raise ValueError("Conclusion claim remap produced empty aligned ids; refusing fallback remap.")
    reasoning_claims_mapped = [
        _remap_claim(claim, reasoning_id_map)
        for claim in reasoning_claims
    ]

    compact = {
        "sample_id": int(sample.get("sample_id", -1)),
        "sample_uid": str(sample.get("sample_uid", "")),
        "features": conclusion_features,
        "attention_mask": conclusion_attention,
        "label": sample.get("label", -1),
        "question": sample.get("question", ""),
        "generated_text": sample.get("generated_text", ""),
        "n_hops": sample.get("n_hops", 0),
        "dataset_name": sample.get("dataset_name", ""),
        "conclusion_claim": conclusion_claim_mapped,
        "conclusion_verified": conclusion_verified,
        "n_reasoning_claims": len(reasoning_claims),
        "n_conclusion_claims": 1 if isinstance(conclusion_claim_mapped, dict) else 0,
        "n_reasoning_correct": n_reasoning_correct,
        "n_conclusion_correct": n_conclusion_correct,
        "claims_total": len(claims),
        "claims_correct": claims_correct,
        "claims_incorrect": claims_incorrect,
        "claims_pending": claims_pending,
    }
    sidecar = {
        "reasoning_claims": reasoning_claims_mapped,
        "reasoning_verified": reasoning_verified,
        "reasoning_features": reasoning_features,
        "reasoning_attention_mask": reasoning_attention,
        "token_probs": sample.get("token_probs"),
        "log_likelihoods": sample.get("log_likelihoods"),
    }
    return compact, sidecar


def cleanup_split(cache_dir: str, split: str) -> Dict:
    """
    Remove samples with pending claims from all chunks in a split.
    
    Returns:
        dict with keys:
            - total_samples_before
            - total_samples_after
            - removed_samples
            - chunks_modified (list of chunk files that were modified)
    """
    split_dir = Path(cache_dir) / split
    
    if not split_dir.is_dir():
        log.warning("Split directory not found: %s", split_dir)
        return {
            "total_samples_before": 0,
            "total_samples_after": 0,
            "removed_samples": 0,
            "chunks_modified": [],
        }
    
    # Find all chunk files
    chunk_files = sorted(
        [
            f.name
            for f in split_dir.glob("chunk_*.pt")
            if not f.name.endswith("_reasoning.pt")
        ]
    )
    if not chunk_files:
        log.info("No chunks found in %s", split_dir)
        return {
            "total_samples_before": 0,
            "total_samples_after": 0,
            "removed_samples": 0,
            "chunks_modified": [],
        }

    # Early check: skip only when no pending claims and split-storage sidecars already exist.
    manifest = load_manifest(str(split_dir))
    has_all_sidecars = all(
        (split_dir / f"{Path(cf).stem}_reasoning.pt").is_file() for cf in chunk_files
    )
    if manifest and manifest.get("total_claim_pending", 0) == 0 and has_all_sidecars:
        log.info(
            "[%s] No pending claims and split storage already prepared, skipping cleanup",
            split,
        )
        return {
            "total_samples_before": manifest.get("total_samples", 0),
            "total_samples_after": manifest.get("total_samples", 0),
            "removed_samples": 0,
            "chunks_modified": [],
        }
    
    stats = {
        "total_samples_before": 0,
        "total_samples_after": 0,
        "removed_samples": 0,
        "chunks_modified": [],
    }
    
    # Process each chunk independently to avoid memory buildup
    # Don't keep all chunks in memory at once
    for i, chunk_file in enumerate(chunk_files):
        chunk_path = split_dir / chunk_file
        sidecar_path = split_dir / f"{chunk_path.stem}_reasoning.pt"
        
        try:
            chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
            if not isinstance(chunk_data, list):
                log.warning("Chunk %s is not a list, skipping", chunk_file)
                continue
        except Exception as e:
            log.warning("Failed to load chunk %s: %s", chunk_file, e)
            continue

        sidecar_data = None
        if sidecar_path.is_file():
            try:
                loaded = torch.load(sidecar_path, map_location="cpu", weights_only=False)
                if isinstance(loaded, list):
                    sidecar_data = loaded
            except Exception as e:
                log.warning("Failed to load sidecar %s: %s", sidecar_path.name, e)
                sidecar_data = None
        
        original_size = len(chunk_data)
        stats["total_samples_before"] += original_size
        
        # Filter out samples with pending claims and convert to split-storage format.
        filtered_chunk = []
        reasoning_sidecar_chunk = []
        removed_in_chunk = 0
        
        for local_idx, sample in enumerate(chunk_data):
            merged_sample = sample
            if (
                isinstance(sidecar_data, list)
                and local_idx < len(sidecar_data)
                and isinstance(sidecar_data[local_idx], dict)
            ):
                merged_sample = dict(sample)
                merged_sample.update(sidecar_data[local_idx])

            if not has_pending_claims(merged_sample):
                try:
                    compact, sidecar = _convert_sample_to_compact_and_sidecar(merged_sample)
                except Exception as exc:
                    removed_in_chunk += 1
                    stats["removed_samples"] += 1
                    log.warning(
                        "Dropping sample from %s due to strict alignment cleanup failure: %s",
                        chunk_file,
                        exc,
                    )
                    continue
                filtered_chunk.append(compact)
                reasoning_sidecar_chunk.append(sidecar)
            else:
                removed_in_chunk += 1
                stats["removed_samples"] += 1
                log.debug(
                    "Removing sample from %s: question=%s...",
                    chunk_file,
                    sample.get("question", "")[:50],
                )
        
        stats["total_samples_after"] += len(filtered_chunk)
        
        # Always rewrite chunk into compact + reasoning-sidecar storage.
        reasoning_chunk_path = split_dir / f"{Path(chunk_file).stem}_reasoning.pt"
        if removed_in_chunk > 0:
            log.info(
                "Chunk %d/%d (%s): removing %d sample(s) with pending claims (%d remain)",
                i + 1,
                len(chunk_files),
                chunk_file,
                removed_in_chunk,
                len(filtered_chunk),
            )
        else:
            log.info(
                "Chunk %d/%d (%s): no pending claims (%d samples), converting to split storage",
                i + 1,
                len(chunk_files),
                chunk_file,
                original_size,
            )
        save_torch_atomic(filtered_chunk, chunk_path)
        save_torch_atomic(reasoning_sidecar_chunk, reasoning_chunk_path)
        stats["chunks_modified"].append(chunk_file)
        log.info("Saved compact + reasoning chunks: %s / %s", chunk_file, reasoning_chunk_path.name)
        
        # Clean up memory immediately after processing this chunk
        del chunk_data
        del filtered_chunk
        del reasoning_sidecar_chunk
        if sidecar_data is not None:
            del sidecar_data
        gc.collect()
    
    # Rebuild index and manifest stats from cleaned chunks
    log.info("Rebuilding index for %s...", split)
    new_index, rebuilt_stats = _build_index_and_stats_from_chunks_streaming(split_dir, chunk_files)
    index_path = split_dir / "index.json"
    write_json_atomic(index_path, new_index)
    log.info("Rebuilt index with %d samples", len(new_index))
    
    # Update manifest with correct statistics
    manifest = load_manifest(str(split_dir))
    if manifest:
        manifest.setdefault("schema_version", SCHEMA_VERSION)
        manifest.update(rebuilt_stats)
        
        manifest_path = split_dir / "manifest.json"
        write_json_atomic(manifest_path, manifest)
        log.info(
            "Updated manifest: total_samples=%d total_claims=%d total_claim_pending=%d no_usable_verified_claim_samples=%d",
            rebuilt_stats["total_samples"],
            rebuilt_stats["total_claims"],
            rebuilt_stats["total_claim_pending"],
            rebuilt_stats["no_usable_verified_claim_samples"],
        )
    
    return stats


def _build_index_and_stats_from_chunks_streaming(split_dir: Path, chunk_files: List[str]):
    """Rebuild index and manifest stats by streaming cleaned chunks."""
    index = []
    chunk_sample_counts = []
    total_claim_pending = 0
    type_label_counter = defaultdict(Counter)
    type_stats = defaultdict(lambda: {"count": 0, "correct": 0, "incorrect": 0, "pending": 0, "char_total": 0})
    nhop_stats = defaultdict(
        lambda: {
            "samples": 0,
            "sample_correct": 0,
            "claims": 0,
            "claims_correct": 0,
            "claims_incorrect": 0,
            "claims_pending": 0,
            "char_total": 0,
            "type_stats": defaultdict(lambda: {"count": 0, "correct": 0, "incorrect": 0, "char_total": 0}),
        }
    )

    for chunk_file in sorted(chunk_files):
        chunk_path = split_dir / chunk_file
        sidecar_path = split_dir / f"{Path(chunk_file).stem}_reasoning.pt"
        try:
            chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
            if not isinstance(chunk_data, list):
                log.warning("Index rebuild: chunk %s is not a list, skipping", chunk_file)
                continue
        except Exception as e:
            log.warning("Index rebuild: failed to load %s: %s", chunk_file, e)
            continue

        sidecar_data = []
        if sidecar_path.is_file():
            try:
                sidecar_data = torch.load(sidecar_path, map_location="cpu", weights_only=False)
                if not isinstance(sidecar_data, list):
                    sidecar_data = []
            except Exception as e:
                log.warning("Index rebuild: failed to load sidecar %s: %s", sidecar_path.name, e)
                sidecar_data = []

        chunk_count = 0
        for local_idx, sample in enumerate(chunk_data):
            chunk_count += 1
            sidecar = sidecar_data[local_idx] if local_idx < len(sidecar_data) and isinstance(sidecar_data[local_idx], dict) else {}
            claims = list(sidecar.get("reasoning_claims", []) or [])
            verified = list(sidecar.get("reasoning_verified", []) or [])
            conc_claim = sample.get("conclusion_claim")
            conc_verified = sample.get("conclusion_verified", -1)
            if isinstance(conc_claim, dict):
                claims.append(conc_claim)
                verified.append(conc_verified)
            label = sample.get("label", -1)
            n_hops = sample.get("n_hops", 0)

            nh = nhop_stats[n_hops]
            nh["samples"] += 1
            if int(label) == 1:
                nh["sample_correct"] += 1

            n_reasoning = 0
            n_conclusion = 0
            n_reasoning_correct = 0
            n_conclusion_correct = 0
            sample_claims_correct = 0
            sample_claims_incorrect = 0
            sample_claims_pending = 0

            if isinstance(claims, list) and isinstance(verified, list) and claims:
                for ci, claim in enumerate(claims):
                    ctype = "unknown"
                    char_len = 0
                    if isinstance(claim, dict):
                        ctype = str(claim.get("claim_type", "unknown")).strip().lower() or "unknown"
                        char_len = len(claim.get("text", "") or claim.get("claim", "") or "")

                    iv = None
                    lbl = "invalid"
                    if ci < len(verified):
                        try:
                            iv = int(verified[ci])
                            if iv in (-1, 0, 1):
                                lbl = str(iv)
                        except Exception:
                            pass
                    type_label_counter[ctype][lbl] += 1

                    if ctype == "reasoning":
                        n_reasoning += 1
                        if iv == 1:
                            n_reasoning_correct += 1
                    elif ctype == "conclusion":
                        n_conclusion += 1
                        if iv == 1:
                            n_conclusion_correct += 1

                    if iv == -1:
                        total_claim_pending += 1
                        sample_claims_pending += 1
                    elif iv == 1:
                        sample_claims_correct += 1
                    elif iv == 0:
                        sample_claims_incorrect += 1

                    ts = type_stats[ctype]
                    ts["count"] += 1
                    ts["char_total"] += char_len
                    if iv == 1:
                        ts["correct"] += 1
                    elif iv == 0:
                        ts["incorrect"] += 1
                    elif iv == -1:
                        ts["pending"] += 1

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
            else:
                # Compact split-storage fallback: use pre-computed counts.
                n_reasoning = int(sample.get("n_reasoning_claims", 0) or 0)
                n_conclusion = int(sample.get("n_conclusion_claims", 0) or 0)
                n_reasoning_correct = int(sample.get("n_reasoning_correct", 0) or 0)
                n_conclusion_correct = int(sample.get("n_conclusion_correct", 0) or 0)
                sample_claims_correct = int(sample.get("claims_correct", 0) or 0)
                sample_claims_incorrect = int(sample.get("claims_incorrect", 0) or 0)
                sample_claims_pending = int(sample.get("claims_pending", 0) or 0)
                nh["claims"] += int(sample.get("claims_total", 0) or 0)
                nh["claims_correct"] += sample_claims_correct
                nh["claims_incorrect"] += sample_claims_incorrect
                nh["claims_pending"] += sample_claims_pending
                total_claim_pending += sample_claims_pending

            index.append({
                "chunk_file": chunk_file,
                "local_idx": local_idx,
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
                "claims_total": len(claims),
                "claims_pending": sample_claims_pending,
            })
        chunk_sample_counts.append(chunk_count)

        # Release memory immediately after processing this chunk
        del chunk_data
        del sidecar_data
        gc.collect()

    total_samples = len(index)
    sample_correct = sum(1 for entry in index if int(entry.get("label", -1)) == 1)
    sample_incorrect = sum(1 for entry in index if int(entry.get("label", -1)) == 0)
    sample_pending = sum(1 for entry in index if int(entry.get("label", -1)) == -1)
    total_labeled = sample_correct + sample_incorrect

    total_claims = sum(sum(counter.values()) for counter in type_label_counter.values())
    claims_correct = sum(counter.get("1", 0) for counter in type_label_counter.values())
    claims_incorrect = sum(counter.get("0", 0) for counter in type_label_counter.values())
    claims_pending = sum(counter.get("-1", 0) for counter in type_label_counter.values())
    no_usable_verified = sum(
        1
        for entry in index
        if int(entry.get("claims_total", 0) or 0) <= 0
        or int(entry.get("claims_pending", 0) or 0) > 0
    )

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

    claim_type_label_ratio = {}
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
        claim_type_label_ratio[ctype] = ordered

    total_char = sum(ts["char_total"] for ts in type_stats.values())
    avg_claim_char_length = round(total_char / max(total_claims, 1), 1)

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

    stats = {
        "total_samples": total_samples,
        "chunk_sample_counts": chunk_sample_counts,
        "sample_correct": sample_correct,
        "sample_incorrect": sample_incorrect,
        "sample_pending": sample_pending,
        "correct_rate": round(sample_correct / total_labeled, 4) if total_labeled > 0 else 0.0,
        "incorrect_rate": round(sample_incorrect / total_labeled, 4) if total_labeled > 0 else 0.0,
        "total_claims": total_claims,
        "claims_correct": claims_correct,
        "claims_incorrect": claims_incorrect,
        "claims_pending": claims_pending,
        "total_claim_pending": total_claim_pending,
        "avg_claim_char_length": avg_claim_char_length,
        "claim_type_stats": claim_type_stats,
        "claim_type_label_ratio": claim_type_label_ratio,
        "per_nhop_stats": per_nhop_stats,
        "no_fully_verified_claim_samples": no_usable_verified,
        "no_usable_verified_claim_samples": no_usable_verified,
    }
    return index, stats


def main():
    configure_logging(force=True)
    parser = argparse.ArgumentParser(
        description="Remove samples with pending claims from cached chunks"
    )
    parser.add_argument(
        "--cache_dir",
        required=True,
        help="Cache directory (e.g., artifacts/cached_features/StepGame/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--split",
        default="train,validation,test",
        help="Splits to process (CSV, default: train,validation,test)",
    )
    args = parser.parse_args()
    workflow = WorkflowLogger(log, "cleanup")
    
    cache_dir = args.cache_dir
    if not os.path.isdir(cache_dir):
        log.error("Cache directory not found: %s", cache_dir)
        sys.exit(1)
    
    splits = args.split.split(",")
    total_stats = {
        "total_samples_before": 0,
        "total_samples_after": 0,
        "removed_samples": 0,
    }
    
    workflow.header("Cleanup Pending Claims", cache_dir=cache_dir, splits=splits)
    
    for split in splits:
        split = split.strip()
        workflow.stage_start("split", split=split)
        
        stats = cleanup_split(cache_dir, split)
        
        total_stats["total_samples_before"] += stats["total_samples_before"]
        total_stats["total_samples_after"] += stats["total_samples_after"]
        total_stats["removed_samples"] += stats["removed_samples"]
        
        workflow.stage_end(
            "split",
            split=split,
            samples_before=stats["total_samples_before"],
            samples_after=stats["total_samples_after"],
            removed_samples=stats["removed_samples"],
        )
        if stats["chunks_modified"]:
            workflow.event("split.modified_chunks", split=split, chunks=stats["chunks_modified"])
    
    workflow.stage_end(
        "run",
        total_samples_before=total_stats["total_samples_before"],
        total_samples_after=total_stats["total_samples_after"],
        total_removed=total_stats["removed_samples"],
    )
    
    if total_stats["removed_samples"] > 0:
        workflow.warning(
            "run.cleaned",
            removed_samples=total_stats["removed_samples"],
            next_step="training can proceed with clean data",
        )
    else:
        workflow.event("run.cleaned", removed_samples=0)


if __name__ == "__main__":
    main()
