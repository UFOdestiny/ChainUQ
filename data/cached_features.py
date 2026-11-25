"""CachedFeatureDataset - loads pre-extracted Phase 1 features for training."""
import json
import os
import gc
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import torch
from torch.utils.data import Dataset
from utils.log import get_logger

log = get_logger(__name__)


class CachedFeatureDataset(Dataset):
    """Loads pre-extracted features from Phase 1 chunks for UQ head training.

    Chunk format: {cache_dir}/{split}/chunk_*.pt
    Optional reasoning sidecar: {cache_dir}/{split}/chunk_*_reasoning.pt
    Each chunk is a list of dicts with keys:
        features, attention_mask, label,
        n_hops, question, generated_text, dataset_name,
        plus split claim fields (reasoning_claims/conclusion_claim + *_verified).
    """

    def __init__(
        self,
        cache_dir: str,
        split: str = "train",
        n_hop_values: Optional[List[int]] = None,
        max_samples: int = 0,
        preload_all: bool = False,
        max_cached_chunks: int = 0,
        skip_pending: bool = True,
        skip_no_verified_claims: bool = True,
        mem_budget_gb: Optional[float] = None,
        load_reasoning_sidecar: bool = True,
    ):
        self.cache_dir = cache_dir
        self.split = split
        self.n_hop_values = n_hop_values
        self.max_samples = max_samples
        self.skip_pending = skip_pending
        self.skip_no_verified_claims = skip_no_verified_claims
        self.load_reasoning_sidecar = bool(load_reasoning_sidecar)
        if mem_budget_gb is None or mem_budget_gb <= 0:
            raise ValueError("CachedFeatureDataset requires a positive mem_budget_gb")
        self._mem_budget_gb = float(mem_budget_gb)
        self._manifest = {}
        self._claim_stats = None
        self._filter_stats = {
            "initial_samples": 0,
            "removed_n_hops": 0,
            "removed_pending": 0,
            "removed_no_usable_verified_claims": 0,
            "final_samples": 0,
            "skip_pending": bool(skip_pending),
            "skip_no_verified_claims": bool(skip_no_verified_claims),
        }

        # Load manifest
        split_dir = os.path.join(cache_dir, split)
        manifest_path = os.path.join(split_dir, "manifest.json")
        if os.path.isfile(manifest_path):
            with open(manifest_path) as f:
                self._manifest = json.load(f)

        # Load index from index.json (or rebuild from chunks)
        index_path = os.path.join(split_dir, "index.json")
        if os.path.isfile(index_path):
            with open(index_path) as f:
                self._index = json.load(f)
        else:
            self._index = self._build_index(split_dir)

        # Auto-detect max_cached_chunks based on chunk file sizes
        if max_cached_chunks <= 0:
            max_cached_chunks = self._auto_max_cached_chunks(split_dir, mem_budget_gb=self._mem_budget_gb)

        self._chunk_cache = OrderedDict()
        self._reasoning_cache = OrderedDict()
        self._max_cached = max_cached_chunks

        # Apply filters
        self._filter_stats["initial_samples"] = len(self._index)
        if self.n_hop_values:
            before = len(self._index)
            self._index = [e for e in self._index if e.get("n_hops") in self.n_hop_values]
            self._filter_stats["removed_n_hops"] = before - len(self._index)
        if self.skip_pending:
            before = len(self._index)
            self._index = [e for e in self._index if e.get("label", -1) != -1]
            self._filter_stats["removed_pending"] = before - len(self._index)
        if self.skip_no_verified_claims:
            before = len(self._index)
            self._index = [
                e for e in self._index
                if int(e.get("claims_total", 0) or 0) > 0
                and int(e.get("claims_pending", 0) or 0) == 0
            ]
            removed = before - len(self._index)
            self._filter_stats["removed_no_usable_verified_claims"] = removed
            if removed > 0:
                log.info(
                    "CachedFeatureDataset: dropped %d sample(s) with incomplete claim verification from %s/%s.",
                    removed,
                    cache_dir,
                    split,
                )
        if self.max_samples > 0:
            self._index = self._index[:self.max_samples]
        self._filter_stats["final_samples"] = len(self._index)

        log.info(
            "CachedFeatureDataset: %s/%s — %d samples (n_hop_values=%s)",
            cache_dir, split, len(self._index),
            self.n_hop_values,
        )

        if preload_all:
            self._preload_all(split_dir)

    def _auto_max_cached_chunks(self, split_dir: str, mem_budget_gb: float = 50.0) -> int:
        """Choose max_cached_chunks so total cache stays within mem_budget_gb."""
        chunk_files = sorted(
            f
            for f in os.listdir(split_dir)
            if f.startswith("chunk_") and f.endswith(".pt") and not f.endswith("_reasoning.pt")
        )
        if not chunk_files:
            return 2
        sample_path = os.path.join(split_dir, chunk_files[0])
        chunk_bytes = os.path.getsize(sample_path)
        chunk_gb = chunk_bytes / (1024 ** 3)
        if chunk_gb <= 0:
            return 2
        n = max(1, int(mem_budget_gb / chunk_gb))
        log.info(
            "Auto max_cached_chunks=%d (chunk ~%.1f GB, budget %.0f GB)",
            n, chunk_gb, mem_budget_gb,
        )
        return n

    def label_distribution(self):
        """Return (n_correct, n_hallucination) from index metadata (no disk I/O)."""
        n_correct = sum(1 for e in self._index if e.get("label", 0) == 1)
        n_hall = len(self._index) - n_correct
        return n_correct, n_hall

    def filter_stats(self) -> dict:
        return dict(self._filter_stats)

    def clear_cache(self) -> None:
        """Evict all cached chunks to free RAM (keeps the index/metadata)."""
        self._chunk_cache.clear()
        self._reasoning_cache.clear()

    def claim_distribution(self) -> dict:
        """Return claim counts from the filtered dataset, not the raw manifest.

        Computed from the in-memory index (no chunk loading required).
        """
        if self._claim_stats is not None:
            return dict(self._claim_stats)

        total_claims = sum(int(e.get("claims_total", 0) or 0) for e in self._index)
        pending_claims = sum(int(e.get("claims_pending", 0) or 0) for e in self._index)
        correct_claims = sum(int(e.get("claims_correct", 0) or 0) for e in self._index)
        incorrect_claims = sum(int(e.get("claims_incorrect", 0) or 0) for e in self._index)
        reasoning_claims = sum(int(e.get("n_reasoning_claims", 0) or 0) for e in self._index)
        conclusion_claims = sum(int(e.get("n_conclusion_claims", 0) or 0) for e in self._index)
        reasoning_correct = sum(int(e.get("n_reasoning_correct", 0) or 0) for e in self._index)
        conclusion_correct = sum(int(e.get("n_conclusion_correct", 0) or 0) for e in self._index)

        self._claim_stats = {
            "total_claims": total_claims,
            "verified_claims": total_claims - pending_claims,
            "pending_claims": pending_claims,
            "correct_claims": correct_claims,
            "incorrect_claims": incorrect_claims,
            "reasoning_claims": reasoning_claims,
            "reasoning_correct": reasoning_correct,
            "conclusion_claims": conclusion_claims,
            "conclusion_correct": conclusion_correct,
        }
        return dict(self._claim_stats)

    def _build_index(self, split_dir: str) -> list:
        """Build index by scanning chunk files."""
        index = []
        chunk_files = sorted(
            f
            for f in os.listdir(split_dir)
            if f.startswith("chunk_") and f.endswith(".pt") and not f.endswith("_reasoning.pt")
        )
        for chunk_file in chunk_files:
            chunk_path = os.path.join(split_dir, chunk_file)
            try:
                chunk_data = torch.load(chunk_path, map_location="cpu", weights_only=False)
                if not isinstance(chunk_data, list) or len(chunk_data) == 0:
                    raise ValueError("empty or invalid chunk")
            except Exception as e:
                log.warning("Chunk %s corrupt (%s), deleting.", chunk_file, e)
                os.remove(chunk_path)
                continue
            for local_idx, sample in enumerate(chunk_data):
                n_reasoning = int(sample.get("n_reasoning_claims", 0) or 0)
                n_conclusion = int(sample.get("n_conclusion_claims", 0) or 0)
                n_reasoning_correct = int(sample.get("n_reasoning_correct", 0) or 0)
                n_conclusion_correct = int(sample.get("n_conclusion_correct", 0) or 0)
                claims_correct = int(sample.get("claims_correct", 0) or 0)
                claims_incorrect = int(sample.get("claims_incorrect", 0) or 0)
                claims_pending = int(sample.get("claims_pending", 0) or 0)
                claims_total = int(sample.get("claims_total", 0) or 0)
                index.append({
                    "chunk_file": chunk_file,
                    "local_idx": local_idx,
                    "sample_id": int(sample.get("sample_id", -1)),
                    "sample_uid": str(sample.get("sample_uid", "")),
                    "label": sample.get("label", -1),
                    "n_hops": sample.get("n_hops", 0),
                    "n_reasoning_claims": n_reasoning,
                    "n_conclusion_claims": n_conclusion,
                    "n_reasoning_correct": n_reasoning_correct,
                    "n_conclusion_correct": n_conclusion_correct,
                    "claims_correct": claims_correct,
                    "claims_incorrect": claims_incorrect,
                    "dataset_name": sample.get("dataset_name", ""),
                    "claims_total": claims_total,
                    "claims_pending": claims_pending,
                })
            del chunk_data
            gc.collect()
        return index

    def _preload_all(self, split_dir: str):
        """Preload all chunks into memory."""
        chunk_files = set(e["chunk_file"] for e in self._index)

        def _load(cf):
            return cf, torch.load(os.path.join(split_dir, cf), map_location="cpu", weights_only=False)

        with ThreadPoolExecutor(max_workers=4) as pool:
            for cf, data in pool.map(_load, chunk_files):
                self._chunk_cache[cf] = data

    def _load_chunk(self, chunk_file: str):
        """Load a chunk with LRU caching."""
        if chunk_file in self._chunk_cache:
            self._chunk_cache.move_to_end(chunk_file)
            return self._chunk_cache[chunk_file]

        # Evict before loading to avoid transient memory spikes where
        # old+new chunks coexist and exceed the configured cache budget.
        while self._max_cached > 0 and len(self._chunk_cache) >= self._max_cached:
            self._chunk_cache.popitem(last=False)

        chunk_path = os.path.join(self.cache_dir, self.split, chunk_file)
        data = torch.load(chunk_path, map_location="cpu", weights_only=False)

        if self._max_cached > 0:
            self._chunk_cache[chunk_file] = data
        return data

    def _load_reasoning_chunk(self, chunk_file: str):
        """Load sidecar reasoning chunk with LRU caching."""
        sidecar_file = f"{os.path.splitext(chunk_file)[0]}_reasoning.pt"
        if sidecar_file in self._reasoning_cache:
            self._reasoning_cache.move_to_end(sidecar_file)
            return self._reasoning_cache[sidecar_file]

        while self._max_cached > 0 and len(self._reasoning_cache) >= self._max_cached:
            self._reasoning_cache.popitem(last=False)

        sidecar_path = os.path.join(self.cache_dir, self.split, sidecar_file)
        if not os.path.isfile(sidecar_path):
            return None
        data = torch.load(sidecar_path, map_location="cpu", weights_only=False)
        if self._max_cached > 0:
            self._reasoning_cache[sidecar_file] = data
        return data

    def load_raw(self, idx: int) -> dict:
        """Load the original sample dict for diagnostics or cache inspection."""
        entry = self._index[idx]
        chunk_data = self._load_chunk(entry["chunk_file"])
        if not self.load_reasoning_sidecar:
            return chunk_data[entry["local_idx"]]

        sample = dict(chunk_data[entry["local_idx"]])
        sidecar_data = self._load_reasoning_chunk(entry["chunk_file"])
        if (
            isinstance(sidecar_data, list)
            and entry["local_idx"] < len(sidecar_data)
            and isinstance(sidecar_data[entry["local_idx"]], dict)
        ):
            sidecar_entry = sidecar_data[entry["local_idx"]]
            for key, value in sidecar_entry.items():
                if key not in sample:
                    sample[key] = value
        return sample

    def __len__(self):
        return len(self._index)

    def __getitem__(self, idx) -> dict:
        raw = self.load_raw(idx)

        features = raw["features"]
        if features.dtype == torch.bfloat16:
            features = features.float()

        attention_mask = raw.get("attention_mask")
        if attention_mask is None:
            attention_mask = torch.ones(features.shape[0], dtype=torch.long)

        label = float(raw.get("label", 0))

        item = {
            "features": features,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.float32),
        }

        # Include split claim-level data if available.
        if "conclusion_claim" in raw or "reasoning_claims" in raw:
            item["conclusion_claim"] = raw.get("conclusion_claim")
            item["conclusion_verified"] = raw.get("conclusion_verified", -1)
            item["reasoning_claims"] = raw.get("reasoning_claims", []) or []
            item["reasoning_verified"] = raw.get("reasoning_verified", []) or []
        if "token_probs" in raw:
            item["token_probs"] = raw.get("token_probs")
        if "log_likelihoods" in raw:
            item["log_likelihoods"] = raw.get("log_likelihoods")
        reasoning_features = raw.get("reasoning_features") if self.load_reasoning_sidecar else None
        if isinstance(reasoning_features, torch.Tensor) and reasoning_features.numel() > 0:
            rf = reasoning_features.float()
            if rf.ndim == 1:
                rf = rf.unsqueeze(0)
            flat = rf.reshape(rf.shape[0], -1)
            token_l2 = flat.norm(dim=1)
            abs_vals = flat.abs().reshape(-1)
            active_tokens = int(flat.shape[0])
            reasoning_attention = raw.get("reasoning_attention_mask")
            if isinstance(reasoning_attention, torch.Tensor):
                active_tokens = int((reasoning_attention > 0).sum().item())
            item["reasoning_feature_stats"] = {
                "tokens": int(flat.shape[0]),
                "active_tokens": int(active_tokens),
                "mean_l2": float(token_l2.mean().item()),
                "std_l2": float(token_l2.std(unbiased=False).item()) if token_l2.numel() > 1 else 0.0,
                "mean_abs": float(abs_vals.mean().item()),
                "std_abs": float(abs_vals.std(unbiased=False).item()) if abs_vals.numel() > 1 else 0.0,
            }
            item["reasoning_feature_mean"] = flat.mean(dim=0).to(dtype=torch.float32).cpu()

        # Pass through metadata needed for stratified evaluation.
        # n_hops comes from the pre-computed index to avoid extra claim scans.
        entry = self._index[idx]
        item["n_hops"] = entry.get("n_hops", 0)
        sample_id = entry.get("sample_id", raw.get("sample_id", -1))
        item["sample_id"] = int(sample_id if sample_id is not None else -1)
        sample_uid = entry.get("sample_uid", raw.get("sample_uid", ""))
        item["sample_uid"] = str(sample_uid if sample_uid is not None else "")
        item["question"] = raw.get("question", "")
        item["generated_text"] = raw.get("generated_text", "")
        item["dataset_name"] = raw.get("dataset_name", "")

        return item
