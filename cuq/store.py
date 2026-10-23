"""Assemble per-sample records from the gen / judge / feature artifacts.

The *evaluation set* of a (backbone, dataset, split, fmt) cell is fixed by the
main judge: a sample is kept when it parsed, has a response-level label, and
has a valid main-judge step verdict. Every method is scored on exactly this
uid set, so methods that do not use the judge are not evaluated on a
different population.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from cuq.io import index_by_uid, read_jsonl
from cuq.registry import DATASETS, MAIN_JUDGE, feat_dir, gen_path, judge_path


@dataclass
class Sample:
    uid: str
    y: int                      # response-level correctness (1 = correct)
    steps: list                 # [{"text", "tok": [s, e)}]
    conclusion: dict            # {"text", "tok": [s, e)}
    step_labels: Optional[list]  # main-judge Stage-2 labels (answer-blind)
    gen: dict                   # full generation record
    feat: Optional[dict] = None  # tensors from pipeline/features.py


def load_labels(backbone: str, dataset: str, split: str, fmt: str = "step",
                label_judge: str = MAIN_JUDGE) -> Dict[str, int]:
    gens = read_jsonl(gen_path(backbone, dataset, split, fmt))
    if DATASETS[dataset]["label"] == "em":
        return {g["uid"]: int(g["label_em"]) for g in gens if g.get("parse_ok") and g.get("label_em") in (0, 1)}
    path = judge_path(label_judge, "correct", backbone, dataset, split, fmt)
    return {r["uid"]: int(r["value"]) for r in read_jsonl(path) if r["value"] in (0, 1)}


def load_judge_task(judge: str, task: str, backbone: str, dataset: str, split: str, fmt: str = "step") -> Dict[str, dict]:
    path = judge_path(judge, task, backbone, dataset, split, fmt)
    if not Path(path).exists():
        return {}
    return index_by_uid(read_jsonl(path))


def load_features(backbone: str, dataset: str, split: str, fmt: str = "step",
                  uids: Optional[set] = None, keys: Optional[List[str]] = None, device=None) -> Dict[str, dict]:
    """Load feature shards; with ``device`` each shard is moved there before the next is read,
    so CPU memory never holds more than one shard (the Slurm job is capped at 120 GB)."""
    out = {}
    for shard in sorted(feat_dir(backbone, dataset, split, fmt).glob("shard_*.pt")):
        data = torch.load(shard, weights_only=False)
        for uid, rec in data.items():
            if uids is not None and uid not in uids:
                continue
            rec = {k: rec[k] for k in keys} if keys else rec
            if device is not None:
                rec = {k: v.to(device) for k, v in rec.items()}
            out[uid] = rec
        del data
    return out


def load_samples(backbone: str, dataset: str, split: str, fmt: str = "step",
                 with_features: bool = True, feature_keys: Optional[List[str]] = None,
                 require_steps: bool = True, device=None) -> List[Sample]:
    gens = {g["uid"]: g for g in read_jsonl(gen_path(backbone, dataset, split, fmt)) if g.get("parse_ok")}
    labels = load_labels(backbone, dataset, split, fmt)
    steps = load_judge_task(MAIN_JUDGE, "steps", backbone, dataset, split, fmt)
    uids = [u for u in gens if u in labels]
    if require_steps:
        uids = [u for u in uids if u in steps and isinstance(steps[u]["value"], list)]
    feats = load_features(backbone, dataset, split, fmt, set(uids), feature_keys, device) if with_features else {}
    if with_features:
        uids = [u for u in uids if u in feats]
    samples = []
    for u in uids:
        g = gens[u]
        samples.append(Sample(
            uid=u, y=labels[u], steps=g["steps"], conclusion=g["conclusion"],
            step_labels=steps[u]["value"] if u in steps else None,
            gen=g, feat=feats.get(u),
        ))
    return samples
