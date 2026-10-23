"""Assemble per-response records from the generation, judge, and feature artifacts.

The evaluation set of a (backbone, dataset, split) cell is fixed by the main
judge: a response is kept when it parsed, has a correctness label, and has a
valid answer-blind step judgment. Every method is scored on exactly this set,
so methods that do not use the judge are not evaluated on another population.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch

from chainuq import registry
from chainuq.artifacts import feat_dir, gen_path, index_by_uid, judge_path, read_jsonl


@dataclass
class Sample:
    uid: str
    y: int                       # response correctness (1 = correct; -1 = unlabeled)
    steps: list                  # [{"text", "tok": [s, e)}] in generated-token coordinates
    conclusion: dict             # {"text", "tok": [s, e)}
    step_labels: Optional[list]  # answer-blind step judgments of the main judge
    gen: dict                    # full generation record
    feat: Optional[dict] = None  # cached feature tensors (pipeline/features.py)


def load_labels(backbone: str, dataset: str, split: str, judge: Optional[str] = None) -> Dict[str, int]:
    """Correctness labels: answer matching for exact-match datasets, else the ``correct`` task of the judge."""
    if registry.DATASETS[dataset]["label"] == "em":
        gens = read_jsonl(gen_path(backbone, dataset, split))
        return {g["uid"]: int(g["label_em"]) for g in gens if g.get("parse_ok") and g.get("label_em") in (0, 1)}
    path = judge_path(judge or registry.MAIN_JUDGE, "correct", backbone, dataset, split)
    return {r["uid"]: int(r["value"]) for r in read_jsonl(path) if r["value"] in (0, 1)}


def load_judge_task(judge: str, task: str, backbone: str, dataset: str, split: str) -> Dict[str, dict]:
    path = judge_path(judge, task, backbone, dataset, split)
    return index_by_uid(read_jsonl(path)) if Path(path).exists() else {}


def load_step_labels(judge: str, backbone: str, dataset: str, split: str, task: str = "steps") -> Dict[str, list]:
    """uid -> list of 0/1 step judgments (responses whose judgment failed to parse are absent)."""
    return {u: r["value"] for u, r in load_judge_task(judge, task, backbone, dataset, split).items()
            if isinstance(r["value"], list)}


def verdict_value(rec: dict) -> Optional[float]:
    """Answer-blind chain verdict of the judge: the verbalized probability, else P(true) from token log-probs."""
    pv, pt = rec.get("p_verbal"), rec.get("p_true")
    if pv is None and pt is None:
        return None
    return pv if pv is not None else pt


def load_verdicts(backbone: str, dataset: str, split: str, judge: Optional[str] = None) -> Dict[str, float]:
    """uid -> chain verdict in [0, 1]; the training target of the verdict output (privileged, training only)."""
    recs = load_judge_task(judge or registry.MAIN_JUDGE, "final_blind", backbone, dataset, split)
    return {u: float(min(1.0, max(0.0, v))) for u, r in recs.items() if (v := verdict_value(r)) is not None}


def load_features(backbone: str, dataset: str, split: str, uids: Optional[set] = None,
                  keys: Optional[List[str]] = None, device=None) -> Dict[str, dict]:
    """Load feature shards; with ``device`` each shard is moved there before the next one is read,
    so host memory never holds more than one shard."""
    out = {}
    for shard in sorted(feat_dir(backbone, dataset, split).glob("shard_*.pt")):
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


def load_samples(backbone: str, dataset: str, split: str, with_features: bool = True,
                 feature_keys: Optional[List[str]] = None, require_steps: bool = True, device=None,
                 unlabeled: bool = False, max_n: int = 0, judge: Optional[str] = None) -> List[Sample]:
    """Responses of one cell. ``unlabeled`` keeps responses without a correctness label (``y = -1``), e.g. an
    unlabeled target pool; ``judge`` selects whose step judgments are attached (default: the main judge)."""
    gens = {g["uid"]: g for g in read_jsonl(gen_path(backbone, dataset, split)) if g.get("parse_ok")}
    labels = {u: -1 for u in gens} if unlabeled else load_labels(backbone, dataset, split)
    steps = load_judge_task(judge or registry.MAIN_JUDGE, "steps", backbone, dataset, split)
    uids = [u for u in gens if u in labels]
    if require_steps:
        uids = [u for u in uids if u in steps and isinstance(steps[u]["value"], list)]
    if max_n:
        uids = uids[:max_n]
    feats = load_features(backbone, dataset, split, set(uids), feature_keys, device) if with_features else {}
    if with_features:
        uids = [u for u in uids if u in feats]
    return [Sample(uid=u, y=labels[u], steps=gens[u]["steps"], conclusion=gens[u]["conclusion"],
                   step_labels=steps[u]["value"] if u in steps else None, gen=gens[u], feat=feats.get(u))
            for u in uids]
