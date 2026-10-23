"""Artifact layout and JSONL helpers.

Every stage reads and writes through these functions, so file names are defined
in exactly one place::

    gen/<backbone>/<dataset>/<split>.jsonl                      greedy generations (+ .samples.jsonl, .stats.json)
    judge/<judge>/<task>/<backbone>/<dataset>/<split>.jsonl     judge annotations
    feat/<backbone>/<dataset>/<split>/shard_*.pt                cached backbone features (+ DONE.json)
    cotuq/<backbone>/<dataset>/<split>.jsonl                    CoT-UQ keywords
    ckpt/<method>/<backbone>/seed<k>/{model.pt,meta.json}       trained heads
    preds/<method>/<backbone>/seed<k>/<dataset>/<split>.jsonl   per-response scores of every method
    results/                                                    evaluation outputs
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List

from chainuq import config

_root = config.ARTIFACTS


def root() -> Path:
    return _root


@contextmanager
def use_root(path: Path):
    """Temporarily read and write another artifact tree (e.g. a separately built unlabeled pool)."""
    global _root
    previous, _root = _root, Path(path)
    try:
        yield
    finally:
        _root = previous


def gen_path(backbone: str, dataset: str, split: str, tag: str = "") -> Path:
    return _root / "gen" / backbone / dataset / (f"{split}.{tag}.jsonl" if tag else f"{split}.jsonl")


def judge_path(judge: str, task: str, backbone: str, dataset: str, split: str) -> Path:
    return _root / "judge" / judge / task / backbone / dataset / f"{split}.jsonl"


def feat_dir(backbone: str, dataset: str, split: str) -> Path:
    return _root / "feat" / backbone / dataset / split


def cotuq_path(backbone: str, dataset: str, split: str) -> Path:
    return _root / "cotuq" / backbone / dataset / f"{split}.jsonl"


def ckpt_dir(method: str, backbone: str, seed: int) -> Path:
    return _root / "ckpt" / method / backbone / f"seed{seed}"


def pred_path(method: str, backbone: str, seed: int, dataset: str, split: str) -> Path:
    return _root / "preds" / method / backbone / f"seed{seed}" / dataset / f"{split}.jsonl"


def results_dir() -> Path:
    return _root / "results"


def feature_info(backbone: str) -> dict:
    """Backbone dimensions recorded by the feature extractor (``hidden_size``, ``n_layers``, ...)."""
    from chainuq.registry import SOURCE_DATASET
    return json.loads((feat_dir(backbone, SOURCE_DATASET, "train") / "DONE.json").read_text())


def write_jsonl(path: Path, records: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def read_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def index_by_uid(records: Iterable[dict]) -> dict:
    return {r["uid"]: r for r in records}
