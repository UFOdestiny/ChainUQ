"""Aggregate measured compute from the stage statistics files (no new GPU work).

Per backbone and stage: total seconds and tokens plus per-response averages over the main run (HotpotQA
train and the validation/test splits of every dataset; test2 and the adaptation pools are not counted).
Generation, judge, and feature statistics are written by the stages themselves; training time comes from the
checkpoint metadata. Writes ``results/costs.json``.

Example:
    python -m chainuq.experiments.costs
"""

from __future__ import annotations

import fnmatch
import json
import os
from collections import defaultdict
from pathlib import Path

from chainuq.artifacts import results_dir, root
from chainuq.registry import SOURCE_DATASET


def _find(base: Path, pattern: str):
    """Files under ``base`` matching ``pattern``, following symlinked directories (artifacts span filesystems)."""
    for dirpath, _, files in os.walk(base, followlinks=True):
        for name in fnmatch.filter(files, pattern):
            yield Path(dirpath) / name


def _main_run(d: dict) -> bool:
    return d["split"] in ("validation", "test") or (d["split"] == "train" and d["dataset"] == SOURCE_DATASET)


def main():
    rows = defaultdict(lambda: defaultdict(float))
    A = root()
    for f in _find(A / "gen", "*.stats.json"):
        d = json.loads(f.read_text())
        if not _main_run(d):
            continue
        key = (d["backbone"], f"gen[{d['tag']}]" if d.get("tag") else "gen")
        rows[key]["seconds"] += d["gen_seconds"]
        rows[key]["samples"] += d["n_generated"]
        rows[key]["gen_tokens"] += d["gen_tokens"]
        rows[key]["prompt_tokens"] += d["prompt_tokens"]
    for f in _find(A / "judge", "*.stats.json"):
        d = json.loads(f.read_text())
        if not _main_run(d):
            continue
        key = (d["backbone"], f"judge[{d['judge']}/{d['task']}]")
        rows[key]["seconds"] += d["seconds"]
        rows[key]["samples"] += d["n"]
        rows[key]["gen_tokens"] += d["completion_tokens"]
        rows[key]["prompt_tokens"] += d["prompt_tokens"]
    for f in _find(A / "feat", "DONE.json"):
        d = json.loads(f.read_text())
        if not _main_run(d):
            continue
        key = (d["backbone"], "features")
        rows[key]["seconds"] += d["seconds"]
        rows[key]["samples"] += d["n_new"]
        rows[key]["prompt_tokens"] += d["tokens_forwarded"]
    for f in _find(A / "ckpt", "meta.json"):
        d = json.loads(f.read_text())
        method, backbone = f.relative_to(A / "ckpt").parts[:2]
        key = (backbone, f"train[{method}]")
        rows[key]["seconds"] += float(d.get("train_seconds", 0.0))
        rows[key]["runs"] += 1
        rows[key]["n_params"] = float(d.get("n_params", 0))

    out = []
    for (bb, stage), v in sorted(rows.items()):
        rec = {"backbone": bb, "stage": stage, **{k: round(x, 2) for k, x in v.items()}}
        if v.get("samples"):
            rec["sec_per_sample"] = round(v["seconds"] / v["samples"], 5)
            rec["tokens_per_sample"] = round((v.get("gen_tokens", 0) + v.get("prompt_tokens", 0)) / v["samples"], 1)
        out.append(rec)
    results_dir().mkdir(parents=True, exist_ok=True)
    (results_dir() / "costs.json").write_text(json.dumps(out, indent=2))
    for r in out:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
