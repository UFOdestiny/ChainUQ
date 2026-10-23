#!/usr/bin/env python3
"""Add a fresh held-out ``test2`` split (built after the original test split had been evaluated twice).

Sources are data never used anywhere in the project (seed 2027, disjoint from every existing split):
  HotpotQA  1500 ``hard``-level questions from official train not in the 30k training subsample
  MuSiQue   1200 questions from the official (answerable) train split
  bAbI      2000 = 100 per task from the official test split, outside the 5000 already used
  GSM8K     1000 questions from official train, outside the 500 used for validation
  MATH      1000 problems from MATH test, outside MATH-500 and the 500 used for validation
StrategyQA has no unused data and gets no test2.

Example:
    python scripts/add_test2.py --root $DATASETS_ROOT
"""

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

from datasets import DatasetDict, concatenate_datasets, load_dataset, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SEED = 2027


def sample(ds, n, key=None, seed=SEED):
    idx = list(range(len(ds)))
    rng = random.Random(seed)
    if key is None:
        rng.shuffle(idx)
        return ds.select(sorted(idx[:n]))
    groups = defaultdict(list)
    for i, r in enumerate(ds):
        groups[key(r)].append(i)
    per = max(1, n // len(groups))
    out = []
    for k in sorted(groups):
        g = groups[k][:]
        rng.shuffle(g)
        out.extend(g[:per])
    return ds.select(sorted(out[:n]))


def add_uid(ds, prefix, field=None):
    return ds.map(lambda r, i: {"split_uid": f"{prefix}:{r[field] if field else i}"}, with_indices=True)


def save(root, name, dd):
    """Write next to the old copy, then swap by renaming (the old copy is memory-mapped by this process,
    so it cannot be deleted on this network filesystem; it is kept as hf_dataset.pre_test2)."""
    path = root / name / "hf_dataset"
    tmp = root / name / "hf_dataset.new"
    dd.save_to_disk(str(tmp))
    path.rename(root / name / "hf_dataset.pre_test2")
    tmp.rename(path)


def has_test2(root, name):
    return "test2" in load_from_disk(str(root / name / "hf_dataset"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    root = Path(p.parse_args().root)
    summary = {}

    if has_test2(root, "hotpot_qa"):
        print("hotpotqa: test2 exists")
    else:
        build_hotpot(root, summary)
    if not has_test2(root, "MuSiQue"):
        build_musique(root, summary)
    if not has_test2(root, "babi"):
        build_babi(root, summary)
    if not has_test2(root, "gsm8k"):
        build_gsm8k(root, summary)
    if not has_test2(root, "math500"):
        build_math(root, summary)
    print(summary)


def build_hotpot(root, summary):
    dd = load_from_disk(str(root / "hotpot_qa" / "hf_dataset"))
    used = set(dd["train"]["id"]) | set(dd["validation"]["id"]) | set(dd["test"]["id"])
    pool = dd["train_full"].filter(lambda r: r["level"] == "hard" and r["id"] not in used)
    t2 = add_uid(sample(pool, 1500).remove_columns(["split_uid"]), "hotpotqa-t2", "id")
    dd = DatasetDict({k: v for k, v in dd.items()} | {"test2": t2})
    save(root, "hotpot_qa", dd.flatten_indices())
    summary["hotpotqa"] = len(t2)


def build_musique(root, summary):
    dd = load_from_disk(str(root / "MuSiQue" / "hf_dataset"))
    t2 = add_uid(sample(dd["train"], 1200).remove_columns(["split_uid"]), "musique-t2", "id")
    dd = DatasetDict({k: v for k, v in dd.items()} | {"test2": t2})
    save(root, "MuSiQue", dd.flatten_indices())
    summary["musique"] = len(t2)


def build_babi(root, summary):
    dd = load_from_disk(str(root / "babi" / "hf_dataset"))
    used = set(dd["test"]["split_uid"])
    pool = dd["test_full"].filter(lambda r: r["split_uid"] not in used)
    t2 = sample(pool, 2000, key=lambda r: int(r["task"]))
    t2 = t2.map(lambda r: {"split_uid": r["split_uid"].replace("babi-test:", "babi-t2:")})
    dd = DatasetDict({k: v for k, v in dd.items()} | {"test2": t2})
    save(root, "babi", dd.flatten_indices())
    summary["babi"] = len(t2)


def build_gsm8k(root, summary):
    dd = load_from_disk(str(root / "gsm8k" / "hf_dataset"))
    raw = load_dataset("openai/gsm8k", "main")["train"]
    used_q = set(dd["validation"]["question"])
    pool = raw.filter(lambda r: r["question"] not in used_q)
    t2 = add_uid(sample(pool, 1000), "gsm8k-t2")
    dd = DatasetDict({k: v for k, v in dd.items()} | {"test2": t2})
    save(root, "gsm8k", dd.flatten_indices())
    summary["gsm8k"] = len(t2)


def build_math(root, summary):
    from data.math_reasoning import last_boxed
    from scripts.prepare_datasets import MATH_SUBJECTS
    dd = load_from_disk(str(root / "math500" / "hf_dataset"))
    used_p = {x.strip() for x in dd["test"]["problem"]} | {x.strip() for x in dd["validation"]["problem"]}
    pool = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s)["test"] for s in MATH_SUBJECTS])
    pool = pool.map(lambda r: {"answer": last_boxed(r["solution"]), "subject": r["type"], "unique_id": ""})
    pool = pool.filter(lambda r: bool(r["answer"]) and r["problem"].strip() not in used_p)
    t2 = sample(pool, 1000, key=lambda r: (r["subject"], r["level"]))
    cols = ["problem", "solution", "answer", "subject", "level", "unique_id"]
    t2 = add_uid(t2.select_columns(cols), "math-t2")
    dd = DatasetDict({k: v for k, v in dd.items()} | {"test2": t2})
    save(root, "math500", dd.flatten_indices())
    summary["math500"] = len(t2)


if __name__ == "__main__":
    main()
