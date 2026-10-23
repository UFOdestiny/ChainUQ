#!/usr/bin/env python3
"""Download the four benchmarks and freeze the train/validation/test splits.

Split protocol (seed 2026, deterministic):
  HotpotQA (distractor): official dev (7,405) halved into validation/test;
      train = 30k subsample stratified by level x type (full train kept as
      ``train_full``).
  MuSiQue (answerable): official dev (2,417) halved into validation/test;
      official train kept as ``train`` (not used for head training).
  StrategyQA (ChilleD/StrategyQA): official train (1,603) split 90/10 into
      train/validation; official test (687) kept as test.
  bAbI (Muennighoff/babi): official train (18,013) / validation (1,987);
      test = 250 per task (5,000; full 20,000 test kept as ``test_full``).

Every split also gets a ``split_uid`` column so later stages can join on it.

Usage:
    python scripts/prepare_datasets.py --out_root $DATASETS_ROOT [--only hotpotqa,babi]
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from datasets import Dataset, DatasetDict, load_dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SEED = 2026
HOTPOT_TRAIN_SIZE = 30000
BABI_TEST_PER_TASK = 250


def _halve(ds: Dataset, seed: int = SEED):
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    half = len(idx) // 2
    return ds.select(sorted(idx[:half])), ds.select(sorted(idx[half:]))


def _split_frac(ds: Dataset, frac: float, seed: int = SEED):
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    cut = int(round(len(idx) * frac))
    return ds.select(sorted(idx[:cut])), ds.select(sorted(idx[cut:]))


def _stratified(ds: Dataset, key_fn, n_total: int = None, per_group: int = None, seed: int = SEED):
    groups = defaultdict(list)
    for i, row in enumerate(ds):
        groups[key_fn(row)].append(i)
    rng = random.Random(seed)
    picked = []
    if per_group is not None:
        for key in sorted(groups):
            members = groups[key][:]
            rng.shuffle(members)
            picked.extend(members[:per_group])
    else:
        total = len(ds)
        for key in sorted(groups):
            members = groups[key][:]
            rng.shuffle(members)
            quota = int(round(n_total * len(members) / total))
            picked.extend(members[:quota])
        rng.shuffle(picked)
        picked = picked[:n_total]
    return ds.select(sorted(picked))


def _add_uid(ds: Dataset, prefix: str, id_field: str = None) -> Dataset:
    def _fn(row, i):
        base = str(row[id_field]) if id_field and row.get(id_field) not in (None, "") else str(i)
        return {"split_uid": f"{prefix}:{base}"}
    return ds.map(_fn, with_indices=True)


def prep_hotpotqa(out_root: Path):
    raw = load_dataset("hotpotqa/hotpot_qa", "distractor")
    val, test = _halve(raw["validation"])
    train = _stratified(raw["train"], lambda r: (r["level"], r["type"]), n_total=HOTPOT_TRAIN_SIZE)
    dd = DatasetDict({
        "train": _add_uid(train, "hotpotqa", "id"),
        "validation": _add_uid(val, "hotpotqa", "id"),
        "test": _add_uid(test, "hotpotqa", "id"),
        "train_full": _add_uid(raw["train"], "hotpotqa", "id"),
    })
    dd.save_to_disk(str(out_root / "hotpot_qa" / "hf_dataset"))
    return dd


def prep_musique(out_root: Path):
    raw = load_dataset("dgslibisey/MuSiQue")
    val, test = _halve(raw["validation"])
    dd = DatasetDict({
        "train": _add_uid(raw["train"], "musique", "id"),
        "validation": _add_uid(val, "musique", "id"),
        "test": _add_uid(test, "musique", "id"),
    })
    dd.save_to_disk(str(out_root / "MuSiQue" / "hf_dataset"))
    return dd


def prep_strategyqa(out_root: Path):
    raw = load_dataset("ChilleD/StrategyQA")
    val, train = _split_frac(raw["train"], 0.1)
    dd = DatasetDict({
        "train": _add_uid(train, "strategyqa", "qid"),
        "validation": _add_uid(val, "strategyqa", "qid"),
        "test": _add_uid(raw["test"], "strategyqa", "qid"),
    })
    dd.save_to_disk(str(out_root / "StrategyQA" / "hf_dataset"))
    return dd


def prep_babi(out_root: Path):
    raw = load_dataset("Muennighoff/babi")
    test_full = _add_uid(raw["test"], "babi-test")
    test = _stratified(test_full, lambda r: int(r["task"]), per_group=BABI_TEST_PER_TASK)
    dd = DatasetDict({
        "train": _add_uid(raw["train"], "babi-train"),
        "validation": _add_uid(raw["validation"], "babi-val"),
        "test": test,
        "test_full": test_full,
    })
    dd.save_to_disk(str(out_root / "babi" / "hf_dataset"))
    return dd


def prep_gsm8k(out_root: Path):
    raw = load_dataset("openai/gsm8k", "main")
    val, _ = _split_frac(raw["train"], 500 / len(raw["train"]))
    dd = DatasetDict({
        "validation": _add_uid(val, "gsm8k-train"),
        "test": _add_uid(raw["test"], "gsm8k-test"),
    })
    dd.save_to_disk(str(out_root / "gsm8k" / "hf_dataset"))
    return dd


MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus")


def prep_math500(out_root: Path):
    """Test = MATH-500; validation = 500 problems from the rest of the MATH *test* split.

    The validation problems are drawn from MATH test (not train) because backbones have seen
    MATH train: Llama-3.1-8B scored 0.63 on Level-5 train problems vs 0.24 on Level-5 MATH-500.
    """
    from datasets import concatenate_datasets
    from data.math_reasoning import last_boxed

    test = load_dataset("HuggingFaceH4/MATH-500")["test"]
    held = {p.strip() for p in test["problem"]}
    pool = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s)["test"] for s in MATH_SUBJECTS])
    pool = pool.map(lambda r: {"answer": last_boxed(r["solution"]), "subject": r["type"], "unique_id": ""})
    pool = pool.filter(lambda r: bool(r["answer"]) and r["problem"].strip() not in held)
    val = _stratified(pool, lambda r: (r["subject"], r["level"]), n_total=500)
    cols = ["problem", "solution", "answer", "subject", "level", "unique_id"]
    dd = DatasetDict({
        "validation": _add_uid(val.select_columns(cols), "math-test"),
        "test": _add_uid(test.select_columns(cols), "math500", "unique_id"),
    })
    dd.save_to_disk(str(out_root / "math500" / "hf_dataset"))
    return dd


PREPARERS = {
    "hotpotqa": prep_hotpotqa,
    "musique": prep_musique,
    "strategyqa": prep_strategyqa,
    "babi": prep_babi,
    "gsm8k": prep_gsm8k,
    "math500": prep_math500,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--only", default=",".join(PREPARERS))
    args = parser.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    summary_path = out_root / "split_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in [x.strip() for x in args.only.split(",") if x.strip()]:
        dd = PREPARERS[name](out_root)
        summary[name] = {split: len(ds) for split, ds in dd.items()}
        print(name, summary[name], flush=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
