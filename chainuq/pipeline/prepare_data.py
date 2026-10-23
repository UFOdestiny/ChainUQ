"""Stage 0: download the benchmarks and freeze every split (deterministic).

Splits (seed 2026 unless noted):
  HotpotQA (distractor)  official dev (7,405) halved into validation/test; train = 30k subsample stratified by
                         level x type (the full train split is kept as ``train_full``).
  MuSiQue (answerable)   official dev (2,417) halved into validation/test; official train kept as ``train``.
  StrategyQA             official train (1,603) split 90/10 into train/validation; official test kept as test.
  bAbI                   official train / validation; test = 250 questions per task (5,000; the full test split
                         is kept as ``test_full``).
  GSM8K                  validation = 500 questions from official train; test = official test.
  MATH                   test = MATH-500; validation = 500 problems from the rest of MATH test, stratified by
                         subject x level (backbones have seen MATH train).
Held-out ``test2`` split (seed 2027, disjoint from every other split; StrategyQA has no unused data):
  HotpotQA 1,500 ``hard`` questions from official train outside the training subsample; MuSiQue 1,200 questions
  from official train; bAbI 100 per task from the unused test questions; GSM8K 1,000 from official train outside
  validation; MATH 1,000 problems from MATH test outside MATH-500 and validation.

``--pools`` also writes the unlabeled GSM8K/MATH question pools of label-free adaptation to
``<out>/pools/n<N>/<dataset>/hf_dataset`` (questions outside every evaluation split; MuSiQue, bAbI, and
StrategyQA pools are random subsets of their ``train`` split, drawn by ``pipeline.generate --max_samples``).

Example:
    python -m chainuq.pipeline.prepare_data --pools 2000,6000
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from chainuq import config
from chainuq.data.answers import last_boxed
from chainuq.registry import DATASETS
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

SEED, TEST2_SEED = config.SEED, 2027
HOTPOT_TRAIN_SIZE = 30000
BABI_TEST_PER_TASK = 250
MATH_SUBJECTS = ("algebra", "counting_and_probability", "geometry", "intermediate_algebra",
                 "number_theory", "prealgebra", "precalculus")
MATH_COLUMNS = ["problem", "solution", "answer", "subject", "level", "unique_id"]


def _shuffled(n: int, seed: int) -> list:
    idx = list(range(n))
    random.Random(seed).shuffle(idx)
    return idx


def _halve(ds: Dataset):
    idx = _shuffled(len(ds), SEED)
    half = len(idx) // 2
    return ds.select(sorted(idx[:half])), ds.select(sorted(idx[half:]))


def _split_frac(ds: Dataset, frac: float):
    idx = _shuffled(len(ds), SEED)
    cut = int(round(len(idx) * frac))
    return ds.select(sorted(idx[:cut])), ds.select(sorted(idx[cut:]))


def _stratified(ds: Dataset, key_fn, n_total: int = None, per_group: int = None):
    """Per-group random subset: ``per_group`` rows of every group, or ``n_total`` rows in proportion."""
    groups = defaultdict(list)
    for i, row in enumerate(ds):
        groups[key_fn(row)].append(i)
    rng = random.Random(SEED)
    picked = []
    for key in sorted(groups):
        members = groups[key][:]
        rng.shuffle(members)
        picked.extend(members[:per_group if per_group is not None else int(round(n_total * len(members) / len(ds)))])
    if per_group is None:
        rng.shuffle(picked)
        picked = picked[:n_total]
    return ds.select(sorted(picked))


def _test2_sample(ds: Dataset, n: int, key=None):
    """Held-out sample with seed 2027: uniform, or an equal share per group of ``key``."""
    rng = random.Random(TEST2_SEED)
    if key is None:
        idx = list(range(len(ds)))
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


def _add_uid(ds: Dataset, prefix: str, id_field: str = None) -> Dataset:
    def fn(row, i):
        base = str(row[id_field]) if id_field and row.get(id_field) not in (None, "") else str(i)
        return {"split_uid": f"{prefix}:{base}"}
    return ds.map(fn, with_indices=True)


def _math_test_pool():
    pool = concatenate_datasets([load_dataset("EleutherAI/hendrycks_math", s)["test"] for s in MATH_SUBJECTS])
    pool = pool.map(lambda r: {"answer": last_boxed(r["solution"]), "subject": r["type"], "unique_id": ""})
    return pool.filter(lambda r: bool(r["answer"]))


def prep_hotpotqa():
    raw = load_dataset("hotpotqa/hotpot_qa", "distractor")
    val, test = _halve(raw["validation"])
    train = _stratified(raw["train"], lambda r: (r["level"], r["type"]), n_total=HOTPOT_TRAIN_SIZE)
    used = set(train["id"]) | set(val["id"]) | set(test["id"])
    t2 = _test2_sample(raw["train"].filter(lambda r: r["level"] == "hard" and r["id"] not in used), 1500)
    return DatasetDict({"train": _add_uid(train, "hotpotqa", "id"), "validation": _add_uid(val, "hotpotqa", "id"),
                        "test": _add_uid(test, "hotpotqa", "id"), "train_full": _add_uid(raw["train"], "hotpotqa", "id"),
                        "test2": _add_uid(t2, "hotpotqa-t2", "id")})


def prep_musique():
    raw = load_dataset("dgslibisey/MuSiQue")
    val, test = _halve(raw["validation"])
    return DatasetDict({"train": _add_uid(raw["train"], "musique", "id"), "validation": _add_uid(val, "musique", "id"),
                        "test": _add_uid(test, "musique", "id"),
                        "test2": _add_uid(_test2_sample(raw["train"], 1200), "musique-t2", "id")})


def prep_strategyqa():
    raw = load_dataset("ChilleD/StrategyQA")
    val, train = _split_frac(raw["train"], 0.1)
    return DatasetDict({"train": _add_uid(train, "strategyqa", "qid"), "validation": _add_uid(val, "strategyqa", "qid"),
                        "test": _add_uid(raw["test"], "strategyqa", "qid")})


def prep_babi():
    raw = load_dataset("Muennighoff/babi")
    test_full = _add_uid(raw["test"], "babi-test")
    test = _stratified(test_full, lambda r: int(r["task"]), per_group=BABI_TEST_PER_TASK)
    used = set(test["split_uid"])
    t2 = _test2_sample(test_full.filter(lambda r: r["split_uid"] not in used), 2000, key=lambda r: int(r["task"]))
    t2 = t2.map(lambda r: {"split_uid": r["split_uid"].replace("babi-test:", "babi-t2:")})
    return DatasetDict({"train": _add_uid(raw["train"], "babi-train"), "validation": _add_uid(raw["validation"], "babi-val"),
                        "test": test, "test_full": test_full, "test2": t2})


def prep_gsm8k():
    raw = load_dataset("openai/gsm8k", "main")
    val, _ = _split_frac(raw["train"], 500 / len(raw["train"]))
    used = set(val["question"])
    t2 = _test2_sample(raw["train"].filter(lambda r: r["question"] not in used), 1000)
    return DatasetDict({"validation": _add_uid(val, "gsm8k-train"), "test": _add_uid(raw["test"], "gsm8k-test"),
                        "test2": _add_uid(t2, "gsm8k-t2")})


def prep_math500():
    test = load_dataset("HuggingFaceH4/MATH-500")["test"]
    held = {p.strip() for p in test["problem"]}
    pool = _math_test_pool().filter(lambda r: r["problem"].strip() not in held)
    val = _stratified(pool, lambda r: (r["subject"], r["level"]), n_total=500)
    used = held | {p.strip() for p in val["problem"]}
    t2 = _test2_sample(pool.filter(lambda r: r["problem"].strip() not in used), 1000, key=lambda r: (r["subject"], r["level"]))
    return DatasetDict({"validation": _add_uid(val.select_columns(MATH_COLUMNS), "math-test"),
                        "test": _add_uid(test.select_columns(MATH_COLUMNS), "math500", "unique_id"),
                        "test2": _add_uid(t2.select_columns(MATH_COLUMNS), "math-t2")})


PREPARERS = {"hotpotqa": prep_hotpotqa, "musique": prep_musique, "strategyqa": prep_strategyqa, "babi": prep_babi,
             "gsm8k": prep_gsm8k, "math500": prep_math500}


def pools(out_root: Path, sizes):
    """Unlabeled GSM8K / MATH train questions outside every evaluation split (seed-2026 shuffle, first N)."""
    from datasets import load_from_disk
    gsm = load_from_disk(str(out_root / "gsm8k" / "hf_dataset"))
    used = set().union(*(gsm[s]["question"] for s in gsm))
    train = load_dataset("openai/gsm8k", "main")["train"]
    gsm_rows = [i for i, q in enumerate(train["question"]) if q not in used]
    random.Random(SEED).shuffle(gsm_rows)
    math_rows = [(s, i, r) for s in MATH_SUBJECTS
                 for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", s)["train"]) if last_boxed(r["solution"])]
    random.Random(SEED).shuffle(math_rows)
    for n in sizes:
        root = out_root / "pools" / f"n{n}"
        sel = sorted(gsm_rows[:n])
        gsm_pool = train.select(sel).map(lambda r, i: {"split_uid": f"gsm8k-train:{sel[i]}"}, with_indices=True)
        DatasetDict({"train": gsm_pool}).save_to_disk(str(root / "gsm8k" / "hf_dataset"))
        DatasetDict({"train": Dataset.from_list([
            {"problem": r["problem"], "solution": r["solution"], "answer": last_boxed(r["solution"]), "subject": r["type"],
             "level": r["level"], "unique_id": f"train/{s}/{i}", "split_uid": f"math-train:{s}:{i}"}
            for s, i, r in math_rows[:n]])}).save_to_disk(str(root / "math500" / "hf_dataset"))
        print(f"pools n={n}: gsm8k {len(sel)}, math {min(n, len(math_rows))}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", default=str(config.DATASETS_ROOT))
    p.add_argument("--only", default=",".join(PREPARERS), help="datasets to (re)build")
    p.add_argument("--pools", default="", help="comma-separated pool sizes, e.g. 2000,6000")
    args = p.parse_args()
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "split_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for name in [x for x in args.only.split(",") if x]:
        dd = PREPARERS[name]().flatten_indices()
        dd.save_to_disk(str(out_root / DATASETS[name]["dir"] / "hf_dataset"))
        summary[name] = {split: len(ds) for split, ds in dd.items()}
        print(name, summary[name], flush=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    if args.pools:
        pools(out_root, [int(n) for n in args.pools.split(",")])


if __name__ == "__main__":
    main()
