#!/usr/bin/env python3
"""Read-only data-quality audit of one backbone's artifacts (CPU only).

Generation: parse failures, truncation (finish_reason=length), steps per response, accuracy.
Judges:     parse failures, label balance, Stage-1 vs exact-match agreement where both exist,
            all-steps-consistent rate, steps vs steps_noconcl agreement, Judge-Final consistency.
Features:   coverage vs parsed generations, NaN/Inf, greedy-argmax check, span bounds.
Side outputs: temperature samples, CoT-UQ keywords, NLI step verdicts.

Example:
    python pipeline/audit.py --backbone llama
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq.io import index_by_uid, read_jsonl
from cuq.registry import ARTIFACTS, BACKBONES, DATASETS, FORMATS, MAIN_JUDGE, feat_dir, gen_path, judge_path


def pct(x):
    return f"{100 * x:.1f}%"


def audit_gen(bb):
    print("\n## generation (fmt / dataset / split: n, parse-fail, truncated, mean steps, acc[label])")
    for fmt in FORMATS:
        for ds in DATASETS:
            for split in ("train", "validation", "test"):
                p = gen_path(bb, ds, split, fmt)
                if not p.exists():
                    continue
                rs = read_jsonl(p)
                ok = [r for r in rs if r["parse_ok"]]
                trunc = np.mean([r["finish_reason"] == "length" for r in rs])
                steps = np.mean([len(r["steps"]) for r in ok]) if ok else float("nan")
                lab = load_label(bb, ds, split, fmt)
                acc = np.mean([lab[r["uid"]] for r in ok if r["uid"] in lab]) if lab else float("nan")
                drops = Counter(r["drop_reason"].split(":")[0] for r in rs if not r["parse_ok"]).most_common(2)
                flag = "  <-- check" if (1 - len(ok) / len(rs)) > 0.1 or trunc > 0.05 else ""
                print(f"{fmt:8s} {ds:10s} {split:10s} n={len(rs):6d} fail={pct(1 - len(ok) / len(rs)):>6s} "
                      f"trunc={pct(trunc):>6s} steps={steps:4.1f} acc={acc:.3f} {dict(drops)}{flag}")


def load_label(bb, ds, split, fmt):
    if DATASETS[ds]["label"] == "em":
        return {r["uid"]: r["label_em"] for r in read_jsonl(gen_path(bb, ds, split, fmt)) if r.get("parse_ok")}
    p = judge_path(MAIN_JUDGE, "correct", bb, ds, split, fmt)
    return {r["uid"]: r["value"] for r in read_jsonl(p) if r["value"] in (0, 1)} if p.exists() else {}


def audit_judges(bb):
    print("\n## main judge (task / dataset / split: n, parse-fail, positive rate)")
    for task in ("correct", "steps", "steps_noconcl", "final_blind"):
        for ds in DATASETS:
            for split in ("train", "validation", "test"):
                p = judge_path(MAIN_JUDGE, task, bb, ds, split)
                if not p.exists():
                    continue
                rs = read_jsonl(p)
                fail = np.mean([r["value"] is None for r in rs])
                vals = [r["value"] for r in rs if r["value"] is not None]
                if task in ("steps", "steps_noconcl"):
                    pos = np.mean([np.mean(v) for v in vals])
                    allc = np.mean([all(x == 1 for x in v) for v in vals])
                    extra = f"step-valid={pos:.3f} all-steps-valid={allc:.3f}"
                elif task == "final_blind":
                    pv = [r.get("p_verbal") for r in rs if r.get("p_verbal") is not None]
                    incons = np.mean([(r["value"] == 1) != (r["p_verbal"] > 0.5) for r in rs
                                      if r["value"] is not None and r.get("p_verbal") is not None])
                    extra = f"judged-correct={np.mean(vals):.3f} mean p={np.mean(pv):.3f} verdict/prob-inconsistent={pct(incons)}"
                else:
                    extra = f"correct={np.mean(vals):.3f}"
                flag = "  <-- check" if fail > 0.02 else ""
                print(f"{task:13s} {ds:10s} {split:10s} n={len(rs):6d} fail={pct(fail):>6s} {extra}{flag}")

    print("\n## steps vs steps_noconcl (test, step-level agreement) and accuracy|all-steps-valid")
    for ds in DATASETS:
        a = judge_path(MAIN_JUDGE, "steps", bb, ds, "test")
        b = judge_path(MAIN_JUDGE, "steps_noconcl", bb, ds, "test")
        if not (a.exists() and b.exists()):
            continue
        A, B = index_by_uid(read_jsonl(a)), index_by_uid(read_jsonl(b))
        x, y = [], []
        for u, r in A.items():
            if isinstance(r["value"], list) and u in B and isinstance(B[u]["value"], list) and len(B[u]["value"]) == len(r["value"]):
                x += r["value"]
                y += B[u]["value"]
        lab = load_label(bb, ds, "test", "step")
        allv = [(lab[u], all(v == 1 for v in r["value"])) for u, r in A.items() if u in lab and isinstance(r["value"], list)]
        acc_valid = np.mean([l for l, v in allv if v]) if any(v for _, v in allv) else float("nan")
        acc_invalid = np.mean([l for l, v in allv if not v]) if any(not v for _, v in allv) else float("nan")
        print(f"{ds:10s} agree={np.mean(np.asarray(x) == np.asarray(y)):.3f}  acc|all-valid={acc_valid:.3f}  "
              f"acc|some-invalid={acc_invalid:.3f}")


def audit_features(bb, n_check=200):
    print("\n## features (dataset / split / fmt: stored vs parsed, NaN, argmax-match, span ok)")
    rng = random.Random(0)
    for fmt in FORMATS:
        for ds in DATASETS:
            for split in ("train", "validation", "test"):
                d = feat_dir(bb, ds, split, fmt)
                if not (d / "DONE.json").exists():
                    continue
                gens = {r["uid"]: r for r in read_jsonl(gen_path(bb, ds, split, fmt)) if r["parse_ok"]}
                shards = sorted(d.glob("shard_*.pt"))
                n, bad, match, span_bad, checked = 0, 0, [], 0, 0
                for sp in shards:
                    data = torch.load(sp, weights_only=False)
                    n += len(data)
                    for uid, rec in data.items():
                        if checked >= n_check and rng.random() > 0.05:
                            continue
                        checked += 1
                        g = gens[uid]
                        T = len(g["gen_ids"])
                        if rec["tok"].shape[0] != T or g["conclusion"]["tok"][1] > T:
                            span_bad += 1
                        for k in ("tok", "ll", "ent", "layers_anchor", "lookback"):
                            if not torch.isfinite(rec[k].float()).all():
                                bad += 1
                                break
                        match.append(float(((rec["ll"] - rec["top1"]).abs() < 1e-3).float().mean()))
                flag = "  <-- check" if n != len(gens) or bad or span_bad or np.mean(match) < 0.97 else ""
                print(f"{ds:10s} {split:10s} {fmt:8s} stored={n:6d}/{len(gens):6d} nonfinite={bad} "
                      f"argmax={np.mean(match):.3f} span_bad={span_bad} (checked {checked}){flag}")


def audit_side(bb):
    print("\n## side outputs")
    for ds in DATASETS:
        for split in ("validation", "test"):
            p = gen_path(bb, ds, split, "step", tag="samples")
            if p.exists():
                rs = read_jsonl(p)
                n_s = Counter(len(r["samples"]) for r in rs)
                empty = np.mean([np.mean([not s["answer"] for s in r["samples"]]) for r in rs])
                distinct = np.mean([len(set(s["answer"].lower() for s in r["samples"])) for r in rs])
                print(f"samples  {ds:10s} {split:10s} n={len(rs)} per-q={dict(n_s)} empty-answer={pct(empty)} "
                      f"distinct-answers/q={distinct:.2f}")
            p = ARTIFACTS / "cotuq" / bb / ds / "step" / f"{split}.jsonl"
            if p.exists():
                rs = read_jsonl(p)
                print(f"cotuq    {ds:10s} {split:10s} n={len(rs)} with-keywords={pct(np.mean([bool(r['keywords']) for r in rs]))} "
                      f"kw/resp={np.mean([len(r['keywords']) for r in rs]):.1f}")
            p = judge_path("nli", "steps", bb, ds, split)
            if p.exists():
                rs = read_jsonl(p)
                print(f"nli      {ds:10s} {split:10s} n={len(rs)} step-entailed={np.mean([np.mean(r['value']) for r in rs if r['value']]):.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    args = p.parse_args()
    print(f"# audit {args.backbone}  ({ARTIFACTS})")
    audit_gen(args.backbone)
    audit_judges(args.backbone)
    audit_features(args.backbone)
    audit_side(args.backbone)


if __name__ == "__main__":
    main()
