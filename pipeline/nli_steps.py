#!/usr/bin/env python3
"""E3 non-LLM step verifier: DeBERTa-large-MNLI entailment of each step by the context.

A step is labelled consistent (1) if some context passage (or the question
itself) entails it with probability > 0.5. Output uses the judge-file format
under judge name ``nli`` so that it can replace the LLM step judge in E3.

Example:
    CUDA_VISIBLE_DEVICES=2 python pipeline/nli_steps.py --backbone llama --datasets hotpotqa,musique
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq.io import read_jsonl, write_jsonl
from cuq.nli import NLI
from cuq.registry import BACKBONES, DATASETS, gen_path, judge_path


def passages(user_content: str):
    body = user_content.split("\n\nQuestion:")[0]
    body = re.sub(r"^(Context|Passage|Facts):\n", "", body)
    parts = [p.strip() for p in re.split(r"\n(?=\[)|\n(?=- )", body) if p.strip()]
    if len(parts) <= 1:  # bAbI passages: sliding windows of sentences
        sents = [s for s in body.split("\n") if s.strip()]
        parts = ["\n".join(sents[i: i + 8]) for i in range(0, len(sents), 4)] or [body]
    return parts


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--splits", default="validation,test")
    p.add_argument("--fmt", default="step")
    args = p.parse_args()
    nli = NLI()
    for ds in args.datasets.split(","):
        for split in args.splits.split(","):
            src = gen_path(args.backbone, ds, split, args.fmt)
            if not src.exists():
                continue
            recs = [r for r in read_jsonl(src) if r.get("parse_ok")]
            pairs, owner = [], []
            for ri, r in enumerate(recs):
                ctx = passages(r["user_content"]) + [r["question"]]
                for si, st in enumerate(r["steps"]):
                    for c in ctx:
                        pairs.append((c, st["text"]))
                        owner.append((ri, si))
            probs = nli.probs(pairs)
            best = {}
            for (ri, si), pr in zip(owner, probs):
                best[(ri, si)] = max(best.get((ri, si), 0.0), float(pr[2]))
            rows = []
            for ri, r in enumerate(recs):
                ent = [best.get((ri, si), 0.0) for si in range(len(r["steps"]))]
                rows.append({"uid": r["uid"], "value": [int(e > 0.5) for e in ent], "p_entail": ent})
            write_jsonl(judge_path("nli", "steps", args.backbone, ds, split, args.fmt), rows)
            print(f"[nli_steps] {args.backbone} {ds} {split}: {len(rows)} samples, {len(pairs)} pairs", flush=True)


if __name__ == "__main__":
    main()
