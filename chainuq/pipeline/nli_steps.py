"""Non-LLM step verifier: DeBERTa-large-MNLI entailment of each reasoning step by the task input.

A step is labeled consistent (1) if some context passage (or the question itself) entails it with
probability > 0.5. The output uses the judge-file format under the judge name ``nli``, so it can replace the
LLM step judge in the verifier-swap analysis.

Example:
    CUDA_VISIBLE_DEVICES=0 python -m chainuq.pipeline.nli_steps --backbone llama --datasets hotpotqa,musique
"""

from __future__ import annotations

import argparse
import re

from chainuq.artifacts import gen_path, judge_path, read_jsonl, write_jsonl
from chainuq.nli import NLI
from chainuq.registry import BACKBONES, DATASETS


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
    args = p.parse_args()
    nli = NLI()
    for ds in args.datasets.split(","):
        for split in args.splits.split(","):
            src = gen_path(args.backbone, ds, split)
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
            write_jsonl(judge_path("nli", "steps", args.backbone, ds, split), rows)
            print(f"[nli_steps] {args.backbone} {ds} {split}: {len(rows)} samples, {len(pairs)} pairs", flush=True)


if __name__ == "__main__":
    main()
