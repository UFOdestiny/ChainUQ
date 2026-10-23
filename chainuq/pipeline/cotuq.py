"""CoT-UQ keyword extraction (Zhang & Zhang, Findings of ACL 2025), re-implemented.

The backbone itself lists, for each of its reasoning steps, the keywords that matter for the final answer with
an importance score in 1-10. ``pipeline.scores --methods cotuq`` then aggregates the token probabilities of those
keywords weighted by importance (the probability-aggregation variant).

Example:
    CUDA_VISIBLE_DEVICES=0 python -m chainuq.pipeline.cotuq --backbone llama --datasets hotpotqa,musique
"""

from __future__ import annotations

import argparse
import re

from chainuq import config
from chainuq.artifacts import cotuq_path, gen_path, read_jsonl, write_jsonl
from chainuq.data.prompts import render_prompt_ids
from chainuq.registry import BACKBONES, DATASETS

INSTRUCTION = (
    "Below is a question, your reasoning steps, and your final answer.\n"
    "For each reasoning step, extract the keywords (exact words or short phrases copied from that step) "
    "that are most important for reaching the final answer, and rate the importance of each keyword "
    "from 1 (barely relevant) to 10 (essential).\n"
    "Output one line per keyword in the form:\n"
    "Step <k>: <keyword> | <importance>\n"
    "Output nothing else."
)

LINE = re.compile(r"Step\s*(\d+)\s*:\s*(.+?)\s*\|\s*(\d+)")


def build_messages(rec):
    steps = "\n".join(f"Step {i + 1}: {s['text']}" for i, s in enumerate(rec["steps"]))
    body = f"Question: {rec['question']}\n\nReasoning:\n{steps}\nConclusion: {rec['conclusion']['text']}"
    return [{"role": "user", "content": f"{INSTRUCTION}\n\n{body}"}]


def parse(text):
    out = []
    for m in LINE.finditer(text or ""):
        kw = m.group(2).strip().strip("\"'")
        if kw:
            out.append({"step": int(m.group(1)), "keyword": kw, "importance": min(10, int(m.group(3)))})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--splits", default="validation,test")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    spec = BACKBONES[args.backbone]
    tok = AutoTokenizer.from_pretrained(spec.path)
    llm = LLM(model=str(spec.path), max_model_len=4096, gpu_memory_utilization=args.gpu_memory_utilization,
              seed=config.SEED, dtype="bfloat16")
    params = SamplingParams(max_tokens=256, temperature=0.0)
    for ds in args.datasets.split(","):
        for split in args.splits.split(","):
            src = gen_path(args.backbone, ds, split)
            if not src.exists():
                continue
            recs = [r for r in read_jsonl(src) if r.get("parse_ok")]
            if args.max_samples:
                recs = recs[: args.max_samples]
            prompts = [{"prompt_token_ids": render_prompt_ids(tok, build_messages(r), spec.chat_template_kwargs)} for r in recs]
            outs = llm.generate(prompts, params, use_tqdm=False)
            rows = [{"uid": r["uid"], "raw": o.outputs[0].text, "keywords": parse(o.outputs[0].text)}
                    for r, o in zip(recs, outs)]
            write_jsonl(cotuq_path(args.backbone, ds, split), rows)
            n_ok = sum(bool(r["keywords"]) for r in rows)
            print(f"[cotuq] {args.backbone} {ds} {split}: {n_ok}/{len(rows)} with keywords", flush=True)


if __name__ == "__main__":
    main()
