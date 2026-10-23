#!/usr/bin/env python3
"""Stage 1: greedy generation with a frozen backbone (vLLM, text only).

Writes one JSONL record per question to ``registry.gen_path``. Hidden-state
features are extracted later by ``pipeline/features.py`` from the stored token
ids, so generation and feature extraction can run on different GPUs or at
different times.

With ``--n_samples > 1`` the script instead draws temperature samples for the
sampling baselines (Semantic Entropy / SelfCheckGPT) and writes the
``samples`` tag file.

Example:
    CUDA_VISIBLE_DEVICES=2 python pipeline/gen.py --backbone llama \
        --dataset hotpotqa --splits train,validation,test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq.formats import build_messages, parse_response, render_prompt_ids
from cuq.io import write_jsonl
from cuq.registry import BACKBONES, DATASETS, DATASETS_ROOT, SEED, gen_path
from data.datasets import get_dataset

DATASET_DIRS = {
    "hotpotqa": "hotpot_qa",
    "musique": "MuSiQue",
    "strategyqa": "StrategyQA",
    "babi": "babi",
    "gsm8k": "gsm8k",
    "math500": "math500",
}
# Math solutions need a longer budget than short-answer QA.
MAX_NEW_TOKENS = {"gsm8k": 512, "math500": 768}


def load_split(dataset: str, split: str, max_samples: int = 0):
    path = DATASETS_ROOT / DATASET_DIRS[dataset] / "hf_dataset"
    return get_dataset(DATASETS[dataset]["loader"], split=split, max_samples=max_samples, dataset_path=str(path))


def gold_aliases(raw: dict) -> list:
    aliases = raw.get("answer_aliases") or []
    return [str(a) for a in aliases if str(a).strip()]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--dataset", required=True, help="comma-separated dataset names")
    p.add_argument("--splits", default="train,validation,test")
    p.add_argument("--fmt", default="step")
    p.add_argument("--max_samples", type=int, default=0, help="0 = full split")
    p.add_argument("--max_new_tokens", type=int, default=320, help="QA default; math datasets use MAX_NEW_TOKENS")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0, help="only used with --n_samples > 1")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    spec = BACKBONES[args.backbone]
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(spec.path)
    llm = None
    sampling_mode = args.n_samples > 1
    tag = "samples" if sampling_mode else ""

    jobs = [(d.strip(), s.strip()) for d in args.dataset.split(",") if d.strip()
            for s in args.splits.split(",") if s.strip()]
    for dataset, split in jobs:
        assert dataset in DATASETS, dataset
        args.dataset = dataset
        out = gen_path(args.backbone, args.dataset, split, args.fmt, tag=tag)
        if out.exists() and not args.overwrite:
            print(f"[skip] {out} exists", flush=True)
            continue
        ds = load_split(args.dataset, split, args.max_samples)
        max_new = MAX_NEW_TOKENS.get(args.dataset, args.max_new_tokens)
        items = list(ds.data)
        prompts, metas = [], []
        n_too_long = 0
        for raw in items:
            messages = build_messages(ds, raw, args.fmt)
            ids = render_prompt_ids(tokenizer, messages, spec.chat_template_kwargs)
            if len(ids) > args.max_model_len - max_new:
                n_too_long += 1
                continue
            prompts.append(ids)
            metas.append((raw, messages, ids))

        if llm is None:
            llm = LLM(
                model=str(spec.path),
                tokenizer=str(spec.path),
                max_model_len=args.max_model_len,
                gpu_memory_utilization=args.gpu_memory_utilization,
                seed=SEED,
                dtype="bfloat16",
            )
        if sampling_mode:
            params = SamplingParams(max_tokens=max_new, temperature=args.temperature,
                                    top_p=1.0, n=args.n_samples, seed=SEED)
        else:
            params = SamplingParams(max_tokens=max_new, temperature=0.0, n=1)

        t0 = time.time()
        outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
        elapsed = time.time() - t0

        records = []
        n_parse_fail = 0
        for (raw, messages, ids), output in zip(metas, outputs):
            gold = ds.get_ground_truth(raw)
            base = {
                "uid": raw["split_uid"],
                "dataset": args.dataset,
                "split": split,
                "fmt": args.fmt,
                "backbone": args.backbone,
                "question": raw.get("question", ""),
                "user_content": next(m["content"] for m in messages if m["role"] == "user"),
                "gold": gold,
                "gold_aliases": gold_aliases(raw),
                "n_hops": int(ds.get_difficulty(raw)),
            }
            if sampling_mode:
                base["samples"] = [
                    {"text": o.text, "answer": ds.parse_answer(o.text), "n_tokens": len(o.token_ids)}
                    for o in output.outputs
                ]
                records.append(base)
                continue
            o = output.outputs[0]
            gen_ids = [int(t) for t in o.token_ids]
            parsed = parse_response(o.text, args.fmt, gen_ids, tokenizer)
            n_parse_fail += int(not parsed["parse_ok"])
            pred = ds.parse_answer(o.text)
            rec = dict(base)
            rec.update({
                "prompt_ids": ids,
                "gen_ids": gen_ids,
                "gen_text": o.text,
                "finish_reason": o.finish_reason,
                "pred_answer": pred,
                # Exact-match label is only authoritative for EM datasets (StrategyQA).
                "label_em": int(ds.check_correctness(o.text, gold)) if DATASETS[args.dataset]["label"] == "em" else None,
                **parsed,
            })
            records.append(rec)

        write_jsonl(out, records)
        stats = {
            "backbone": args.backbone, "dataset": args.dataset, "split": split, "fmt": args.fmt,
            "tag": tag, "n_items": len(items), "n_prompt_too_long": n_too_long,
            "n_generated": len(records), "n_parse_fail": n_parse_fail,
            "gen_seconds": round(elapsed, 1),
            "gen_tokens": int(sum(len(o.outputs[0].token_ids) for o in outputs)),
            "prompt_tokens": int(sum(len(p) for p in prompts)),
        }
        out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
