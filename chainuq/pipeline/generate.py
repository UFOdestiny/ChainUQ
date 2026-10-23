"""Stage 1: greedy generation with a frozen backbone (vLLM).

Writes one JSONL record per question to ``gen/<backbone>/<dataset>/<split>.jsonl``: prompt and generated
token ids, the response text, the parsed reasoning steps and conclusion with their token spans, the reference
answer, and (for answer-matching datasets) the correctness label. Features are extracted later from the
stored token ids (``pipeline.features``), so generation and extraction can run on different GPUs.

With ``--n_samples > 1`` the script instead draws temperature samples for the sampling baselines
(Semantic Entropy, SelfCheckGPT) and writes ``<split>.samples.jsonl``.

Example:
    CUDA_VISIBLE_DEVICES=0 python -m chainuq.pipeline.generate --backbone llama --dataset hotpotqa \\
        --splits train,validation,test
"""

from __future__ import annotations

import argparse
import json
import time

from chainuq import config
from chainuq.artifacts import gen_path, write_jsonl
from chainuq.data import load_dataset
from chainuq.data.prompts import build_messages, render_prompt_ids
from chainuq.data.response import parse_response
from chainuq.registry import BACKBONES, DATASETS

MAX_NEW_TOKENS = {"gsm8k": 512, "math500": 768}  # math solutions need a longer budget than short-answer QA


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--dataset", required=True, help="comma-separated dataset names")
    p.add_argument("--splits", default="train,validation,test")
    p.add_argument("--max_samples", type=int, default=0, help="0 = full split; otherwise a random subset (seed 2026)")
    p.add_argument("--max_new_tokens", type=int, default=320, help="QA budget; math datasets use MAX_NEW_TOKENS")
    p.add_argument("--max_model_len", type=int, default=4096)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--n_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=1.0, help="only used with --n_samples > 1")
    p.add_argument("--data_root", default="", help="dataset root (default: CHAINUQ_DATASETS), e.g. an adaptation pool")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    spec = BACKBONES[args.backbone]
    tokenizer = AutoTokenizer.from_pretrained(spec.path)
    llm = None
    sampling = args.n_samples > 1
    tag = "samples" if sampling else ""

    for dataset in [d for d in args.dataset.split(",") if d]:
        assert dataset in DATASETS, dataset
        for split in [s for s in args.splits.split(",") if s]:
            out = gen_path(args.backbone, dataset, split, tag=tag)
            if out.exists() and not args.overwrite:
                print(f"[skip] {out} exists", flush=True)
                continue
            ds = load_dataset(dataset, split, args.max_samples, root=args.data_root or None)
            max_new = MAX_NEW_TOKENS.get(dataset, args.max_new_tokens)
            prompts, metas, n_too_long = [], [], 0
            for raw in ds.data:
                messages = build_messages(ds, raw)
                ids = render_prompt_ids(tokenizer, messages, spec.chat_template_kwargs)
                if len(ids) > args.max_model_len - max_new:
                    n_too_long += 1
                    continue
                prompts.append(ids)
                metas.append((raw, messages, ids))

            if llm is None:
                llm = LLM(model=str(spec.path), tokenizer=str(spec.path), max_model_len=args.max_model_len,
                          gpu_memory_utilization=args.gpu_memory_utilization, seed=config.SEED, dtype="bfloat16")
            params = (SamplingParams(max_tokens=max_new, temperature=args.temperature, top_p=1.0, n=args.n_samples,
                                     seed=config.SEED) if sampling else SamplingParams(max_tokens=max_new, temperature=0.0, n=1))
            t0 = time.time()
            outputs = llm.generate([{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False)
            elapsed = time.time() - t0

            records, n_parse_fail = [], 0
            for (raw, messages, ids), output in zip(metas, outputs):
                gold = ds.get_ground_truth(raw)
                rec = {"uid": raw["split_uid"], "dataset": dataset, "split": split, "backbone": args.backbone,
                       "question": raw.get("question", ""),
                       "user_content": next(m["content"] for m in messages if m["role"] == "user"),
                       "gold": gold, "gold_aliases": [str(a) for a in raw.get("answer_aliases") or [] if str(a).strip()],
                       "n_hops": int(ds.get_difficulty(raw))}
                if sampling:
                    rec["samples"] = [{"text": o.text, "answer": ds.parse_answer(o.text), "n_tokens": len(o.token_ids)}
                                      for o in output.outputs]
                    records.append(rec)
                    continue
                o = output.outputs[0]
                gen_ids = [int(t) for t in o.token_ids]
                parsed = parse_response(o.text, gen_ids, tokenizer)
                n_parse_fail += int(not parsed["parse_ok"])
                rec.update({
                    "prompt_ids": ids, "gen_ids": gen_ids, "gen_text": o.text, "finish_reason": o.finish_reason,
                    "pred_answer": ds.parse_answer(o.text),
                    "label_em": int(ds.check_correctness(o.text, gold)) if DATASETS[dataset]["label"] == "em" else None,
                    **parsed,
                })
                records.append(rec)

            write_jsonl(out, records)
            stats = {"backbone": args.backbone, "dataset": dataset, "split": split, "tag": tag,
                     "n_items": len(ds.data), "n_prompt_too_long": n_too_long, "n_generated": len(records),
                     "n_parse_fail": n_parse_fail, "gen_seconds": round(elapsed, 1),
                     "gen_tokens": int(sum(len(o.outputs[0].token_ids) for o in outputs)),
                     "prompt_tokens": int(sum(len(p) for p in prompts))}
            out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
            print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
