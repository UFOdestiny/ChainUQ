"""Stage 2 with a black-box API judge (Anthropic Messages API): same prompts, tasks, and output files as
``chainuq.pipeline.judge``. The API returns no token log-probabilities, so ``p_true`` is None and the verdict
is the verbalized probability.

Example:
    python -m chainuq.pipeline.judge_api --judge haiku --task final_blind --backbone llama \
        --dataset hotpotqa --splits validation,test
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path

from chainuq.artifacts import gen_path, judge_path, read_jsonl, write_jsonl
from chainuq.pipeline.judge import PROMPTS, TASKS, _lenient_json, parse_output
from chainuq.registry import BACKBONES, DATASETS, JUDGES


def api_schema(task: str) -> dict:
    """Output schema without the numeric/length constraints the API does not enforce (checked in parse_output)."""
    if task == "steps":
        props = {"reasoning_verified": {"type": "array", "items": {"type": "integer", "enum": [0, 1]}}}
    elif task == "final_blind":
        props = {"correct": {"type": "boolean"}, "probability_correct": {"type": "integer"}}
    else:
        props = {"correct": {"type": "boolean"}}
    return {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}


def load_api_key() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    env = Path(os.environ.get("CHAINUQ_ANTHROPIC_ENV", "/u/kytrio/proj/TrAC/.env"))
    for line in env.read_text().splitlines():
        if line.strip().startswith("ANTHROPIC_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("ANTHROPIC_API_KEY not found")


async def judge_one(client, model, task, rec, sem, max_tokens, effort):
    import anthropic

    kwargs = dict(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": PROMPTS[task](rec)}],
        output_config={"effort": effort, "format": {"type": "json_schema", "schema": api_schema(task)}},
    )
    async with sem:
        for attempt in range(6):
            try:
                resp = await client.messages.create(**kwargs)
                break
            except (anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError) as exc:
                if attempt == 5:
                    return {"uid": rec["uid"], "value": None, "error": repr(exc)[:200]}
                await asyncio.sleep(min(60, 4 * 2**attempt))
            except anthropic.APIStatusError as exc:
                return {"uid": rec["uid"], "value": None, "error": repr(exc)[:200]}
    content = next((b.text for b in resp.content if b.type == "text"), "")
    out = {
        "uid": rec["uid"],
        "value": parse_output(task, rec, content),
        "raw": content[:400],
        "stop_reason": resp.stop_reason,
        "prompt_tokens": resp.usage.input_tokens,
        "completion_tokens": resp.usage.output_tokens,
    }
    if task == "final_blind":
        out["p_true"] = None
        try:
            out["p_verbal"] = min(1.0, max(0.0, float(json.loads(content)["probability_correct"]) / 100.0))
        except Exception:
            lenient = _lenient_json(content) or {}
            out["p_verbal"] = lenient["probability_correct"] / 100.0 if "probability_correct" in lenient else None
    return out


async def run(args):
    import anthropic

    spec = JUDGES[args.judge]
    client = anthropic.AsyncAnthropic(api_key=load_api_key(), max_retries=2, timeout=120)
    sem = asyncio.Semaphore(args.concurrency)
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        out = judge_path(args.judge, args.task, args.backbone, args.dataset, split)
        if out.exists() and not args.overwrite:
            print(f"[skip] {out} exists", flush=True)
            continue
        recs = [r for r in read_jsonl(gen_path(args.backbone, args.dataset, split)) if r.get("parse_ok")]
        if args.max_samples:
            recs = recs[: args.max_samples]
        t0 = time.time()
        results = await asyncio.gather(*[judge_one(client, spec.name, args.task, r, sem, args.max_tokens, args.effort) for r in recs])
        write_jsonl(out, results)
        stats = {
            "judge": args.judge,
            "task": args.task,
            "backbone": args.backbone,
            "dataset": args.dataset,
            "split": split,
            "n": len(results),
            "n_parse_fail": sum(r["value"] is None for r in results),
            "n_error": sum("error" in r for r in results),
            "seconds": round(time.time() - t0, 1),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in results),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in results),
        }
        out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", default="haiku", choices=sorted(JUDGES))
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--splits", default="validation,test")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--effort", default="low")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
