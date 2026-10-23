"""Stage 2: judge annotations through an OpenAI-compatible vLLM server.

Tasks (one output file per task, ``judge/<judge>/<task>/<backbone>/<dataset>/<split>.jsonl``):

  correct        Correctness labeler. Sees the reference answer (+ aliases) and the model's conclusion only;
                 produces the response label y. Not run for answer-matching datasets.
  steps          Step verifier, answer-blind: sees the task input, the model's reasoning steps and its own
                 conclusion, never the reference answer. One 0/1 label per step (u_judge evidence; PCS targets).
  steps_noconcl  Control: as ``steps``, with the model's conclusion also withheld.
  final_blind    Chain verdict, answer-blind: task input + full response; whether the conclusion is correct,
                 with a verbalized probability and P(true) from the token log-probabilities (Judge-Final
                 baseline; PCS verdict target).

The server is started separately (``scripts/serve_model.sh``); this client only needs ``--port``.

Example:
    python -m chainuq.pipeline.judge --judge gptoss --task steps --backbone llama \
        --dataset hotpotqa --splits validation,test --port 8100
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import time

from chainuq.artifacts import gen_path, judge_path, read_jsonl, write_jsonl
from chainuq.registry import BACKBONES, DATASETS, JUDGES

TASKS = ("correct", "steps", "steps_noconcl", "final_blind")


def _gold_text(rec: dict) -> str:
    aliases = [a for a in rec.get("gold_aliases") or [] if a and a != rec["gold"]]
    if aliases:
        return f"{rec['gold']} (also accepted: {'; '.join(aliases[:8])})"
    return rec["gold"]


def prompt_correct(rec: dict) -> str:
    return (
        "You are an exact answer verifier.\n"
        "Objective: decide whether the model's conclusion matches the ground-truth answer.\n"
        "Treat paraphrases, aliases and equivalent forms as matching; treat partial or "
        "different entities as not matching.\n"
        "Input fields:\n"
        f"Question: {rec['question']}\n"
        f"Ground Truth Answer: {_gold_text(rec)}\n"
        f"Conclusion Claim: {rec['conclusion']['text']}\n"
        'Output schema (strict JSON): {"correct": true/false}'
    )


def prompt_steps(rec: dict) -> str:
    steps = "\n".join(f"Step {i + 1}: {s['text']}" for i, s in enumerate(rec["steps"]))
    n = len(rec["steps"])
    return (
        "You are an exact reasoning-step verifier.\n"
        "Objective: evaluate each reasoning step for contextual support and logical "
        "consistency toward the conclusion. A step is valid (1) only if it is supported "
        "by the provided task input, logically coherent with the preceding steps, and "
        "relevant to reaching the conclusion; otherwise it is invalid (0).\n"
        "You are NOT given the reference answer; judge the steps on their own merits.\n\n"
        f"Task input:\n{rec['user_content']}\n\n"
        f"Model reasoning:\n{steps}\n"
        f"Model conclusion: {rec['conclusion']['text']}\n\n"
        'Output schema (strict JSON): {"reasoning_verified": [0 or 1 for each step]}\n'
        f"The array must contain exactly {n} values."
    )


def prompt_steps_noconcl(rec: dict) -> str:
    """Control for the 'step judge secretly grades the answer' concern: the conclusion is withheld."""
    steps = "\n".join(f"Step {i + 1}: {s['text']}" for i, s in enumerate(rec["steps"]))
    n = len(rec["steps"])
    return (
        "You are an exact reasoning-step verifier.\n"
        "Objective: evaluate each reasoning step for contextual support and logical "
        "consistency. A step is valid (1) only if it is supported by the provided task "
        "input and logically coherent with the preceding steps; otherwise it is invalid (0).\n"
        "You are NOT given the reference answer or the model's final answer; judge the steps on their own merits.\n\n"
        f"Task input:\n{rec['user_content']}\n\n"
        f"Model reasoning:\n{steps}\n\n"
        'Output schema (strict JSON): {"reasoning_verified": [0 or 1 for each step]}\n'
        f"The array must contain exactly {n} values."
    )


def prompt_final_blind(rec: dict) -> str:
    return (
        "You are a careful answer verifier.\n"
        "Objective: decide whether the model's final conclusion correctly answers the "
        "question, using only the task input below. You are NOT given the reference answer.\n\n"
        f"Task input:\n{rec['user_content']}\n\n"
        f"Model response:\n{rec['gen_text']}\n\n"
        "Also give probability_correct: your probability (0-100) that the model's final conclusion IS "
        "correct (100 = certainly correct, 0 = certainly wrong). It must agree with \"correct\": "
        "values above 50 when correct is true, below 50 when correct is false.\n"
        'Output schema (strict JSON): {"correct": true/false, "probability_correct": <integer 0-100>}'
    )


def schema_for(task: str, rec: dict) -> dict:
    if task == "final_blind":
        return {"type": "object",
                "properties": {"correct": {"type": "boolean"},
                               "probability_correct": {"type": "integer", "minimum": 0, "maximum": 100}},
                "required": ["correct", "probability_correct"], "additionalProperties": False}
    if task == "correct":
        return {"type": "object", "properties": {"correct": {"type": "boolean"}},
                "required": ["correct"], "additionalProperties": False}
    n = len(rec["steps"])
    return {
        "type": "object",
        "properties": {"reasoning_verified": {
            "type": "array", "items": {"type": "integer", "enum": [0, 1]},
            "minItems": n, "maxItems": n}},
        "required": ["reasoning_verified"],
        "additionalProperties": False,
    }


PROMPTS = {"correct": prompt_correct, "steps": prompt_steps, "steps_noconcl": prompt_steps_noconcl,
           "final_blind": prompt_final_blind}


def _p_true_from_logprobs(logprobs) -> float | None:
    """P(true) at the last generated true/false token, renormalised over {true, false}."""
    if not logprobs or not getattr(logprobs, "content", None):
        return None
    for entry in reversed(logprobs.content):
        if entry.token.strip().lower() not in ("true", "false"):
            continue
        lp = {"true": -math.inf, "false": -math.inf}
        for alt in entry.top_logprobs or []:
            key = alt.token.strip().lower()
            if key in lp:
                lp[key] = max(lp[key], alt.logprob)
        key = entry.token.strip().lower()
        lp[key] = max(lp[key], entry.logprob)
        m = max(lp.values())
        if m == -math.inf:
            return None
        a, b = math.exp(lp["true"] - m), math.exp(lp["false"] - m)
        return a / (a + b)
    return None


def _lenient_json(content: str):
    """Recover the payload when the grammar let the model loop on whitespace before the closing brace."""
    obj = {}
    m = re.search(r'"reasoning_verified"\s*:\s*\[([0-9,\s]*)\]', content or "")
    if m:
        obj["reasoning_verified"] = [int(x) for x in re.findall(r"\d+", m.group(1))]
    m = re.search(r'"correct"\s*:\s*(true|false)', content or "")
    if m:
        obj["correct"] = m.group(1) == "true"
    m = re.search(r'"probability_correct"\s*:\s*(\d+)', content or "")
    if m:
        obj["probability_correct"] = int(m.group(1))
    return obj or None


def parse_output(task: str, rec: dict, content: str):
    try:
        obj = json.loads((content or "").strip())
    except json.JSONDecodeError:
        obj = _lenient_json(content)
        if obj is None:
            return None
    if task in ("correct", "final_blind"):
        val = obj.get("correct")
        return int(val) if isinstance(val, bool) else None
    arr = obj.get("reasoning_verified")
    if not isinstance(arr, list) or len(arr) != len(rec["steps"]):
        return None
    if any(v not in (0, 1) or isinstance(v, bool) for v in arr):
        return None
    return [int(v) for v in arr]


async def judge_one(client, model_name, task, rec, extra, sem, max_tokens):
    messages = [{"role": "user", "content": PROMPTS[task](rec)}]
    kwargs = dict(
        model=model_name,
        messages=messages,
        temperature=0.0,
        max_tokens=max_tokens,
        response_format={"type": "json_schema",
                         "json_schema": {"name": task, "schema": schema_for(task, rec)}},
    )
    if task == "final_blind":
        kwargs.update(logprobs=True, top_logprobs=20)
    kwargs.update(extra)
    async with sem:
        for attempt in range(3):
            try:
                resp = await client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # transient server errors
                if attempt == 2:
                    return {"uid": rec["uid"], "value": None, "error": repr(exc)[:200]}
                await asyncio.sleep(2 * (attempt + 1))
    choice = resp.choices[0]
    content = choice.message.content
    value = parse_output(task, rec, content)
    out = {
        "uid": rec["uid"],
        "value": value,
        "raw": (content or "")[:400],
        "prompt_tokens": resp.usage.prompt_tokens if resp.usage else None,
        "completion_tokens": resp.usage.completion_tokens if resp.usage else None,
    }
    if task == "final_blind":
        out["p_true"] = _p_true_from_logprobs(choice.logprobs)
        try:
            out["p_verbal"] = float(json.loads(content)["probability_correct"]) / 100.0
        except Exception:
            lenient = _lenient_json(content) or {}
            out["p_verbal"] = lenient["probability_correct"] / 100.0 if "probability_correct" in lenient else None
    return out


async def run(args):
    from openai import AsyncOpenAI

    spec = JUDGES[args.judge]
    client = AsyncOpenAI(base_url=f"http://localhost:{args.port}/v1", api_key="EMPTY", timeout=600)
    sem = asyncio.Semaphore(args.concurrency)
    extra = {}
    if spec.judge_request_kwargs:
        extra.update(spec.judge_request_kwargs)
    if spec.chat_template_kwargs:
        extra["extra_body"] = {"chat_template_kwargs": spec.chat_template_kwargs}

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        src = gen_path(args.backbone, args.dataset, split)
        out = judge_path(args.judge, args.task, args.backbone, args.dataset, split)
        if out.exists() and not args.overwrite:
            print(f"[skip] {out} exists", flush=True)
            continue
        recs = [r for r in read_jsonl(src) if r.get("parse_ok")]
        if args.max_samples:
            recs = recs[: args.max_samples]
        t0 = time.time()
        # Only reasoning judges need a long budget; plain instruct judges answer in a few dozen tokens.
        max_tokens = args.max_tokens if spec.judge_request_kwargs.get("reasoning_effort") else min(args.max_tokens, 384)
        results = await asyncio.gather(*[
            judge_one(client, str(spec.path), args.task, r, extra, sem, max_tokens) for r in recs
        ])
        elapsed = time.time() - t0
        write_jsonl(out, results)
        n_fail = sum(r["value"] is None for r in results)
        stats = {
            "judge": args.judge, "task": args.task, "backbone": args.backbone,
            "dataset": args.dataset, "split": split,
            "n": len(results), "n_parse_fail": n_fail, "seconds": round(elapsed, 1),
            "prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in results),
            "completion_tokens": sum(r.get("completion_tokens") or 0 for r in results),
        }
        out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats), flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--judge", required=True, choices=sorted(JUDGES))
    p.add_argument("--task", required=True, choices=TASKS)
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--splits", default="validation,test")
    p.add_argument("--port", type=int, default=8100)
    p.add_argument("--concurrency", type=int, default=256)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    if args.task == "correct" and DATASETS[args.dataset]["label"] == "em":
        p.error(f"{args.dataset} uses exact-match labels; the 'correct' judge task is not run for it")
    return args


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
