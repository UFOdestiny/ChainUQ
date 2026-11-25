#!/usr/bin/env python3
"""Stage 4b: scores for methods that are not trained heads.

Each method writes the same prediction format as ``pipeline/train.py``
(``{"uid", "logit", "step_logits"}``; ``logit`` holds the raw confidence score)
under seed ``HEAD_SEEDS[0]``, for the validation and test splits.

  Tier A (no judge at inference):
    mcp         probability of the conclusion claim (product of token probs)
    ppl         mean token log-likelihood of the response (negative log-perplexity)
    mte         negative mean token entropy
    ccp         claim-conditioned probability of the conclusion (NLI over top-10 alternatives)
    cotuq       CoT-UQ, probability-aggregation variant (needs pipeline/cotuq.py output)
    se          negative semantic entropy over N temperature samples (needs gen --n_samples)
    selfcheck   1 - mean NLI contradiction between the greedy answer and the samples
  Tier B (judge / external scorer at inference):
    judge_final P(true) from the answer-blind judge (``final_blind`` task)
    judge_step  fraction of steps the answer-blind step judge marks consistent
    rm          Skywork-Reward-V2 score of (task input, response)

Example:
    CUDA_VISIBLE_DEVICES=2 python pipeline/scores.py --backbone llama --methods mcp,ppl,mte,ccp
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq.io import index_by_uid, read_jsonl, write_jsonl
from cuq.registry import (BACKBONES, DATASETS, HEAD_SEEDS, MAIN_JUDGE, RM_MODEL, ARTIFACTS, gen_path, judge_path,
                          pred_path)
from cuq.store import load_samples

SEED = HEAD_SEEDS[0]


def _write(method, backbone, ds, split, fmt, uids, scores):
    write_jsonl(pred_path(method, backbone, SEED, ds, split, fmt),
                [{"uid": u, "logit": float(s), "step_logits": None} for u, s in zip(uids, scores)])


def score_token_methods(samples):
    out = {"mcp": [], "ppl": [], "mte": []}
    for s in samples:
        c0, c1 = s.conclusion["tok"]
        ll, ent = s.feat["ll"].float().numpy(), s.feat["ent"].float().numpy()
        out["mcp"].append(float(np.exp(ll[c0:c1].sum())))
        out["ppl"].append(float(ll.mean()))
        out["mte"].append(float(-ent.mean()))
    return out


def score_ccp(samples, tokenizer, nli):
    """Claim-conditioned probability (Fadeeva et al., 2024) of the conclusion claim."""
    pairs, owners = [], []
    for si, s in enumerate(samples):
        ids = s.feat["concl_ids"].tolist()
        top_ids = s.feat["concl_topk_ids"].tolist()
        top_lp = s.feat["concl_topk_lp"].float().numpy()
        original = tokenizer.decode(ids, skip_special_tokens=True).strip()
        for t in range(len(ids)):
            for k in range(top_lp.shape[1]):
                alt = top_ids[t][k]
                p = float(np.exp(top_lp[t, k]))
                if alt == ids[t]:
                    owners.append((si, t, p, "self"))
                    pairs.append(None)
                    continue
                variant = tokenizer.decode(ids[:t] + [alt] + ids[t + 1:], skip_special_tokens=True).strip()
                owners.append((si, t, p, "alt"))
                pairs.append((f"{s.gen['question']} {original}", f"{s.gen['question']} {variant}"))
    real = [p for p in pairs if p is not None]
    probs = nli.probs(real) if real else np.zeros((0, 3))
    it = iter(probs)
    num = {}
    den = {}
    for (si, t, p, kind), pair in zip(owners, pairs):
        key = (si, t)
        if kind == "self":
            num[key] = num.get(key, 0.0) + p
            den[key] = den.get(key, 0.0) + p
            continue
        c, _, e = next(it)
        if e > 0.5:      # same meaning
            num[key] = num.get(key, 0.0) + p
            den[key] = den.get(key, 0.0) + p
        elif c > 0.5:    # different claim
            den[key] = den.get(key, 0.0) + p
    scores = []
    for si, s in enumerate(samples):
        n_tok = len(s.feat["concl_ids"])
        logp = sum(np.log(max(num.get((si, t), 0.0) / max(den.get((si, t), 1e-12), 1e-12), 1e-12)) for t in range(n_tok))
        scores.append(float(np.exp(logp)))
    return scores


def score_judge(samples, backbone, ds, split, fmt, judge):
    fb = index_by_uid(read_jsonl(judge_path(judge, "final_blind", backbone, ds, split, fmt)))
    st = index_by_uid(read_jsonl(judge_path(judge, "steps", backbone, ds, split, fmt)))
    jf, js = [], []
    for s in samples:
        r = fb.get(s.uid, {})
        # Verbalised probability is the main score (token P(true) saturates after judge reasoning);
        # P(true) breaks ties.
        pv, pt = r.get("p_verbal"), r.get("p_true")
        if pv is None and pt is None:
            jf.append(np.nan)
        else:
            jf.append((pv if pv is not None else pt) + 1e-3 * (pt if pt is not None else 0.0))
        v = st.get(s.uid, {}).get("value")
        js.append(float(np.mean(v)) if isinstance(v, list) and v else np.nan)
    return jf, js


@torch.inference_mode()
def score_rm(samples, rm, rm_tok, device, batch_size=16):
    scores = []
    for k in range(0, len(samples), batch_size):
        chunk = samples[k: k + batch_size]
        convs = [[{"role": "user", "content": s.gen["user_content"]},
                  {"role": "assistant", "content": s.gen["gen_text"]}] for s in chunk]
        texts = [rm_tok.apply_chat_template(c, tokenize=False) for c in convs]
        enc = rm_tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=4096).to(device)
        scores.extend(rm(**enc).logits[:, 0].float().cpu().tolist())
    return scores


def score_sampling(samples, backbone, ds, split, fmt, nli):
    path = gen_path(backbone, ds, split, fmt, tag="samples")
    smp = index_by_uid(read_jsonl(path))
    se, sc = [], []
    for s in samples:
        rec = smp.get(s.uid)
        if rec is None:
            se.append(np.nan)
            sc.append(np.nan)
            continue
        q = s.gen["question"]
        answers = [a["answer"] for a in rec["samples"]]
        greedy = s.gen["pred_answer"]
        # Semantic clustering by bidirectional entailment (Kuhn et al., 2023).
        clusters = []
        for a in answers:
            for c in clusters:
                pr = nli.probs([(f"{q} {c[0]}", f"{q} {a}"), (f"{q} {a}", f"{q} {c[0]}")])
                if pr[0, 2] > 0.5 and pr[1, 2] > 0.5:
                    c.append(a)
                    break
            else:
                clusters.append([a])
        freq = np.asarray([len(c) for c in clusters], dtype=np.float64) / len(answers)
        se.append(float((freq * np.log(freq)).sum()))  # = -entropy
        pr = nli.probs([(f"{q} {a}", f"{q} {greedy}") for a in answers])
        sc.append(float(1.0 - pr[:, 0].mean()))
    return se, sc


def score_cotuq(samples, backbone, ds, split, fmt, tokenizer):
    path = ARTIFACTS / "cotuq" / backbone / ds / fmt / f"{split}.jsonl"
    kw = index_by_uid(read_jsonl(path)) if path.exists() else {}
    out = []
    for s in samples:
        rec = kw.get(s.uid)
        ll = s.feat["ll"].float().numpy()
        gen_ids = s.gen["gen_ids"]
        if not rec or not rec.get("keywords"):
            out.append(float(np.exp(ll[s.conclusion["tok"][0]: s.conclusion["tok"][1]].mean())))
            continue
        num = den = 0.0
        text = s.gen["gen_text"]
        for item in rec["keywords"]:
            kw_text, w = item["keyword"], float(item["importance"])
            pos = text.find(kw_text)
            if pos < 0 or w <= 0:
                continue
            prefix = len(tokenizer(text[:pos], add_special_tokens=False)["input_ids"])
            n = max(1, len(tokenizer(kw_text, add_special_tokens=False)["input_ids"]))
            span = ll[prefix: min(len(gen_ids), prefix + n)]
            if span.size == 0:
                continue
            num += w * float(np.exp(span).mean())
            den += w
        out.append(num / den if den > 0 else float(np.exp(ll.mean())))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--methods", default="mcp,ppl,mte,ccp,judge_final,judge_step,rm")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--fmt", default="step")
    p.add_argument("--judge", default=MAIN_JUDGE)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--splits", default="validation,test")
    args = p.parse_args()
    methods = set(args.methods.split(","))
    device = "cuda"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONES[args.backbone].path)
    nli = None
    if methods & {"ccp", "se", "selfcheck"}:
        from cuq.nli import NLI
        nli = NLI(device)
    rm = rm_tok = None
    if "rm" in methods:
        from transformers import AutoModelForSequenceClassification
        rm_tok = AutoTokenizer.from_pretrained(RM_MODEL.path)
        rm_tok.padding_side = "right"
        rm = AutoModelForSequenceClassification.from_pretrained(RM_MODEL.path, dtype=torch.bfloat16, num_labels=1).to(device).eval()

    keys = ["ll", "ent", "concl_ids", "concl_topk_ids", "concl_topk_lp"]
    for ds in args.datasets.split(","):
        for split in args.splits.split(","):
            if not gen_path(args.backbone, ds, split, args.fmt).exists():
                continue
            if all(pred_path(m, args.backbone, SEED, ds, split, args.fmt).exists() for m in methods) and not args.overwrite:
                print(f"[skip] {args.backbone} {ds} {split}: all requested scores exist", flush=True)
                continue
            samples = load_samples(args.backbone, ds, split, args.fmt, feature_keys=keys)
            uids = [s.uid for s in samples]
            tm = score_token_methods(samples)
            for m in ("mcp", "ppl", "mte"):
                if m in methods:
                    _write(m, args.backbone, ds, split, args.fmt, uids, tm[m])
            if "ccp" in methods:
                _write("ccp", args.backbone, ds, split, args.fmt, uids, score_ccp(samples, tokenizer, nli))
            if methods & {"judge_final", "judge_step"}:
                jf, js = score_judge(samples, args.backbone, ds, split, args.fmt, args.judge)
                if "judge_final" in methods:
                    _write("judge_final", args.backbone, ds, split, args.fmt, uids, jf)
                if "judge_step" in methods:
                    _write("judge_step", args.backbone, ds, split, args.fmt, uids, js)
            if "rm" in methods:
                _write("rm", args.backbone, ds, split, args.fmt, uids, score_rm(samples, rm, rm_tok, device))
            if methods & {"se", "selfcheck"} and gen_path(args.backbone, ds, split, args.fmt, tag="samples").exists():
                se, sc = score_sampling(samples, args.backbone, ds, split, args.fmt, nli)
                _write("se", args.backbone, ds, split, args.fmt, uids, se)
                _write("selfcheck", args.backbone, ds, split, args.fmt, uids, sc)
            if "cotuq" in methods:
                _write("cotuq", args.backbone, ds, split, args.fmt, uids,
                       score_cotuq(samples, args.backbone, ds, split, args.fmt, tokenizer))
            print(f"[scores] {args.backbone} {ds} {split}: {len(samples)} samples", flush=True)


if __name__ == "__main__":
    main()
