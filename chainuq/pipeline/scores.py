"""Stage 4b: scores of the baselines that are not trained heads.

Each method writes the prediction format of ``pipeline.train`` (``{"uid", "logit", "step_logits"}``; ``logit``
holds the raw confidence score) under seed 2026 for every requested split.

  Tier A (no external model at inference):
    mcp          probability of the conclusion claim (product of its token probabilities)
    ppl          mean token log-likelihood of the response (negative log-perplexity)
    mte          negative mean token entropy
    ccp          claim-conditioned probability of the conclusion (NLI over top-10 token alternatives)
    cotuq        CoT-UQ, probability-aggregation variant (needs ``pipeline.cotuq``)
    se           negative semantic entropy over 10 temperature samples (needs ``generate --n_samples 10``)
    selfcheck    1 - mean NLI contradiction between the greedy answer and the samples
  Tier B (judge or reward model at inference):
    judge-final  the answer-blind judge's chain verdict (``final_blind`` task)
    judge-step   fraction of steps the answer-blind step judge marks valid
    rm           Skywork-Reward-V2 score of (task input, response)

Example:
    CUDA_VISIBLE_DEVICES=0 python -m chainuq.pipeline.scores --backbone llama --methods mcp,ppl,mte,ccp
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from chainuq.artifacts import cotuq_path, gen_path, index_by_uid, pred_path, read_jsonl, write_jsonl
from chainuq.config import HEAD_SEEDS
from chainuq.evaluation import verdict_score
from chainuq.registry import BACKBONES, DATASETS, MAIN_JUDGE, RM_MODEL, SCORED_BASELINES
from chainuq.samples import load_judge_task, load_samples

SEED = HEAD_SEEDS[0]


def _write(method, backbone, ds, split, uids, scores):
    write_jsonl(pred_path(method, backbone, SEED, ds, split),
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


def score_judge(samples, backbone, ds, split, judge):
    final = load_judge_task(judge, "final_blind", backbone, ds, split)
    steps = load_judge_task(judge, "steps", backbone, ds, split)
    jf = [verdict_score(final.get(s.uid, {})) for s in samples]
    js = []
    for s in samples:
        v = steps.get(s.uid, {}).get("value")
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


def score_sampling(samples, backbone, ds, split, nli):
    path = gen_path(backbone, ds, split, tag="samples")
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


def score_cotuq(samples, backbone, ds, split, tokenizer):
    path = cotuq_path(backbone, ds, split)
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
    p.add_argument("--methods", default=",".join(SCORED_BASELINES))
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--splits", default="validation,test")
    p.add_argument("--judge", default=MAIN_JUDGE)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()
    methods = set(args.methods.split(","))
    device = "cuda"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BACKBONES[args.backbone].path)
    nli = None
    if methods & {"ccp", "se", "selfcheck"}:
        from chainuq.nli import NLI
        nli = NLI(device)
    rm = rm_tok = None
    if "rm" in methods:
        from transformers import AutoModelForSequenceClassification
        rm_tok = AutoTokenizer.from_pretrained(RM_MODEL.path)
        rm_tok.padding_side = "right"
        rm = AutoModelForSequenceClassification.from_pretrained(RM_MODEL.path, dtype=torch.bfloat16, num_labels=1).to(device).eval()

    keys = ["ll", "ent", "concl_ids", "concl_topk_ids", "concl_topk_lp"]
    bb = args.backbone
    for ds in args.datasets.split(","):
        for split in args.splits.split(","):
            if not gen_path(bb, ds, split).exists():
                continue
            if all(pred_path(m, bb, SEED, ds, split).exists() for m in methods) and not args.overwrite:
                print(f"[skip] {bb} {ds} {split}: all requested scores exist", flush=True)
                continue
            samples = load_samples(bb, ds, split, feature_keys=keys)
            uids = [s.uid for s in samples]
            write = lambda m, scores: _write(m, bb, ds, split, uids, scores)
            tm = score_token_methods(samples)
            for m in ("mcp", "ppl", "mte"):
                if m in methods:
                    write(m, tm[m])
            if "ccp" in methods:
                write("ccp", score_ccp(samples, tokenizer, nli))
            if methods & {"judge-final", "judge-step"}:
                jf, js = score_judge(samples, bb, ds, split, args.judge)
                if "judge-final" in methods:
                    write("judge-final", jf)
                if "judge-step" in methods:
                    write("judge-step", js)
            if "rm" in methods:
                write("rm", score_rm(samples, rm, rm_tok, device))
            if methods & {"se", "selfcheck"} and gen_path(bb, ds, split, tag="samples").exists():
                se, sc = score_sampling(samples, bb, ds, split, nli)
                write("se", se)
                write("selfcheck", sc)
            if "cotuq" in methods:
                write("cotuq", score_cotuq(samples, bb, ds, split, tokenizer))
            print(f"[scores] {bb} {ds} {split}: {len(samples)} samples", flush=True)


if __name__ == "__main__":
    main()
