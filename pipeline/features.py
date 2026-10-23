#!/usr/bin/env python3
"""Stage 3: teacher-forced feature extraction from the frozen backbone (HF).

For every parsed generation (``parse_ok``) we run one forward pass over
``prompt_ids + gen_ids`` and store, per sample:

  tok        [T, d + 10 + 3H] bf16  per generated token: weighted fusion of
             hidden layers [-1,-2,-4,-8] (0.4/0.3/0.2/0.1), top-6 log-probs +
             4 summary stats of the distribution that generated the token, and
             last-layer attention from the token to itself and the two
             previous generated tokens (H heads x 3).
  ll, ent    [T] fp32  realised-token log-likelihood and predictive entropy.
  top1, margin [T] fp32  top-1 log-prob and top-1/top-2 margin.
  layers_anchor     [L+1, d] fp16  all hidden layers at the position that
             predicts the first conclusion token (Factoscope / SAPLMA input).
  layers_concl_last [L+1, d] fp16  all hidden layers at the last conclusion token.
  lens_rank [L], lens_topk_lp [L, 10], lens_topk_ids [L, 10]
             logit lens at the anchor position (Factoscope input).
  lookback  [n_spans, L*H] fp16  Lookback-Lens ratios averaged over the
             response, each step, and the conclusion span (in that order).
  concl_topk_ids / concl_topk_lp [n_c, 10]  generating distribution at each
             conclusion token (CCP input); concl_ids [n_c] realised tokens.

Two fixes relative to the original release code (engine/vllm_engine.py):
  * token log-likelihoods use the distribution at position t-1 for token t
    (the original indexed logits[t] for token t, i.e. off by one);
  * attention-history features index generated-token keys in absolute
    coordinates (the original used generation-relative indices as absolute
    key positions, i.e. it read attention to the first prompt tokens).

Example:
    CUDA_VISIBLE_DEVICES=2 python pipeline/features.py --backbone llama \
        --dataset hotpotqa --splits validation,test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq.io import read_jsonl
from cuq.registry import BACKBONES, DATASETS, feat_dir, gen_path

HIDDEN_LAYERS = (-1, -2, -4, -8)
HIDDEN_WEIGHTS = (0.4, 0.3, 0.2, 0.1)
TOPK_FEAT = 6
TOPK_STORE = 10
ATTN_HISTORY = 3


def resolve_layers(n_layers: int):
    # hidden_states has n_layers + 1 entries (embeddings first); -1 is the last layer.
    return [n_layers + 1 + l if l < 0 else l + 1 for l in HIDDEN_LAYERS]


@torch.inference_mode()
def extract_one_batch(model, batch, device, n_layers, n_heads, final_norm, lm_head):
    lens = [len(r["prompt_ids"]) + len(r["gen_ids"]) for r in batch]
    S = max(lens)
    pad_id = 0
    input_ids = torch.full((len(batch), S), pad_id, dtype=torch.long, device=device)
    attn_mask = torch.zeros((len(batch), S), dtype=torch.long, device=device)
    for i, r in enumerate(batch):
        ids = r["prompt_ids"] + r["gen_ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids, device=device)
        attn_mask[i, : len(ids)] = 1

    out = model(input_ids=input_ids, attention_mask=attn_mask, output_hidden_states=True,
                output_attentions=True, use_cache=False, return_dict=True)
    hs = out.hidden_states  # tuple(L+1) of [B, S, d]
    atts = out.attentions   # tuple(L) of [B, H, S, S]
    layer_idx = resolve_layers(n_layers)
    w = torch.tensor(HIDDEN_WEIGHTS, device=device, dtype=torch.float32)
    w = w / w.sum()

    results = []
    for i, r in enumerate(batch):
        P, T = len(r["prompt_ids"]), len(r["gen_ids"])
        gen = slice(P, P + T)
        prev = slice(P - 1, P + T - 1)  # distribution that generated token t
        fused = sum(w[j] * hs[l][i, gen].float() for j, l in enumerate(layer_idx))

        logits_prev = out.logits[i, prev].float()
        logp = torch.log_softmax(logits_prev, dim=-1)
        top_lp, top_ids = logp.topk(TOPK_STORE, dim=-1)
        gen_ids = input_ids[i, gen]
        ll = logp.gather(-1, gen_ids.unsqueeze(-1)).squeeze(-1)
        ent = -(logp.exp() * logp).sum(-1)
        top6 = top_lp[:, :TOPK_FEAT]
        tp_stats = torch.stack([top6[:, 0], top6[:, 0] - top6[:, 1], top6.mean(-1), top6.std(-1, unbiased=False)], -1)

        last_att = atts[-1][i]  # [H, S, S]
        q_pos = torch.arange(P, P + T, device=device)
        hist = []
        for k in range(ATTN_HISTORY):
            key_pos = q_pos - k
            valid = key_pos >= P
            vals = last_att[:, q_pos, key_pos.clamp(min=0)].T  # [T, H]
            hist.append(vals * valid.unsqueeze(-1))
        attn_feat = torch.cat(hist, dim=-1).float()

        tok = torch.cat([fused, top6, tp_stats, attn_feat], dim=-1).to(torch.bfloat16).cpu()

        # Lookback-Lens ratio per (layer, head) and generated token.
        key_idx = torch.arange(S, device=device).unsqueeze(0)
        new_mask = ((key_idx >= P) & (key_idx <= q_pos.unsqueeze(1))).float()  # [T, S]
        ratios = []
        for l in range(n_layers):
            a = atts[l][i, :, gen, :]  # [H, T, S]
            ctx = a[:, :, :P].float().mean(-1)
            new = (a.float() * new_mask).sum(-1) / new_mask.sum(-1).clamp(min=1.0)
            ratios.append(ctx / (ctx + new + 1e-12))  # [H, T]
        lb = torch.stack(ratios, 0).permute(2, 0, 1).reshape(T, -1)  # [T, L*H]
        spans = [(0, T)] + [tuple(s["tok"]) for s in r["steps"]] + [tuple(r["conclusion"]["tok"])]
        lookback = torch.stack([lb[a:b].mean(0) for a, b in spans], 0).to(torch.float16).cpu()

        cs, ce = r["conclusion"]["tok"]
        anchor = P + cs - 1
        last = P + ce - 1
        layers_anchor = torch.stack([h[i, anchor] for h in hs], 0).to(torch.float16).cpu()
        layers_concl_last = torch.stack([h[i, last] for h in hs], 0).to(torch.float16).cpu()

        target = int(input_ids[i, P + cs])
        lens_rank, lens_lp, lens_ids = [], [], []
        for l in range(1, n_layers + 1):
            h = hs[l][i, anchor]
            lg = out.logits[i, anchor].float() if l == n_layers else lm_head(final_norm(h)).float()
            lp_l = torch.log_softmax(lg, -1)
            tl, ti = lp_l.topk(TOPK_STORE)
            lens_rank.append(int((lp_l > lp_l[target]).sum()))
            lens_lp.append(tl)
            lens_ids.append(ti)

        results.append({
            "uid": r["uid"],
            "tok": tok,
            "ll": ll.cpu(), "ent": ent.cpu(), "top1": top_lp[:, 0].cpu(),
            "margin": (top_lp[:, 0] - top_lp[:, 1]).cpu(),
            "layers_anchor": layers_anchor,
            "layers_concl_last": layers_concl_last,
            "lens_rank": torch.tensor(lens_rank, dtype=torch.int32),
            "lens_topk_lp": torch.stack(lens_lp).to(torch.float16).cpu(),
            "lens_topk_ids": torch.stack(lens_ids).to(torch.int32).cpu(),
            "lookback": lookback,
            "concl_topk_ids": top_ids[cs:ce].to(torch.int32).cpu(),
            "concl_topk_lp": top_lp[cs:ce].to(torch.float16).cpu(),
            "concl_ids": gen_ids[cs:ce].to(torch.int32).cpu(),
        })
    del out, hs, atts
    return results


def batches_by_tokens(records, max_tokens):
    order = sorted(records, key=lambda r: len(r["prompt_ids"]) + len(r["gen_ids"]))
    batch, longest = [], 0
    for r in order:
        n = len(r["prompt_ids"]) + len(r["gen_ids"])
        if batch and max(longest, n) * (len(batch) + 1) > max_tokens:
            yield batch
            batch, longest = [], 0
        batch.append(r)
        longest = max(longest, n)
    if batch:
        yield batch


def save_shard(buf, path, retries=3):
    """Atomic shard write with retries (Lustre occasionally fails a large write)."""
    tmp = path.with_suffix(".pt.tmp")
    for attempt in range(retries):
        try:
            torch.save(buf, tmp)
            tmp.replace(path)
            return
        except RuntimeError as exc:
            tmp.unlink(missing_ok=True)
            if attempt == retries - 1:
                raise
            print(f"[warn] shard write failed ({exc!r:.80}); retrying", flush=True)
            time.sleep(10 * (attempt + 1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--splits", default="train,validation,test")
    p.add_argument("--fmt", default="step")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--batch_tokens", type=int, default=6000)
    p.add_argument("--shard_size", type=int, default=500)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    from transformers import AutoModelForCausalLM

    spec = BACKBONES[args.backbone]
    device = torch.device("cuda")
    model = AutoModelForCausalLM.from_pretrained(spec.path, dtype=torch.bfloat16, attn_implementation="eager").to(device)
    model.eval()
    cfg = model.config.get_text_config() if hasattr(model.config, "get_text_config") else model.config
    n_layers, n_heads = cfg.num_hidden_layers, cfg.num_attention_heads
    final_norm = model.model.norm
    lm_head = model.lm_head

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        out_dir = feat_dir(args.backbone, args.dataset, split, args.fmt)
        done_flag = out_dir / "DONE.json"
        if done_flag.exists() and not args.overwrite:
            print(f"[skip] {out_dir} done", flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        records = [r for r in read_jsonl(gen_path(args.backbone, args.dataset, split, args.fmt)) if r.get("parse_ok")]
        if args.max_samples:
            records = records[: args.max_samples]
        existing = set()
        for shard in sorted(out_dir.glob("shard_*.pt")):
            try:
                existing.update(torch.load(shard, weights_only=False).keys())
            except Exception as exc:  # partial write from an earlier crash: redo its samples
                print(f"[warn] removing unreadable {shard.name}: {exc!r:.80}", flush=True)
                shard.unlink()
        todo = [r for r in records if r["uid"] not in existing]
        shard_id = max([int(x.stem.split("_")[1]) + 1 for x in out_dir.glob("shard_*.pt")], default=0)
        buf = {}
        t0 = time.time()
        n_tokens = 0
        for batch in batches_by_tokens(todo, args.batch_tokens):
            for res in extract_one_batch(model, batch, device, n_layers, n_heads, final_norm, lm_head):
                buf[res.pop("uid")] = res
            n_tokens += sum(len(r["prompt_ids"]) + len(r["gen_ids"]) for r in batch)
            if len(buf) >= args.shard_size:
                save_shard(buf, out_dir / f"shard_{shard_id:04d}.pt")
                shard_id, buf = shard_id + 1, {}
        if buf:
            save_shard(buf, out_dir / f"shard_{shard_id:04d}.pt")
        stats = {
            "backbone": args.backbone, "dataset": args.dataset, "split": split, "fmt": args.fmt,
            "n_records": len(records), "n_new": len(todo), "seconds": round(time.time() - t0, 1),
            "tokens_forwarded": n_tokens, "n_layers": n_layers, "n_heads": n_heads,
            "hidden_size": cfg.hidden_size,
        }
        done_flag.write_text(json.dumps(stats, indent=2))
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
