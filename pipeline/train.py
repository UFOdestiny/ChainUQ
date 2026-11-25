#!/usr/bin/env python3
"""Stage 4: train Stage-1 heads on HotpotQA and write per-sample predictions.

For every method and seed: train on HotpotQA train (label = main-judge answer
correctness), early-stop on HotpotQA validation PR-AUC, then predict the
conclusion logit s (and, for span heads, one logit per reasoning step, the
paper's l_k) on the validation and test splits of every dataset.

Training recipe (unchanged from the original jobs/common.sh): AdamW lr 1.5e-4,
weight decay 0.05, 6% warmup + cosine, 30 epochs, batch 128, balanced BCE
with label smoothing 0.02, early-stopping patience 10, bf16 autocast.

Example:
    CUDA_VISIBLE_DEVICES=2 python pipeline/train.py --backbone llama \
        --methods chainuq,uhead --seeds all
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq import heads as H
from cuq.io import write_jsonl
from cuq.metrics import auroc, prauc
from cuq.registry import (BACKBONES, DATASETS, FORMATS, HEAD_SEEDS, SOURCE_DATASET, ckpt_dir, feat_dir,
                          gen_path, pred_path)
from cuq.store import load_samples

TOK_METHODS = set(H.CHAINUQ_FAMILY) | {"uhead", "meanpool-mlp", "attnpool-mlp", "chainuq2", "chainuq2bd"} \
    | set(H.CHAINUQ2_ABLATIONS) | {m for m in ("chainuq2pcs", "chainuq2pcs-steps", "chainuq2pcs-verdict", "chainuq2pcs-l1",
                                               "chainuq2pcs-l2", "chainuq2pcs-all", "uhead-pcs")}
# Privileged consistency supervision: judge labels are training targets only. (step weight, verdict weight)
PCS_WEIGHTS = {"chainuq2pcs": (0.5, 0.5), "chainuq2pcs-steps": (0.5, 0.0), "chainuq2pcs-verdict": (0.0, 0.5),
               "chainuq2pcs-l1": (1.0, 1.0), "chainuq2pcs-l2": (2.0, 2.0), "chainuq2pcs-all": (0.5, 0.5),
               "uhead-pcs": (0.5, 0.5)}
PCS_METHODS = set(PCS_WEIGHTS)
PCS_STEPS = {"chainuq2pcs-all": 12}  # steps sampled per response (default 2; 12 = every step)
FEATURE_KEYS = {
    "tok": ["tok"],
    "saplma": ["layers_concl_last"],
    "factoscope": ["layers_anchor", "lens_rank", "lens_topk_lp", "lens_topk_ids"],
    "lookback": ["lookback"],
}


def feature_keys_for(method: str):
    if method in ("chainuq2", "chainuq2bd") or method in H.CHAINUQ2_ABLATIONS or (method in PCS_METHODS and method.startswith("chainuq")):
        return ["tok", "layers_anchor", "layers_concl_last"]
    if method in TOK_METHODS:
        return FEATURE_KEYS["tok"]
    if method.startswith("saplma"):
        return FEATURE_KEYS["saplma"]
    return FEATURE_KEYS[method]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_verdicts(backbone, ds, split, fmt="step"):
    """Answer-blind chain-level verdict of the main judge in [0, 1] (training target for PCS heads only)."""
    from cuq.io import read_jsonl
    from cuq.registry import MAIN_JUDGE, judge_path
    path = judge_path(MAIN_JUDGE, "final_blind", backbone, ds, split, fmt)
    out = {}
    for r in (read_jsonl(path) if path.exists() else []):
        pv, pt = r.get("p_verbal"), r.get("p_true")
        if pv is not None or pt is not None:
            out[r["uid"]] = float(min(1.0, max(0.0, pv if pv is not None else pt)))
    return out


def to_items(samples, span_override=None, device=None, verdicts=None):
    """Wrap samples; with ``device`` the feature tensors are moved there once (GPU-resident batches)."""
    items = []
    for s in samples:
        feat = {k: v.to(device, non_blocking=True) for k, v in s.feat.items()} if device is not None else s.feat
        labels = s.step_labels if isinstance(s.step_labels, list) and len(s.step_labels) == len(s.steps) else None
        it = {"uid": s.uid, "y": s.y, "feat": feat, "concl": tuple(s.conclusion["tok"]),
              "steps": [tuple(x["tok"]) for x in s.steps], "step_labels": labels,
              "verdict": (verdicts or {}).get(s.uid)}
        if span_override is not None:
            it["concl"] = span_override(s)
        items.append(it)
    return items


def collate(items, device):
    batch = {"y": torch.tensor([it["y"] for it in items], dtype=torch.float32, device=device)}
    f0 = items[0]["feat"]
    if "tok" in f0:
        seqs = [it["feat"]["tok"].to(device, non_blocking=True) for it in items]
        tok = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True).float()
        lens = torch.tensor([t.shape[0] for t in seqs], device=device)
        pos = torch.arange(tok.shape[1], device=device).unsqueeze(0)
        attn = (pos < lens.unsqueeze(1)).long()
        spans = torch.tensor([it["concl"] for it in items], device=device)
        cm = ((pos >= spans[:, :1]) & (pos < spans[:, 1:])).float()
        batch.update(tok=tok, attn=attn, concl_mask=cm)
    for key in ("layers_concl_last", "layers_anchor", "lens_rank", "lens_topk_lp", "lens_topk_ids"):
        if key in f0:
            batch[key] = torch.stack([it["feat"][key] for it in items]).to(device)
    return batch


def balanced_bce(logits, y, pos_weight, smoothing=0.02):
    y = y * (1.0 - smoothing) + 0.5 * smoothing
    pos = -y * torch.nn.functional.logsigmoid(logits) * pos_weight
    neg = -(1.0 - y) * torch.nn.functional.logsigmoid(-logits)
    return (pos + neg).mean()


@torch.no_grad()
def predict(model, items, device, batch_size=256):
    model.eval()
    out = []
    order = sorted(range(len(items)), key=lambda i: items[i]["feat"]["tok"].shape[0] if "tok" in items[i]["feat"] else 0)
    for k in range(0, len(order), batch_size):
        idx = order[k: k + batch_size]
        logits = model(collate([items[i] for i in idx], device)).float().cpu().numpy()  # fp32: no bf16 ties
        out.extend(zip(idx, logits))
    res = np.zeros(len(items), dtype=np.float64)
    for i, v in out:
        res[i] = v
    return res


@torch.no_grad()
def predict_verdict(model, items, device, batch_size=256):
    model.eval()
    out = np.zeros(len(items))
    for k in range(0, len(items), batch_size):
        _, v = model.forward_all(collate(items[k: k + batch_size], device))
        out[k: k + batch_size] = v.float().cpu().numpy()
    return out


@torch.no_grad()
def predict_branches(model, items, device, batch_size=256):
    model.eval()
    tok, lay = np.zeros(len(items)), np.zeros(len(items))
    for k in range(0, len(items), batch_size):
        a, b = model.branch_logits(collate(items[k: k + batch_size], device))
        tok[k: k + batch_size], lay[k: k + batch_size] = a.float().cpu().numpy(), b.float().cpu().numpy()
    return tok, lay


def predict_steps(model, items, device):
    flat, owner = [], []
    for i, it in enumerate(items):
        for span in it["steps"]:
            flat.append(dict(it, concl=span))
            owner.append(i)
    if hasattr(model, "step_logits"):  # PCS heads score steps as reasoning-type claims
        class _Steps(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, batch):
                return self.m.step_logits(batch)
        logits = predict(_Steps(model), flat, device) if flat else np.zeros(0)
    else:
        logits = predict(model, flat, device) if flat else np.zeros(0)
    per = [[] for _ in items]
    for i, v in zip(owner, logits):
        per[i].append(float(v))
    return per


def pcs_aux_loss(model, chunk, v_logits, device, rng, weights, n_steps=2):
    """Privileged supervision: judge verdict (soft target) and answer-blind step labels of sampled steps."""
    PCS_STEP_WEIGHT, PCS_VERDICT_WEIGHT = weights
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    loss = v_logits.sum() * 0.0
    has_v = [i for i, it in enumerate(chunk) if it["verdict"] is not None] if PCS_VERDICT_WEIGHT else []
    if has_v:
        target = torch.tensor([chunk[i]["verdict"] for i in has_v], device=device, dtype=torch.float32)
        loss = loss + PCS_VERDICT_WEIGHT * bce(v_logits[has_v], target)
    flat, ys = [], []
    for it in (chunk if PCS_STEP_WEIGHT else []):
        if it["step_labels"]:
            for j in rng.sample(range(len(it["steps"])), min(n_steps, len(it["steps"]))):
                flat.append(dict(it, concl=it["steps"][j]))
                ys.append(float(it["step_labels"][j]))
    if flat:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_logits = model.step_logits(collate(flat, device)).float()
        # ~95% of steps are judged consistent: down-weight positives as the answer loss does (pos_weight floor 1/3).
        loss = loss + PCS_STEP_WEIGHT * balanced_bce(s_logits, torch.tensor(ys, device=device), 1 / 3)
    return loss


def train_one(method, dims, width, train_items, val_items, seed, device, epochs, batch_size, patience):
    set_seed(seed)
    model = H.build(method, dims, width).to(device)
    n_params = H.count_params(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=0.05)
    steps_per_epoch = math.ceil(len(train_items) / batch_size)
    total = epochs * steps_per_epoch
    warm = max(1, int(0.06 * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    ys = np.array([it["y"] for it in train_items])
    pos_weight = float(np.clip((ys == 0).sum() / max((ys == 1).sum(), 1), 1 / 3, 3.0))
    y_val = np.array([it["y"] for it in val_items])
    l2 = 0.01 if method in H.CHAINUQ_FAMILY else 0.0

    best, best_state, bad, history = -1.0, None, 0, []
    rng = random.Random(seed)
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        order = list(range(len(train_items)))
        rng.shuffle(order)
        losses = []
        for k in range(0, len(order), batch_size):
            chunk = [train_items[i] for i in order[k: k + batch_size]]
            batch = collate(chunk, device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if method in PCS_METHODS:
                    logits, v_logits = model.forward_all(batch)
                    logits, v_logits = logits.float(), v_logits.float()
                else:
                    logits = model(batch).float()
            loss = balanced_bce(logits, batch["y"], pos_weight)
            if method in PCS_METHODS:
                loss = loss + pcs_aux_loss(model, chunk, v_logits, device, rng, PCS_WEIGHTS[method], PCS_STEPS.get(method, 2))
            if l2:
                loss = loss + l2 * logits.pow(2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            losses.append(float(loss.detach()))
        val_logits = predict(model, val_items, device)
        score = prauc(y_val, val_logits)
        history.append({"epoch": epoch, "loss": float(np.mean(losses)), "val_prauc": score,
                        "val_auroc": auroc(y_val, val_logits)})
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, {"n_params": n_params, "width": width, "best_val_prauc": best, "epochs_run": len(history),
                   "train_seconds": round(time.time() - t0, 1), "pos_weight": pos_weight, "history": history}


EVAL_SPLITS = ("validation", "test", "test2")  # test2 = fresh held-out set (built after two looks at test)


def eval_cells(backbone, datasets, fmts):
    for ds in datasets:
        for split in EVAL_SPLITS:
            for fmt in fmts:
                if fmt != "step" and split not in ("test",):
                    continue
                if (feat_dir(backbone, ds, split, fmt) / "DONE.json").exists() and gen_path(backbone, ds, split, fmt).exists():
                    yield ds, split, fmt


def write_preds_all(method, backbone, models, keys, device, datasets, fmts, with_steps, force=False):
    """Predict every evaluation cell with every seed's model; each cell is loaded once, straight to the GPU.

    ``models`` maps seed -> model. Cells whose prediction file already exists are skipped unless ``force``.
    """
    for ds, split, fmt in eval_cells(backbone, datasets, fmts):
        todo = {sd: m for sd, m in models.items()
                if force or not pred_path(method, backbone, sd, ds, split, fmt).exists()}
        if not todo:
            continue
        samples = load_samples(backbone, ds, split, fmt, feature_keys=keys, device=device)
        items = to_items(samples, device=device)
        for sd, model in todo.items():
            logits = predict(model, items, device)
            steps = predict_steps(model, items, device) if with_steps else [None] * len(items)
            vhat = predict_verdict(model, items, device) if hasattr(model, "forward_all") else [None] * len(items)
            btok, blay = predict_branches(model, items, device) if hasattr(model, "branch_logits") else ([None] * len(items),) * 2
            write_jsonl(pred_path(method, backbone, sd, ds, split, fmt),
                        [{"uid": it["uid"], "logit": float(l), "step_logits": st,
                          **({} if vh is None else {"verdict_hat": float(vh)}),
                          **({} if bt is None else {"branch_tok": float(bt), "branch_lay": float(bl)})}
                         for it, l, st, vh, bt, bl in zip(items, logits, steps, vhat, btok, blay)])
        del samples, items
        torch.cuda.empty_cache()


def load_trained(method, backbone, seeds, device):
    models = {}
    for sd in seeds:
        out = ckpt_dir(method, backbone, sd)
        if not ((out / "meta.json").exists() and (out / "model.pt").exists()):
            continue
        meta = json.loads((out / "meta.json").read_text())
        m = H.build(method, meta["dims"], meta["width"]).to(device)
        m.load_state_dict(torch.load(out / "model.pt", map_location=device))
        models[sd] = m.eval()
    return models


def dims_for(backbone, samples):
    info = json.loads((feat_dir(backbone, SOURCE_DATASET, "train") / "DONE.json").read_text())
    f = samples[0].feat
    return {"feature_dim": int(f["tok"].shape[1]) if "tok" in f else 0, "hidden_size": info["hidden_size"],
            "n_layers": info["n_layers"]}


def run_lookback(backbone, args, device):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    keys = FEATURE_KEYS["lookback"]
    tr = load_samples(backbone, SOURCE_DATASET, "train", feature_keys=keys, require_steps=False)
    va = load_samples(backbone, SOURCE_DATASET, "validation", feature_keys=keys)
    if args.max_train:
        tr = tr[: args.max_train]
    X = lambda ss: np.stack([s.feat["lookback"][0].float().numpy() for s in ss])
    scaler = StandardScaler().fit(X(tr))
    best = None
    for C in (0.01, 0.1, 1.0):
        clf = LogisticRegression(C=C, max_iter=2000).fit(scaler.transform(X(tr)), [s.y for s in tr])
        a = auroc([s.y for s in va], clf.decision_function(scaler.transform(X(va))))
        if best is None or a > best[0]:
            best = (a, C, clf)
    seed = HEAD_SEEDS[0]
    for ds, split, fmt in eval_cells(backbone, args.eval_datasets, args.eval_fmts):
        ss = load_samples(backbone, ds, split, fmt, feature_keys=keys)
        logits = best[2].decision_function(scaler.transform(X(ss)))
        write_jsonl(pred_path("lookback", backbone, seed, ds, split, fmt),
                    [{"uid": s.uid, "logit": float(l), "step_logits": None} for s, l in zip(ss, logits)])
    out = ckpt_dir("lookback", backbone, seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps({"C": best[1], "val_auroc": best[0],
                                               "n_params": int(best[2].coef_.size + 1)}, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--methods", default="chainuq")
    p.add_argument("--seeds", default="all")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--max_train", type=int, default=0)
    p.add_argument("--eval_datasets", default=",".join(DATASETS))
    p.add_argument("--eval_fmts", default=",".join(FORMATS))
    p.add_argument("--predict_only", action="store_true",
                   help="reload saved checkpoints and rewrite predictions for --eval_datasets/--eval_fmts only")
    args = p.parse_args()
    args.eval_datasets = [d for d in args.eval_datasets.split(",") if d]
    args.eval_fmts = [f for f in args.eval_fmts.split(",") if f]
    seeds = HEAD_SEEDS if args.seeds == "all" else [int(s) for s in args.seeds.split(",")]
    device = torch.device("cuda")

    for method in [m for m in args.methods.split(",") if m]:
        if method == "lookback":
            run_lookback(args.backbone, args, device)  # deterministic refit; writes only the requested cells
            continue
        keys = feature_keys_for(method)
        with_steps = method in TOK_METHODS
        if args.predict_only:
            models = load_trained(method, args.backbone, seeds, device)
            write_preds_all(method, args.backbone, models, keys, device, args.eval_datasets, args.eval_fmts,
                            with_steps, force=True)
            print(json.dumps({"predict_only": method, "backbone": args.backbone, "seeds": sorted(models)}), flush=True)
            continue

        to_train = [sd for sd in seeds if not (ckpt_dir(method, args.backbone, sd) / "meta.json").exists()]
        for sd in seeds:
            if sd not in to_train:
                print(f"[skip] {method} {args.backbone} seed {sd}", flush=True)
        if to_train:
            # Features go straight to the GPU, shard by shard (Slurm job memory is capped at 120 GB).
            tr = load_samples(args.backbone, SOURCE_DATASET, "train", feature_keys=keys, require_steps=False, device=device)
            va = load_samples(args.backbone, SOURCE_DATASET, "validation", feature_keys=keys, device=device)
            if args.max_train:
                tr = tr[: args.max_train]
            vt = load_verdicts(args.backbone, SOURCE_DATASET, "train") if method in PCS_METHODS else None
            train_items, val_items = to_items(tr, device=device, verdicts=vt), to_items(va, device=device)
            dims = dims_for(args.backbone, tr)
            del tr, va

            if method == "saplma":
                L = dims["n_layers"]
                cands = sorted({L // 2, (2 * L) // 3, (3 * L) // 4, L})
                scores = {}
                for layer in cands:
                    d = dict(dims, saplma_layer=layer)
                    w = H.size_to_budget("saplma", d)
                    m, info = train_one("saplma", d, w, train_items, val_items, seeds[0], device,
                                        args.epochs, args.batch_size, args.patience)
                    scores[layer] = auroc([it["y"] for it in val_items], predict(m, val_items, device))
                dims["saplma_layer"] = max(scores, key=scores.get)
                dims["saplma_layer_scores"] = scores

            width = H.size_to_budget(method, dims)
            for seed in to_train:
                model, info = train_one(method, dims, width, train_items, val_items, seed, device,
                                        args.epochs, args.batch_size, args.patience)
                out = ckpt_dir(method, args.backbone, seed)
                out.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), out / "model.pt")
                info.update(dims={k: v for k, v in dims.items()}, n_train=len(train_items), n_val=len(val_items))
                (out / "meta.json").write_text(json.dumps(info, indent=2, default=str))
                print(json.dumps({"method": method, "backbone": args.backbone, "seed": seed,
                                  **{k: info[k] for k in ("n_params", "width", "best_val_prauc", "epochs_run", "train_seconds")}}),
                      flush=True)
            del train_items, val_items
            torch.cuda.empty_cache()

        # Predictions for every trained seed (also completes seeds whose earlier run died mid-prediction).
        models = load_trained(method, args.backbone, seeds, device)
        write_preds_all(method, args.backbone, models, keys, device, args.eval_datasets, args.eval_fmts, with_steps)
        print(json.dumps({"predicted": method, "backbone": args.backbone, "seeds": sorted(models)}), flush=True)


if __name__ == "__main__":
    main()
