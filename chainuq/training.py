"""Training and prediction for the Stage-1 heads.

Recipe: AdamW (lr 1.5e-4, weight decay 0.05), 6% linear warm-up then cosine decay,
30 epochs, batch 128, class-balanced BCE with label smoothing 0.02, gradient
clipping at 1.0, bf16 autocast, early stopping on validation PR-AUC (patience 10).

PCS heads add the privileged consistency loss: BCE on the judge's chain verdict
(soft target) and class-balanced BCE on the answer-blind labels of two sampled
reasoning steps per response.
"""

from __future__ import annotations

import json
import math
import random
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from chainuq import heads as H
from chainuq.artifacts import ckpt_dir, pred_path, write_jsonl
from chainuq.metrics import auroc, prauc

LR, WEIGHT_DECAY, WARMUP, SMOOTHING = 1.5e-4, 0.05, 0.06, 0.02
PCS_STEPS_PER_RESPONSE = 2
STEP_POS_WEIGHT = 1 / 3  # about 95% of steps are judged valid: down-weight positives (the pos_weight floor)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_items(samples, device=None, verdicts: Optional[Dict[str, float]] = None) -> List[dict]:
    """Wrap samples for batching; with ``device`` the feature tensors are moved there once."""
    items = []
    for s in samples:
        feat = {k: v.to(device, non_blocking=True) for k, v in s.feat.items()} if device is not None else s.feat
        labels = s.step_labels if isinstance(s.step_labels, list) and len(s.step_labels) == len(s.steps) else None
        items.append({"uid": s.uid, "y": s.y, "feat": feat, "concl": tuple(s.conclusion["tok"]),
                      "steps": [tuple(x["tok"]) for x in s.steps], "step_labels": labels,
                      "verdict": (verdicts or {}).get(s.uid)})
    return items


def collate(items, device) -> dict:
    """Pad token features and build the attention and conclusion-span masks."""
    batch = {"y": torch.tensor([it["y"] for it in items], dtype=torch.float32, device=device)}
    f0 = items[0]["feat"]
    if "tok" in f0:
        seqs = [it["feat"]["tok"].to(device, non_blocking=True) for it in items]
        tok = torch.nn.utils.rnn.pad_sequence(seqs, batch_first=True).float()
        lens = torch.tensor([t.shape[0] for t in seqs], device=device)
        pos = torch.arange(tok.shape[1], device=device).unsqueeze(0)
        spans = torch.tensor([it["concl"] for it in items], device=device)
        batch.update(tok=tok, attn=(pos < lens.unsqueeze(1)).long(),
                     concl_mask=((pos >= spans[:, :1]) & (pos < spans[:, 1:])).float())
    for key in ("layers_concl_last", "layers_anchor", "lens_rank", "lens_topk_lp", "lens_topk_ids"):
        if key in f0:
            batch[key] = torch.stack([it["feat"][key] for it in items]).to(device)
    return batch


def balanced_bce(logits, y, pos_weight, smoothing=SMOOTHING):
    y = y * (1.0 - smoothing) + 0.5 * smoothing
    return (-y * F.logsigmoid(logits) * pos_weight - (1.0 - y) * F.logsigmoid(-logits)).mean()


def class_pos_weight(items) -> float:
    ys = np.array([it["y"] for it in items if it.get("labeled", True)])
    return float(np.clip((ys == 0).sum() / max((ys == 1).sum(), 1), 1 / 3, 3.0))


def _step_items(items, rng=None, n_steps=None):
    """Copies of the items whose span is one reasoning step (all steps, or ``n_steps`` sampled ones)."""
    flat, owner, labels = [], [], []
    for i, it in enumerate(items):
        idx = range(len(it["steps"])) if rng is None else rng.sample(range(len(it["steps"])), min(n_steps, len(it["steps"])))
        for j in idx:
            flat.append(dict(it, concl=it["steps"][j]))
            owner.append(i)
            labels.append(float(it["step_labels"][j]) if it["step_labels"] else None)
    return flat, owner, labels


def pcs_loss(model, chunk, v_logits, device, rng, weights):
    """Privileged consistency loss on a batch: verdict BCE + balanced step-label BCE (weights = (step, verdict))."""
    w_step, w_verdict = weights
    loss = v_logits.sum() * 0.0
    has_v = [i for i, it in enumerate(chunk) if it["verdict"] is not None] if w_verdict else []
    if has_v:
        target = torch.tensor([chunk[i]["verdict"] for i in has_v], device=device, dtype=torch.float32)
        loss = loss + w_verdict * F.binary_cross_entropy_with_logits(v_logits[has_v], target)
    labeled = [it for it in chunk if it["step_labels"]] if w_step else []
    flat, _, ys = _step_items(labeled, rng, PCS_STEPS_PER_RESPONSE)
    if flat:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            s_logits = model.step_logits(collate(flat, device)).float()
        loss = loss + w_step * balanced_bce(s_logits, torch.tensor(ys, device=device), STEP_POS_WEIGHT)
    return loss


@torch.no_grad()
def predict(model, items, device, batch_size=256, fn=None) -> np.ndarray:
    """Logits in fp32 (no bf16 ties), computed in length-sorted batches."""
    model.eval()
    fn = fn or model
    order = sorted(range(len(items)), key=lambda i: items[i]["feat"]["tok"].shape[0] if "tok" in items[i]["feat"] else 0)
    res = np.zeros(len(items), dtype=np.float64)
    for k in range(0, len(order), batch_size):
        idx = order[k: k + batch_size]
        res[idx] = fn(collate([items[i] for i in idx], device)).float().cpu().numpy()
    return res


@torch.no_grad()
def predict_verdict(model, items, device, batch_size=256) -> np.ndarray:
    model.eval()
    out = np.zeros(len(items))
    for k in range(0, len(items), batch_size):
        out[k: k + batch_size] = model.forward_all(collate(items[k: k + batch_size], device))[1].float().cpu().numpy()
    return out


def predict_steps(model, items, device) -> List[List[float]]:
    """One logit per reasoning step of every response."""
    flat, owner, _ = _step_items(items)
    logits = predict(model, flat, device, fn=getattr(model, "step_logits", None)) if flat else np.zeros(0)
    per = [[] for _ in items]
    for i, v in zip(owner, logits):
        per[i].append(float(v))
    return per


def prediction_records(model, items, device, with_steps=True) -> List[dict]:
    """The per-response prediction format shared by every method: logit, step logits, predicted verdict."""
    logits = predict(model, items, device)
    steps = predict_steps(model, items, device) if with_steps else [None] * len(items)
    vhat = predict_verdict(model, items, device) if hasattr(model, "verdict") else [None] * len(items)
    return [{"uid": it["uid"], "logit": float(l), "step_logits": st, **({} if vh is None else {"verdict_hat": float(vh)})}
            for it, l, st, vh in zip(items, logits, steps, vhat)]


def train_one(method, dims, width, train_items, val_items, seed, device, epochs=30, batch_size=128, patience=10):
    """Train one head; returns the best-validation-PR-AUC model and a summary of the run."""
    set_seed(seed)
    spec = H.HEADS[method]
    model = H.build(method, dims, width).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total = epochs * math.ceil(len(train_items) / batch_size)
    warm = max(1, int(WARMUP * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))
    pos_weight = class_pos_weight(train_items)
    y_val = np.array([it["y"] for it in val_items])

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
                if spec.pcs:
                    logits, v_logits = (x.float() for x in model.forward_all(batch))
                else:
                    logits = model(batch).float()
            lab = [i for i, it in enumerate(chunk) if it.get("labeled", True)]  # label-efficiency runs mask some answers
            loss = balanced_bce(logits[lab], batch["y"][lab], pos_weight) if lab else logits.sum() * 0.0
            if spec.pcs:
                loss = loss + pcs_loss(model, chunk, v_logits, device, rng, spec.pcs)
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
    return model, {"n_params": H.count_params(model), "width": width, "best_val_prauc": best,
                   "epochs_run": len(history), "train_seconds": round(time.time() - t0, 1),
                   "pos_weight": pos_weight, "history": history}


def save_trained(model, info: dict, run: str, backbone: str, seed: int):
    out = ckpt_dir(run, backbone, seed)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "model.pt")
    (out / "meta.json").write_text(json.dumps(info, indent=2, default=str))


def load_trained(run: str, backbone: str, seeds: Iterable[int], device, method: Optional[str] = None) -> dict:
    """seed -> model for every finished seed of a training run (``method`` names the architecture if the run
    name carries a suffix, e.g. ``chainuq-lf10``)."""
    models = {}
    for sd in seeds:
        out = ckpt_dir(run, backbone, sd)
        if not ((out / "meta.json").exists() and (out / "model.pt").exists()):
            continue
        meta = json.loads((out / "meta.json").read_text())
        m = H.build(method or run, meta["dims"], meta["width"]).to(device)
        m.load_state_dict(torch.load(out / "model.pt", map_location=device))
        models[sd] = m.eval()
    return models


def write_predictions(run, backbone, models: dict, items, dataset, split, device, with_steps=True):
    for sd, model in models.items():
        write_jsonl(pred_path(run, backbone, sd, dataset, split), prediction_records(model, items, device, with_steps))
