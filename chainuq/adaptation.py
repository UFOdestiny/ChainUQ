"""Label-free target adaptation of a trained PCS head.

The head is fine-tuned on source batches (answer loss + PCS loss, as in training)
interleaved with batches from an unlabeled pool of target-domain responses. Target
responses contribute only the judge's answer-blind consistency targets; their
correctness labels and reference answers are never read.

Target objectives (``mode``):
  pcs     step labels and chain verdict as PCS targets (the method)
  steps   step labels only
  pseudo  the chain verdict used directly as a soft pseudo-label of the answer output (control)
  both    pcs + pseudo
With ``lam = 0`` the same updates use source batches only (source-only control).
"""

from __future__ import annotations

import copy
import random

import torch
import torch.nn.functional as F

from chainuq import heads as H
from chainuq import registry
from chainuq.training import balanced_bce, class_pos_weight, collate, pcs_loss

MODES = ("pcs", "steps", "pseudo", "both")


def adapt(model, src, tgt, device, steps=300, lr=5e-5, lam=1.0, seed=2026, batch=128, mode="pcs"):
    """Return an adapted copy of ``model``; ``src`` / ``tgt`` are item lists (``chainuq.training.to_items``)."""
    assert mode in MODES, mode
    model = copy.deepcopy(model).train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05)
    pos_weight = class_pos_weight(src)
    weights = H.HEADS[registry.FINAL_HEAD].pcs
    rng = random.Random(seed)
    for _ in range(steps):
        sc = rng.sample(src, batch)
        tc = rng.sample(tgt, min(batch, len(tgt)))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, v = model.forward_all(collate(sc, device))
            lt, vt = model.forward_all(collate(tc, device))
        y = torch.tensor([it["y"] for it in sc], device=device, dtype=torch.float32)
        loss = balanced_bce(logits.float(), y, pos_weight) + pcs_loss(model, sc, v.float(), device, rng, weights)
        if mode in ("pcs", "both"):
            loss = loss + lam * pcs_loss(model, tc, vt.float(), device, rng, weights)
        if mode == "steps":
            loss = loss + lam * pcs_loss(model, tc, vt.float(), device, rng, (weights[0], 0.0))
        if mode in ("pseudo", "both"):
            has = [i for i, it in enumerate(tc) if it["verdict"] is not None]
            if has:
                target = torch.tensor([tc[i]["verdict"] for i in has], device=device, dtype=torch.float32)
                loss = loss + lam * F.binary_cross_entropy_with_logits(lt.float()[has], target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model.eval()
