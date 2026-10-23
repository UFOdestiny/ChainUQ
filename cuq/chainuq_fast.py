"""Batched forward pass for the original ChainUQ heads.

``models/heads/uq_head_base.py`` loops over samples in Python, which is too slow
for 10 seeds x 5 variants x 3 backbones on an 8-core node. This module
re-implements the *same* computation over a padded batch, reusing the
original modules and parameters unchanged. ``tests/test_chainuq_fast.py``
checks that the batched and per-sample outputs match in eval mode.
"""

from __future__ import annotations

import torch

from models.heads.uq_head_abl_v2 import UQAblationHeadV2


def _masked_mean(x, m):
    m = m.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


def _masked_max(x, m, fallback):
    mb = m.bool().unsqueeze(-1)
    pooled = x.masked_fill(~mb, torch.finfo(x.dtype).min).max(1).values
    return torch.where(torch.isfinite(pooled) & mb.any(1), pooled, fallback)


def _sample_state(head, h, attn):
    mean_pool = _masked_mean(h, attn)
    max_pool = _masked_max(h, attn, mean_pool)
    gate = torch.sigmoid(head.sample_gate(torch.cat([mean_pool, max_pool], -1)))
    fused = gate * mean_pool + (1.0 - gate) * max_pool
    out = head.sample_summary_proj(torch.cat([fused, torch.abs(mean_pool - max_pool)], -1))
    return head.sample_refine(out)


def _mask_stats(head, cmask, attn):
    mask = cmask.float()
    valid = attn.float()
    valid_len = valid.sum(1).clamp(min=1.0)
    span_len = mask.sum(1).clamp(min=1.0)
    pos = torch.arange(mask.shape[1], device=mask.device, dtype=torch.float32).unsqueeze(0)
    active = mask > 0
    big = torch.full_like(pos.expand_as(mask), float("inf"))
    start = torch.where(active, pos.expand_as(mask), big).min(1).values
    end = torch.where(active, pos.expand_as(mask), -big).max(1).values
    has = active.any(1)
    start = torch.where(has, start, torch.zeros_like(start))
    end = torch.where(has, end, torch.zeros_like(end))
    center = torch.where(has, (pos * mask).sum(1) / span_len, torch.zeros_like(span_len))
    denom = (valid_len - 1.0).clamp(min=1.0)
    stats = torch.stack([
        span_len / valid_len,
        torch.log1p(span_len) / torch.log1p(valid_len),
        center / denom,
        (end - start + 1.0).clamp(min=1.0) / valid_len,
    ], dim=-1)
    return head.mask_stats_proj(stats.to(head.claim_type_embedding.weight.dtype))


def _refine_v2(head, conclusion_h, sample_h):
    # Static-prototype ablation, broadcast over the batch.
    prototype = head.prototype_bank.mean(dim=0).unsqueeze(0).expand_as(conclusion_h)
    temp = torch.clamp(head.proto_temperature, min=0.35, max=3.0)
    gate = torch.sigmoid(head.prototype_gate(torch.cat(
        [conclusion_h, prototype, sample_h, torch.abs(prototype - conclusion_h)], -1)) / temp)
    mixed = gate * conclusion_h + (1.0 - gate) * prototype
    mixed = head.context_gate(mixed.unsqueeze(0), 0.5 * (prototype + sample_h)).squeeze(0)
    return head.proto_residual(head.refine(mixed))


def batched_forward(head, features, attn, cmask, return_hidden=False, claim_type=1):
    """features [B,T,F], attn [B,T] (0/1), cmask [B,T] (0/1) -> logits [B] (or the fused state [B,h])."""
    token_h = head._project_tokens(features)
    attn = attn.to(token_h.dtype)
    cmask = cmask.to(token_h.dtype) * attn
    type_h = head.claim_type_embedding.weight[claim_type]  # 1: conclusion claim, 0: reasoning claim
    stats_h = _mask_stats(head, cmask, attn)

    marked = token_h + head.conclusion_marker_embedding(cmask.long().clamp(0, 1))
    encoded = head.scope_encoder(marked, src_key_padding_mask=~attn.bool())
    encoded = head.scope_norm(encoded)
    scope_global = _sample_state(head, encoded, attn)
    scope_mean = _masked_mean(encoded, cmask)
    scope_max = _masked_max(encoded, cmask, scope_mean)

    seed = head.conclusion_seed_proj(torch.cat(
        [scope_mean, scope_max, scope_global, stats_h + type_h + torch.abs(scope_mean - scope_global)], -1))
    if isinstance(head, UQAblationHeadV2):
        concl = _refine_v2(head, seed, scope_global)
    else:
        concl = head._refine_conclusion_state(seed, scope_global, None, None, 1)
    fused = head._fuse_sample_and_conclusion(scope_global, concl)
    if return_hidden:
        return fused
    logits = head._classify_state(fused).reshape(-1)

    # Samples with an empty conclusion span fall back to the global sample state (as in the original).
    empty = cmask.sum(1) <= 0
    if bool(empty.any()):
        fallback = head._classify_state(_sample_state(head, token_h, attn)).reshape(-1)
        logits = torch.where(empty, fallback, logits)
    return logits
