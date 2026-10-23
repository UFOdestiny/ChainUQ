"""Shared helpers for claim-aware heads."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def claim_vectors_from_masks(features, attention_mask, claim_masks):
    out = []
    for i in range(features.shape[0]):
        token_feats = features[i]
        cm = prepare_claim_mask(
            claim_masks[i],
            seq_len=token_feats.shape[0],
            device=features.device,
            attention_mask_i=attention_mask[i] if attention_mask is not None else None,
        )
        denom = cm.sum(dim=1, keepdim=True).clamp(min=1.0)
        vecs = (cm @ token_feats) / denom
        out.append(vecs)
    return out


def safe_claim_mask(cm: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Align claim mask to feature sequence length."""
    if cm.shape[1] > seq_len:
        cm = cm[:, :seq_len]
    elif cm.shape[1] < seq_len:
        cm = F.pad(cm, (0, seq_len - cm.shape[1]), value=0.0)
    return cm


def prepare_claim_mask(
    claim_mask_i: torch.Tensor,
    seq_len: int,
    device: torch.device,
    attention_mask_i: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align one sample's claim mask and zero-out padding tokens."""
    cm = safe_claim_mask(claim_mask_i.to(device).float(), seq_len)
    if attention_mask_i is not None:
        cm = cm * attention_mask_i.to(device).float().unsqueeze(0)
    return cm

