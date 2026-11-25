"""Shared modules for the rewritten UQ heads."""
from __future__ import annotations

import torch
import torch.nn as nn


def resolve_heads(head_dim: int, requested_heads: int) -> int:
    for cand in (requested_heads, 8, 4, 2, 1):
        if cand > 0 and head_dim % cand == 0:
            return cand
    return 1


class ResidualMLP(nn.Module):
    def __init__(self, head_dim: int, dropout: float, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(head_dim)
        self.ffn = nn.Sequential(
            nn.Linear(head_dim, head_dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim * expansion, head_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.ffn(self.norm(x))


class CrossContextGate(nn.Module):
    def __init__(self, head_dim: int, dropout: float):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(head_dim * 3, head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_dim, head_dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(head_dim)

    def forward(self, claim_h: torch.Tensor, anchor_h: torch.Tensor) -> torch.Tensor:
        anchor_expand = anchor_h.unsqueeze(0).expand_as(claim_h)
        gate = self.gate(torch.cat([claim_h, anchor_expand, torch.abs(claim_h - anchor_expand)], dim=-1))
        fused = gate * claim_h + (1.0 - gate) * anchor_expand
        return self.norm(fused)


class DepthwiseConvMixer(nn.Module):
    def __init__(self, head_dim: int, dropout: float, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.LayerNorm(head_dim)
        self.dw = nn.Conv1d(
            in_channels=head_dim,
            out_channels=head_dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=head_dim,
        )
        self.pw = nn.Conv1d(head_dim, head_dim, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        self.mlp = ResidualMLP(head_dim=head_dim, dropout=dropout)
        nn.init.xavier_uniform_(self.dw.weight)
        nn.init.xavier_uniform_(self.pw.weight)
        nn.init.zeros_(self.dw.bias)
        nn.init.zeros_(self.pw.bias)

    def forward(self, claim_h: torch.Tensor) -> torch.Tensor:
        x = self.norm(claim_h).transpose(0, 1).unsqueeze(0)
        mixed = self.pw(self.dw(x)).squeeze(0).transpose(0, 1)
        claim_h = claim_h + self.dropout(torch.nn.functional.gelu(mixed))
        return self.mlp(claim_h)
