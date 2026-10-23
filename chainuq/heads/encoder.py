"""Conclusion-aware token encoder: the token branch of the ChainUQ head.

The encoder marks the tokens of one claim span (the conclusion, or a reasoning
step), contextualizes all response tokens with a small transformer, and fuses a
response-level summary with a span state built from span pooling, span-position
statistics, and a claim-type embedding. The output is a ``[B, head_dim]`` state.
"""

from __future__ import annotations

import torch
import torch.nn as nn

CONCLUSION, REASONING = 1, 0  # claim-type ids


class ResidualMLP(nn.Module):
    def __init__(self, head_dim: int, dropout: float, expansion: int = 2):
        super().__init__()
        self.norm = nn.LayerNorm(head_dim)
        self.ffn = nn.Sequential(nn.Linear(head_dim, head_dim * expansion), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(head_dim * expansion, head_dim), nn.Dropout(dropout))

    def forward(self, x):
        return x + self.ffn(self.norm(x))


def _proj(d_in: int, d_out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_out), nn.LayerNorm(d_out), nn.GELU(), nn.Dropout(dropout))


def _masked_mean(x, m):
    m = m.to(x.dtype).unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


def _masked_max(x, m, fallback):
    mb = m.bool().unsqueeze(-1)
    pooled = x.masked_fill(~mb, torch.finfo(x.dtype).min).max(1).values
    return torch.where(torch.isfinite(pooled) & mb.any(1), pooled, fallback)


class ConclusionEncoder(nn.Module):
    def __init__(self, feature_dim: int, head_dim: int, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_norm = nn.LayerNorm(feature_dim)
        self.token_proj = _proj(feature_dim, head_dim, dropout)
        self.sample_gate = nn.Linear(head_dim * 2, head_dim)
        self.sample_summary_proj = _proj(head_dim * 2, head_dim, dropout)
        self.sample_refine = ResidualMLP(head_dim, dropout)
        self.claim_type_embedding = nn.Embedding(3, head_dim)
        self.conclusion_marker_embedding = nn.Embedding(2, head_dim)
        self.mask_stats_proj = _proj(4, head_dim, dropout)
        layer = nn.TransformerEncoderLayer(d_model=head_dim, nhead=n_heads, dim_feedforward=head_dim, dropout=dropout,
                                           activation="gelu", batch_first=True, norm_first=False)
        self.scope_encoder = nn.TransformerEncoder(layer, num_layers=max(1, n_layers))
        self.scope_norm = nn.LayerNorm(head_dim)
        self.conclusion_seed_proj = _proj(head_dim * 4, head_dim, dropout)
        # conclusion_refine, output_classifier and logit_temperature are not on the forward path; they are kept so
        # that the parameter count (hence the budget-matched width) and the released checkpoints stay unchanged.
        self.conclusion_refine = ResidualMLP(head_dim, dropout)
        self.final_gate = nn.Linear(head_dim * 4, head_dim)
        self.final_refine = ResidualMLP(head_dim, dropout)
        self.output_classifier = nn.Sequential(*_proj(head_dim, head_dim, dropout), nn.Linear(head_dim, 1))
        self.logit_temperature = nn.Parameter(torch.tensor(0.7))
        self.refine = ResidualMLP(head_dim, dropout, expansion=3)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        nn.init.normal_(self.claim_type_embedding.weight, std=0.02)
        nn.init.normal_(self.conclusion_marker_embedding.weight, std=0.02)

    def _summary(self, h, attn):
        """Gated mean/max summary of the response tokens."""
        mean_pool = _masked_mean(h, attn)
        max_pool = _masked_max(h, attn, mean_pool)
        gate = torch.sigmoid(self.sample_gate(torch.cat([mean_pool, max_pool], -1)))
        fused = gate * mean_pool + (1.0 - gate) * max_pool
        return self.sample_refine(self.sample_summary_proj(torch.cat([fused, torch.abs(mean_pool - max_pool)], -1)))

    def _span_stats(self, span, attn):
        """Relative length, log-length, center, and extent of the span within the response."""
        valid_len = attn.float().sum(1).clamp(min=1.0)
        span_len = span.float().sum(1).clamp(min=1.0)
        pos = torch.arange(span.shape[1], device=span.device, dtype=torch.float32).unsqueeze(0)
        active = span > 0
        big = torch.full_like(pos.expand_as(span), float("inf"))
        start = torch.where(active, pos.expand_as(span), big).min(1).values
        end = torch.where(active, pos.expand_as(span), -big).max(1).values
        has = active.any(1)
        start = torch.where(has, start, torch.zeros_like(start))
        end = torch.where(has, end, torch.zeros_like(end))
        center = torch.where(has, (pos * span.float()).sum(1) / span_len, torch.zeros_like(span_len))
        stats = torch.stack([span_len / valid_len, torch.log1p(span_len) / torch.log1p(valid_len),
                             center / (valid_len - 1.0).clamp(min=1.0),
                             (end - start + 1.0).clamp(min=1.0) / valid_len], dim=-1)
        return self.mask_stats_proj(stats.to(self.claim_type_embedding.weight.dtype))

    def forward(self, features, attn, span, claim_type: int = CONCLUSION):
        """features [B,T,F], attn [B,T] (0/1), span [B,T] (0/1 mask of the claim) -> state [B, head_dim]."""
        token_h = self.token_proj(self.input_norm(features))
        attn = attn.to(token_h.dtype)
        span = span.to(token_h.dtype) * attn
        type_h = self.claim_type_embedding.weight[claim_type]
        stats_h = self._span_stats(span, attn)

        marked = token_h + self.conclusion_marker_embedding(span.long().clamp(0, 1))
        encoded = self.scope_norm(self.scope_encoder(marked, src_key_padding_mask=~attn.bool()))
        scope_global = self._summary(encoded, attn)
        scope_mean = _masked_mean(encoded, span)
        scope_max = _masked_max(encoded, span, scope_mean)

        seed = self.conclusion_seed_proj(torch.cat(
            [scope_mean, scope_max, scope_global, stats_h + type_h + torch.abs(scope_mean - scope_global)], -1))
        span_h = self.refine(seed)
        gate = torch.sigmoid(self.final_gate(torch.cat(
            [scope_global, span_h, torch.abs(scope_global - span_h), scope_global * span_h], -1)))
        return self.final_refine(gate * span_h + (1.0 - gate) * scope_global)
