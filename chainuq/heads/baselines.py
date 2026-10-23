"""Supervised probe baselines, re-implemented on the same cached features as the ChainUQ head.

Every head maps a batch (``chainuq.training.collate``) to a ``[B]`` logit of P(correct).
Lookback Lens is a logistic regression in its paper and is fit with scikit-learn
in ``chainuq.pipeline.train``.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class UHead(nn.Module):
    """Token-level transformer uncertainty head (Shelmanov et al., 2025), pooled over the conclusion span."""

    def __init__(self, feature_dim: int, head_dim: int, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, head_dim), nn.GELU())
        layer = nn.TransformerEncoderLayer(head_dim, n_heads, dim_feedforward=2 * head_dim, dropout=dropout,
                                           activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, head_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(head_dim, 1))

    def pooled(self, batch, span_key: str = "concl_mask"):
        h = self.enc(self.proj(batch["tok"]), src_key_padding_mask=~batch["attn"].bool())
        m = batch[span_key].unsqueeze(-1)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward(self, batch, span_key: str = "concl_mask"):
        return self.out(self.pooled(batch, span_key)).reshape(-1)


class UHeadPCS(UHead):
    """UHead trained with the same privileged consistency targets as the ChainUQ head."""

    def __init__(self, feature_dim: int, head_dim: int, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1):
        super().__init__(feature_dim, head_dim, n_layers, n_heads, dropout)
        self.step_type = nn.Parameter(torch.zeros(head_dim))
        self.verdict = nn.Linear(head_dim, 1)

    def forward_all(self, batch):
        p = self.pooled(batch)
        return self.out(p).reshape(-1), self.verdict(p).reshape(-1)

    def step_logits(self, batch):
        return self.out(self.pooled(batch) + self.step_type).reshape(-1)


class SpanPoolMLP(nn.Module):
    """Mean (or attention) pooling over the conclusion span followed by an MLP."""

    def __init__(self, feature_dim: int, head_dim: int, attention_pool: bool = False, dropout: float = 0.1):
        super().__init__()
        self.attention_pool = attention_pool
        self.norm = nn.LayerNorm(feature_dim)
        if attention_pool:
            self.score = nn.Linear(feature_dim, 1)
        self.mlp = nn.Sequential(nn.Linear(feature_dim, head_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(head_dim, head_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(head_dim, 1))

    def forward(self, batch, span_key: str = "concl_mask"):
        x = self.norm(batch["tok"])
        m = batch[span_key]
        if self.attention_pool:
            s = self.score(x).squeeze(-1).masked_fill(m <= 0, -1e4)
            pooled = (torch.softmax(s, dim=-1).unsqueeze(-1) * x).sum(1)
        else:
            pooled = (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
        return self.mlp(pooled).reshape(-1)


class SAPLMA(nn.Module):
    """SAPLMA (Azaria & Mitchell, 2023): feed-forward probe on one layer's hidden state at the last
    conclusion token; the layer is selected on validation AUROC."""

    def __init__(self, hidden_size: int, layer: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layer = layer
        self.mlp = nn.Sequential(nn.Linear(hidden_size, head_dim), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(head_dim, head_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(head_dim // 2, head_dim // 4), nn.ReLU(),
                                 nn.Linear(head_dim // 4, 1))

    def forward(self, batch, span_key: str = "concl_mask"):
        return self.mlp(batch["layers_concl_last"][:, self.layer].float()).reshape(-1)


class Factoscope(nn.Module):
    """Factoscope (He et al., 2024), trained with BCE.

    Four sub-encoders over the position that predicts the first conclusion token: per-layer activations,
    logit-lens rank of the realized token, logit-lens top-k probabilities, and top-k token ids.
    """

    def __init__(self, hidden_size: int, n_layers: int, head_dim: int, vocab_hash: int = 32768, dropout: float = 0.1):
        super().__init__()
        self.vocab_hash = vocab_hash
        self.act = nn.Sequential(nn.LayerNorm(hidden_size), nn.Linear(hidden_size, 32), nn.GELU())
        self.act_out = nn.Sequential(nn.Linear(32 * (n_layers + 1), head_dim), nn.GELU(), nn.Dropout(dropout))
        self.rank = nn.Sequential(nn.Linear(n_layers, head_dim // 2), nn.GELU())
        self.prob = nn.Sequential(nn.Linear(n_layers * 10, head_dim // 2), nn.GELU())
        self.ids = nn.EmbeddingBag(vocab_hash, 16, mode="mean")
        self.ids_out = nn.Sequential(nn.Linear(16 * n_layers, head_dim // 2), nn.GELU())
        self.cls = nn.Sequential(nn.Linear(head_dim + 3 * (head_dim // 2), head_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(head_dim, 1))

    def forward(self, batch, span_key: str = "concl_mask"):
        a = self.act_out(self.act(batch["layers_anchor"].float()).flatten(1))  # [B, (L+1)*32] -> [B, h]
        r = self.rank(torch.log1p(batch["lens_rank"].float()))
        p = self.prob(batch["lens_topk_lp"].float().exp().flatten(1))
        ids = batch["lens_topk_ids"].long() % self.vocab_hash  # [B, L, 10]
        B, L, K = ids.shape
        i = self.ids_out(self.ids(ids.reshape(B * L, K)).reshape(B, L * 16))
        return self.cls(torch.cat([a, r, p, i], dim=-1)).reshape(-1)
