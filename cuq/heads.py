"""Trainable Stage-1 heads: ChainUQ (+ ablations) and supervised baselines.

Every head maps one batch dict (see ``cuq.batching.collate``) to a [B] logit of
P(correct). ChainUQ and its ablations reuse the original classes in
``models/heads`` unchanged; the baselines are re-implemented here from their
papers inside the same feature pipeline (the original repository did not
contain them).

Width of each head is auto-sized to the ~2M trainable-parameter budget
(``size_to_budget``) except Lookback Lens, which is a logistic regression in
the original paper and is fit with scikit-learn in ``pipeline/train.py``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from cuq.chainuq_fast import batched_forward
from models.heads import HEAD_REGISTRY

PARAM_BUDGET = 2_000_000
CHAINUQ_FAMILY = {
    "chainuq": "chainuq",
    "chainuq-lb": "uq_abl_v1",   # -LB: no learned reliability bank
    "chainuq-cs": "uq_abl_v2",   # -CS: no conditioned selection
    "chainuq-cf": "uq_abl_v3",   # -CF: no context fusion
    "chainuq-fr": "uq_abl_v4",   # -FR: no final residual refiner
}


class ChainUQWrapper(nn.Module):
    """Adapter around the original ChainUQ heads (conclusion-only supervision)."""

    def __init__(self, registry_key: str, feature_dim: int, head_dim: int):
        super().__init__()
        self.head = HEAD_REGISTRY[registry_key](feature_dim=feature_dim, num_classes=1, head_dim=head_dim,
                                                n_layers=2, n_heads=8, dropout=0.1)

    def forward(self, batch, span_key: str = "concl_mask"):
        return batched_forward(self.head, batch["tok"], batch["attn"], batch[span_key])

    def reference_forward(self, batch, span_key: str = "concl_mask"):
        """Original per-sample implementation (used only by the equivalence test)."""
        masks = batch[span_key].unsqueeze(1)  # [B, 1, T]
        types = torch.ones(masks.shape[0], 1, dtype=torch.long, device=masks.device)
        logits = self.head(batch["tok"], batch["attn"], claim_masks=masks, claim_types=types)
        return logits.reshape(-1)


class UHead(nn.Module):
    """Token-level transformer uncertainty head (Shelmanov et al., 2025), pooled on the conclusion span."""

    def __init__(self, feature_dim: int, head_dim: int, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(feature_dim), nn.Linear(feature_dim, head_dim), nn.GELU())
        layer = nn.TransformerEncoderLayer(head_dim, n_heads, dim_feedforward=2 * head_dim, dropout=dropout,
                                           activation="gelu", batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layers)
        self.out = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, head_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(head_dim, 1))

    def forward(self, batch, span_key: str = "concl_mask"):
        h = self.enc(self.proj(batch["tok"]), src_key_padding_mask=~batch["attn"].bool())
        m = batch[span_key].unsqueeze(-1)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.out(pooled).reshape(-1)


class UHeadPCS(UHead):
    """UHead trained with privileged consistency supervision (same targets as ChainUQ's PCS head)."""

    def __init__(self, feature_dim: int, head_dim: int, n_layers: int = 2, n_heads: int = 8, dropout: float = 0.1):
        super().__init__(feature_dim, head_dim, n_layers, n_heads, dropout)
        self.step_type = nn.Parameter(torch.zeros(head_dim))
        self.verdict = nn.Linear(head_dim, 1)

    def _pooled(self, batch, span_key="concl_mask"):
        h = self.enc(self.proj(batch["tok"]), src_key_padding_mask=~batch["attn"].bool())
        m = batch[span_key].unsqueeze(-1)
        return (h * m).sum(1) / m.sum(1).clamp(min=1.0)

    def forward_all(self, batch):
        p = self._pooled(batch)
        return self.out(p).reshape(-1), self.verdict(p).reshape(-1)

    def step_logits(self, batch):
        return self.out(self._pooled(batch) + self.step_type).reshape(-1)


class SpanPoolMLP(nn.Module):
    """E6 simple head: mean (or attention) pooling over the conclusion span + MLP."""

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
            a = torch.softmax(s, dim=-1).unsqueeze(-1)
            pooled = (a * x).sum(1)
        else:
            pooled = (x * m.unsqueeze(-1)).sum(1) / m.sum(1, keepdim=True).clamp(min=1.0)
        return self.mlp(pooled).reshape(-1)


class SAPLMA(nn.Module):
    """SAPLMA (Azaria & Mitchell, 2023): feed-forward probe on one layer's hidden state.

    Uses the last token of the conclusion statement; the layer is selected on
    validation AUROC in ``pipeline/train.py``.
    """

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
    """Factoscope (He et al., 2024), adapted to supervised BCE training.

    Four sub-encoders over the anchor position (the position that predicts the
    first conclusion token): per-layer activations, logit-lens rank of the
    realised token, logit-lens top-k probabilities, and top-k token ids.
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
        acts = self.act(batch["layers_anchor"].float())  # [B, L+1, 32]
        a = self.act_out(acts.flatten(1))
        r = self.rank(torch.log1p(batch["lens_rank"].float()))
        p = self.prob(batch["lens_topk_lp"].float().exp().flatten(1))
        ids = batch["lens_topk_ids"].long() % self.vocab_hash  # [B, L, 10]
        B, L, K = ids.shape
        e = self.ids(ids.reshape(B * L, K)).reshape(B, L * 16)
        i = self.ids_out(e)
        return self.cls(torch.cat([a, r, p, i], dim=-1)).reshape(-1)


class ChainUQ2(nn.Module):
    """ChainUQ-Head v2: conclusion-aligned layer branch + token branch, fused by a gate.

    Layer branch: per-layer LayerNorm over the hidden states at the anchor position (which
    predicts the first conclusion token) and at the last conclusion token, mixed by learned
    softmax weights over layers (SAPLMA-style mid-layer information without fixing a layer).
    Token branch: the original ChainUQ conclusion-aware encoder over the response tokens,
    returning its fused state instead of a logit.
    """

    def __init__(self, feature_dim: int, hidden_size: int, n_layers: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.token = HEAD_REGISTRY["chainuq"](feature_dim=feature_dim, num_classes=1, head_dim=head_dim,
                                              n_layers=2, n_heads=8, dropout=dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        self.layer_logits = nn.Parameter(torch.zeros(2, n_layers + 1))
        self.layer_proj = nn.Sequential(nn.Linear(2 * hidden_size, head_dim), nn.GELU(), nn.Dropout(dropout))
        self.gate = nn.Linear(2 * head_dim, head_dim)
        self.out = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, head_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(head_dim, 1))

    def layer_state(self, batch):
        x = torch.stack([batch["layers_anchor"], batch["layers_concl_last"]], dim=1).float()  # [B,2,L+1,d]
        x = self.layer_norm(x)
        w = torch.softmax(self.layer_logits, dim=-1)  # [2, L+1]
        mixed = (x * w[None, :, :, None]).sum(2)  # [B,2,d]
        return self.layer_proj(mixed.flatten(1))

    branch_dropout = 0.0

    def forward(self, batch, span_key: str = "concl_mask"):
        tok_h = batched_forward(self.token, batch["tok"], batch["attn"], batch[span_key], return_hidden=True)
        lay_h = self.layer_state(batch)
        g = torch.sigmoid(self.gate(torch.cat([tok_h, lay_h], -1)))
        if self.training and self.branch_dropout > 0:
            # Branch dropout: some examples see only the token branch (g=1) or only the layer branch (g=0),
            # so the gate cannot collapse onto the branch that fits the source domain best.
            r = torch.rand(g.shape[0], 1, device=g.device)
            g = torch.where(r < self.branch_dropout / 2, torch.ones_like(g),
                            torch.where(r < self.branch_dropout, torch.zeros_like(g), g))
        return self.out(g * tok_h + (1 - g) * lay_h).reshape(-1)


class ChainUQ2BD(ChainUQ2):
    """ChainUQ-Head v2 trained with branch dropout (p=0.5: 25% token-only, 25% layer-only)."""

    branch_dropout = 0.5


class ChainUQ2BDAblation(ChainUQ2BD):
    """Ablations of the final head: a token-branch module removed, or one branch only."""

    def __init__(self, feature_dim: int, hidden_size: int, n_layers: int, head_dim: int, token_key: str = "chainuq",
                 mode: str = "both", branch_dropout: float = 0.5, dropout: float = 0.1):
        super().__init__(feature_dim, hidden_size, n_layers, head_dim, dropout)
        self.mode = mode
        self.branch_dropout = branch_dropout
        if token_key != "chainuq":
            self.token = HEAD_REGISTRY[token_key](feature_dim=feature_dim, num_classes=1, head_dim=head_dim,
                                                  n_layers=2, n_heads=8, dropout=dropout)
        if mode != "both":
            self.gate = None
        if mode == "token":
            self.layer_logits, self.layer_proj = None, None
        if mode == "layer":
            self.token = None

    def forward(self, batch, span_key: str = "concl_mask"):
        if self.mode == "both":
            return super().forward(batch, span_key)
        if self.mode == "token":
            h = batched_forward(self.token, batch["tok"], batch["attn"], batch[span_key], return_hidden=True)
        else:
            h = self.layer_state(batch)
        return self.out(h).reshape(-1)


class ChainUQ2PCS(ChainUQ2BDAblation):
    """Final head trained with privileged consistency supervision (judge labels used only in training).

    Besides the answer logit, the token branch scores each reasoning step as a reasoning-type claim
    (target: the judge's answer-blind step label), and a linear output on the fused state predicts the
    judge's chain-level verdict. Neither output needs the judge at inference.
    """

    def __init__(self, feature_dim, hidden_size, n_layers, head_dim, dropout=0.1):
        super().__init__(feature_dim, hidden_size, n_layers, head_dim, token_key="uq_core", dropout=dropout)
        self.verdict = nn.Linear(head_dim, 1)

    def _fused(self, batch, span_key="concl_mask"):
        tok_h = batched_forward(self.token, batch["tok"], batch["attn"], batch[span_key], return_hidden=True)
        lay_h = self.layer_state(batch)
        g = torch.sigmoid(self.gate(torch.cat([tok_h, lay_h], -1)))
        if self.training and self.branch_dropout > 0:
            r = torch.rand(g.shape[0], 1, device=g.device)
            g = torch.where(r < self.branch_dropout / 2, torch.ones_like(g),
                            torch.where(r < self.branch_dropout, torch.zeros_like(g), g))
        return g * tok_h + (1 - g) * lay_h

    def forward(self, batch, span_key: str = "concl_mask"):
        return self.out(self._fused(batch, span_key)).reshape(-1)

    def forward_all(self, batch):
        h = self._fused(batch)
        return self.out(h).reshape(-1), self.verdict(h).reshape(-1)

    def branch_logits(self, batch):
        """Scores of the token-only and layer-only states (the output MLP is trained on both via branch dropout)."""
        tok_h = batched_forward(self.token, batch["tok"], batch["attn"], batch["concl_mask"], return_hidden=True)
        lay_h = self.layer_state(batch)
        return self.out(tok_h).reshape(-1), self.out(lay_h).reshape(-1)

    def step_logits(self, batch):
        h = batched_forward(self.token, batch["tok"], batch["attn"], batch["concl_mask"], return_hidden=True, claim_type=0)
        return self.out(h).reshape(-1)


CHAINUQ2_ABLATIONS = {
    "chainuq2bd-lb": dict(token_key="uq_abl_v1"),
    "chainuq2bd-cs": dict(token_key="uq_abl_v2"),
    "chainuq2bd-cf": dict(token_key="uq_abl_v3"),
    "chainuq2bd-fr": dict(token_key="uq_abl_v4"),
    "chainuq2bd-tok": dict(mode="token"),
    "chainuq2bd-lay": dict(mode="layer"),
    "chainuq2bd-core": dict(token_key="uq_core"),  # all four refinement modules (-LB -CS -CF -FR) removed
    "chainuq2bd-core-tok": dict(token_key="uq_core", mode="token"),
    "chainuq2-core": dict(token_key="uq_core", branch_dropout=0.0),
}


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def build(method: str, dims: dict, head_dim: int):
    F, d, L = dims["feature_dim"], dims["hidden_size"], dims["n_layers"]
    if method in CHAINUQ_FAMILY:
        return ChainUQWrapper(CHAINUQ_FAMILY[method], F, head_dim)
    if method == "uhead":
        return UHead(F, head_dim)
    if method == "uhead-pcs":
        return UHeadPCS(F, head_dim)
    if method == "meanpool-mlp":
        return SpanPoolMLP(F, head_dim)
    if method == "attnpool-mlp":
        return SpanPoolMLP(F, head_dim, attention_pool=True)
    if method.startswith("saplma"):
        return SAPLMA(d, dims["saplma_layer"], head_dim)
    if method == "factoscope":
        return Factoscope(d, L, head_dim)
    if method == "chainuq2":
        return ChainUQ2(F, d, L, head_dim)
    if method == "chainuq2bd":
        return ChainUQ2BD(F, d, L, head_dim)
    if method in CHAINUQ2_ABLATIONS:
        return ChainUQ2BDAblation(F, d, L, head_dim, **CHAINUQ2_ABLATIONS[method])
    if method in ("chainuq2pcs", "chainuq2pcs-steps", "chainuq2pcs-verdict") or method.startswith("chainuq2pcs-"):
        return ChainUQ2PCS(F, d, L, head_dim)
    raise ValueError(method)


def size_to_budget(method: str, dims: dict, budget: int = PARAM_BUDGET, lo: int = 32, hi: int = 1024, step: int = 8) -> int:
    """Largest width (multiple of ``step``; divisible by 8 heads) whose head stays within ``budget``."""
    best = lo
    for width in range(lo, hi + 1, step):
        if width % 8:
            continue
        n = count_params(build(method, dims, width))
        if n <= budget:
            best = width
        else:
            break
    return best


TRAINABLE = tuple(CHAINUQ_FAMILY) + ("uhead", "meanpool-mlp", "attnpool-mlp", "saplma", "factoscope", "chainuq2", "chainuq2bd", "chainuq2pcs", "uhead-pcs") + tuple(CHAINUQ2_ABLATIONS)
