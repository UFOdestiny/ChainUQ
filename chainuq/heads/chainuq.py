"""The ChainUQ head (Stage 1).

Two branches score the conclusion of a response from cached backbone features:

* token branch: the conclusion-aware encoder over the response tokens
  (``ConclusionEncoder``);
* layer branch: per-layer normalized hidden states at the position that predicts
  the first conclusion token and at the last conclusion token, mixed by learned
  softmax weights over layers.

A sigmoid gate fuses the two states. During training, branch dropout replaces
the gate by 1 (token only) or 0 (layer only) for a random half of the responses,
so the fusion cannot rely on the branch that fits the source domain best.

With privileged consistency supervision (PCS), the head also has a verdict output
on the fused state and scores each reasoning step as a reasoning-type claim; both
are trained on the judge's answer-blind step labels and chain verdict, which are
needed only during training.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from chainuq.heads.encoder import CONCLUSION, REASONING, ConclusionEncoder


class ChainUQHead(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int, n_layers: int, head_dim: int, token_branch: bool = True,
                 layer_branch: bool = True, branch_dropout: float = 0.5, pcs: bool = True, dropout: float = 0.1):
        super().__init__()
        self.branch_dropout, self.pcs = branch_dropout, pcs
        self.token = ConclusionEncoder(feature_dim, head_dim, dropout=dropout) if token_branch else None
        self.layer_norm = nn.LayerNorm(hidden_size, elementwise_affine=False)
        if layer_branch:
            self.layer_logits = nn.Parameter(torch.zeros(2, n_layers + 1))
            self.layer_proj = nn.Sequential(nn.Linear(2 * hidden_size, head_dim), nn.GELU(), nn.Dropout(dropout))
        else:
            self.layer_logits, self.layer_proj = None, None
        self.gate = nn.Linear(2 * head_dim, head_dim) if token_branch and layer_branch else None
        self.out = nn.Sequential(nn.LayerNorm(head_dim), nn.Linear(head_dim, head_dim), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(head_dim, 1))
        if pcs:
            self.verdict = nn.Linear(head_dim, 1)

    def layer_state(self, batch):
        x = torch.stack([batch["layers_anchor"], batch["layers_concl_last"]], dim=1).float()  # [B, 2, L+1, d]
        w = torch.softmax(self.layer_logits, dim=-1)  # [2, L+1]
        mixed = (self.layer_norm(x) * w[None, :, :, None]).sum(2)  # [B, 2, d]
        return self.layer_proj(mixed.flatten(1))

    def token_state(self, batch, span_key: str = "concl_mask", claim_type: int = CONCLUSION):
        return self.token(batch["tok"], batch["attn"], batch[span_key], claim_type)

    def state(self, batch, span_key: str = "concl_mask"):
        if self.token is None:
            return self.layer_state(batch)
        tok_h = self.token_state(batch, span_key)
        if self.gate is None:
            return tok_h
        lay_h = self.layer_state(batch)
        g = torch.sigmoid(self.gate(torch.cat([tok_h, lay_h], -1)))
        if self.training and self.branch_dropout > 0:
            r = torch.rand(g.shape[0], 1, device=g.device)
            g = torch.where(r < self.branch_dropout / 2, torch.ones_like(g),
                            torch.where(r < self.branch_dropout, torch.zeros_like(g), g))
        return g * tok_h + (1 - g) * lay_h

    def forward(self, batch, span_key: str = "concl_mask"):
        """Logit of P(correct) for the response (span ``batch[span_key]`` is the conclusion)."""
        return self.out(self.state(batch, span_key)).reshape(-1)

    def forward_all(self, batch):
        """(answer logit, verdict logit); PCS heads only."""
        h = self.state(batch)
        return self.out(h).reshape(-1), self.verdict(h).reshape(-1)

    def step_logits(self, batch):
        """Score of the span in ``batch["concl_mask"]`` as a reasoning step.

        PCS heads read the token-branch state of the span marked as a reasoning-type claim; heads without PCS
        score the step span exactly like a conclusion.
        """
        if self.pcs:
            return self.out(self.token_state(batch, claim_type=REASONING)).reshape(-1)
        return self.forward(batch)
