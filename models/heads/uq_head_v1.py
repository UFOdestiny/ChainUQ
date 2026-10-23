"""Core ChainUQ head: conclusion evidence lens with learned reliability prototypes."""
from __future__ import annotations

import torch
import torch.nn as nn

from models.heads.uq_head_base import UQHeadBase
from models.heads.uq_modules import ResidualMLP


class UQHeadV1(UQHeadBase):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int = 1,
        head_dim: int = 512,
        n_layers: int = 2,
        n_heads: int = 8,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            feature_dim=feature_dim,
            num_classes=num_classes,
            head_dim=head_dim,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            **kwargs,
        )
        self.prototype_bank = nn.Parameter(torch.randn(4, head_dim))
        self.prototype_query = nn.Linear(head_dim * 3, 4)
        self.prototype_gate = nn.Linear(head_dim * 4, head_dim)
        self.refine = ResidualMLP(head_dim=head_dim, dropout=dropout, expansion=3)
        self.proto_residual = ResidualMLP(head_dim=head_dim, dropout=dropout, expansion=2)
        self.proto_temperature = nn.Parameter(torch.tensor(1.0))
        nn.init.normal_(self.prototype_bank, std=0.02)
        nn.init.xavier_uniform_(self.prototype_query.weight)
        nn.init.zeros_(self.prototype_query.bias)
        nn.init.xavier_uniform_(self.prototype_gate.weight)
        nn.init.zeros_(self.prototype_gate.bias)

    def _refine_conclusion_state(self, conclusion_h, sample_h, token_h, attention_mask_i, claim_type_id):
        _ = token_h
        _ = attention_mask_i
        _ = claim_type_id
        proto_weights = torch.softmax(
            self.prototype_query(torch.cat([conclusion_h, sample_h, torch.abs(conclusion_h - sample_h)], dim=-1)),
            dim=-1,
        )
        prototype = proto_weights @ self.prototype_bank
        temp = torch.clamp(self.proto_temperature, min=0.35, max=3.0)
        proto_gate = torch.sigmoid(
            self.prototype_gate(
                torch.cat(
                    [
                        conclusion_h,
                        prototype,
                        sample_h,
                        torch.abs(prototype - conclusion_h),
                    ],
                    dim=-1,
                )
            )
            / temp
        )
        mixed = proto_gate * conclusion_h + (1.0 - proto_gate) * prototype
        mixed = self.context_gate(mixed.unsqueeze(0), 0.5 * (prototype + sample_h)).squeeze(0)
        return self.proto_residual(self.refine(mixed))
