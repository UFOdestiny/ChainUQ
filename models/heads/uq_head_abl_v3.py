"""UQ ablation v3: remove the prototype-to-context gate from ChainUQ."""
from __future__ import annotations

import torch

from models.heads.uq_head_v1 import UQHeadV1


class UQAblationHeadV3(UQHeadV1):
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
        return self.proto_residual(self.refine(mixed))
