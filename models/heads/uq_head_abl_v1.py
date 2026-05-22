"""UQ ablation v1: remove learned reliability prototypes from ChainUQ."""
from __future__ import annotations

import torch

from models.heads.uq_head_v1 import UQHeadV1


class UQAblationHeadV1(UQHeadV1):
    def _refine_conclusion_state(self, conclusion_h, sample_h, token_h, attention_mask_i, claim_type_id):
        _ = token_h
        _ = attention_mask_i
        _ = claim_type_id
        prototype = sample_h
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
        mixed = self.context_gate(mixed.unsqueeze(0), sample_h).squeeze(0)
        return self.proto_residual(self.refine(mixed))
