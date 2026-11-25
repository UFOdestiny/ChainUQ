"""ChainUQ token head without the reliability-refinement block (Step 4).

Removes the learned reliability bank, the conditioned prototype selection, the context gate, and the
final residual refiner; the conclusion seed only passes through the first residual MLP.
"""
from __future__ import annotations

from models.heads.uq_head_v1 import UQHeadV1


class UQCoreHead(UQHeadV1):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for name in ("prototype_bank", "prototype_query", "prototype_gate", "proto_residual", "proto_temperature",
                     "context_gate"):
            delattr(self, name)

    def _refine_conclusion_state(self, conclusion_h, sample_h, token_h, attention_mask_i, claim_type_id):
        _ = sample_h, token_h, attention_mask_i, claim_type_id
        return self.refine(conclusion_h)
