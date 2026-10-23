"""Trainable Stage-1 heads: the ChainUQ head, its ablations, and the supervised probe baselines.

Every head except Lookback Lens is sized to the largest width (a multiple of 8) whose
trainable parameters stay within ``PARAM_BUDGET``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple

import torch.nn as nn

from chainuq.heads.baselines import SAPLMA, Factoscope, SpanPoolMLP, UHead, UHeadPCS
from chainuq.heads.chainuq import ChainUQHead

PARAM_BUDGET = 2_000_000
CHAINUQ_FEATURES = ("tok", "layers_anchor", "layers_concl_last")


@dataclass(frozen=True)
class HeadSpec:
    arch: str
    kwargs: dict = field(default_factory=dict)
    features: Tuple[str, ...] = ("tok",)
    pcs: Optional[Tuple[float, float]] = None  # (step-label weight, verdict weight) of the PCS loss
    step_scores: bool = True  # whether the head scores each reasoning step (for the evidence vector u(r))


HEADS = {
    # ChainUQ head with privileged consistency supervision, and its ablations.
    "chainuq": HeadSpec("chainuq", features=CHAINUQ_FEATURES, pcs=(0.5, 0.5)),
    "chainuq-wo-step": HeadSpec("chainuq", features=CHAINUQ_FEATURES, pcs=(0.0, 0.5)),
    "chainuq-wo-verdict": HeadSpec("chainuq", features=CHAINUQ_FEATURES, pcs=(0.5, 0.0)),
    "chainuq-wo-pcs": HeadSpec("chainuq", dict(pcs=False), CHAINUQ_FEATURES),
    "chainuq-wo-pcs-layer": HeadSpec("chainuq", dict(pcs=False, layer_branch=False), CHAINUQ_FEATURES),
    "chainuq-wo-pcs-token": HeadSpec("chainuq", dict(pcs=False, token_branch=False), CHAINUQ_FEATURES),
    "chainuq-wo-pcs-bdrop": HeadSpec("chainuq", dict(pcs=False, branch_dropout=0.0), CHAINUQ_FEATURES),
    # Supervised probes.
    "uhead": HeadSpec("uhead"),
    "uhead-pcs": HeadSpec("uhead-pcs", pcs=(0.5, 0.5)),
    "meanpool-mlp": HeadSpec("spanpool"),
    "attnpool-mlp": HeadSpec("spanpool", dict(attention_pool=True)),
    "saplma": HeadSpec("saplma", features=("layers_concl_last",), step_scores=False),
    "factoscope": HeadSpec("factoscope", features=("layers_anchor", "lens_rank", "lens_topk_lp", "lens_topk_ids"),
                           step_scores=False),
}
LOOKBACK_FEATURES = ("lookback",)


def predicts_verdict(method: str) -> bool:
    """False for heads whose verdict output exists but is not trained (its prediction is then not used)."""
    spec = HEADS.get(method)
    return not (spec is not None and spec.pcs is not None and spec.pcs[1] == 0)


def build(method: str, dims: dict, head_dim: int) -> nn.Module:
    spec = HEADS[method]
    F, d, L = dims["feature_dim"], dims["hidden_size"], dims["n_layers"]
    if spec.arch == "chainuq":
        return ChainUQHead(F, d, L, head_dim, **spec.kwargs)
    if spec.arch == "uhead":
        return UHead(F, head_dim)
    if spec.arch == "uhead-pcs":
        return UHeadPCS(F, head_dim)
    if spec.arch == "spanpool":
        return SpanPoolMLP(F, head_dim, **spec.kwargs)
    if spec.arch == "saplma":
        return SAPLMA(d, dims["saplma_layer"], head_dim)
    if spec.arch == "factoscope":
        return Factoscope(d, L, head_dim)
    raise ValueError(method)


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def size_to_budget(method: str, dims: dict, budget: int = PARAM_BUDGET, lo: int = 32, hi: int = 1024, step: int = 8) -> int:
    """Largest width (a multiple of 8, so 8 attention heads divide it) whose head stays within ``budget``."""
    best = lo
    for width in range(lo, hi + 1, step):
        if width % 8:
            continue
        if count_params(build(method, dims, width)) > budget:
            break
        best = width
    return best
