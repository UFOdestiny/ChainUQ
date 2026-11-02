"""Trainable Stage-1 heads: the ChainUQ head, its ablations, and the supervised probe baselines.

Every head except Lookback Lens is sized to the largest width (a multiple of 8) whose
trainable parameters stay within ``PARAM_BUDGET``. A head key is also the name of its
training run (``ckpt/<key>``, ``preds/<key>``); ``pipeline.train --tag`` appends a suffix.
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
    pool_size: int = 0  # unlabeled questions per OOD domain that join training (0: labeled source only)
    pool_mode: str = "pcs"  # pool targets: "pcs" (step labels and verdict), "steps", "verdict", or "pseudo"


PCS = (0.5, 0.5)
HEADS = {
    # ChainUQ: PCS on the labeled source and on unlabeled 6,000-question pools of the five OOD domains.
    "chainuq": HeadSpec("chainuq", features=CHAINUQ_FEATURES, pcs=PCS, pool_size=6000),
    # Ablations: PCS on the labeled source only, smaller pools, one PCS target on the pools, no PCS.
    "chainuq-source": HeadSpec("chainuq", features=CHAINUQ_FEATURES, pcs=PCS),
    "chainuq-pool2k": HeadSpec("chainuq", features=CHAINUQ_FEATURES, pcs=PCS, pool_size=2000),
    "chainuq-pool-steps": HeadSpec("chainuq", features=CHAINUQ_FEATURES, pcs=PCS, pool_size=6000, pool_mode="steps"),
    "chainuq-pool-verdict": HeadSpec("chainuq", features=CHAINUQ_FEATURES, pcs=PCS, pool_size=6000, pool_mode="verdict"),
    "chainuq-wo-pcs": HeadSpec("chainuq", dict(pcs=False), CHAINUQ_FEATURES),
    # Supervised probes, and UHead with the same PCS targets and pools.
    "uhead": HeadSpec("uhead"),
    "uhead-pcs": HeadSpec("uhead-pcs", pcs=PCS),
    "uhead-pcs-pool6k": HeadSpec("uhead-pcs", pcs=PCS, pool_size=6000),
    "meanpool-mlp": HeadSpec("spanpool"),
    "attnpool-mlp": HeadSpec("spanpool", dict(attention_pool=True)),
    "saplma": HeadSpec("saplma", features=("layers_concl_last",), step_scores=False),
    # SAPLMA distilled from the judge: the chain verdict of the pools as a soft correctness label.
    "saplma-pool6k-pseudo": HeadSpec("saplma", features=("layers_concl_last",), step_scores=False, pool_size=6000, pool_mode="pseudo"),
    "factoscope": HeadSpec("factoscope", features=("layers_anchor", "lens_rank", "lens_topk_lp", "lens_topk_ids"), step_scores=False),
}
LOOKBACK_FEATURES = ("lookback",)


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
