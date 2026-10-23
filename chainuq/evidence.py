"""Reasoning-chain evidence for the Stage-2 calibrator (paper appendix, Eq. for u(r)).

u(r) = [u_base; u_reason; u_token; u_gap; u_judge]. It reads only (i) the head's
predictions, (ii) backbone features of the generated text, (iii) question metadata
(hop count), and (iv) the answer-blind step judgments. It never receives the
correctness label; ``tests/test_evidence.py`` checks that no feature can.

``span_features`` adds judge-free token uncertainty of the conclusion span and of
the weakest reasoning step.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

GROUPS = {
    "u_base": ["base_logit", "conclusion_logit", "conclusion_prob"],
    "u_reason": ["reasoning_mean_logit", "reasoning_min_logit", "reasoning_max_logit", "reasoning_std_logit",
                 "reasoning_mean_prob", "reasoning_min_prob", "reasoning_max_prob", "reasoning_std_prob",
                 "reasoning_var_prob", "reasoning_gap_prob", "base_minus_conclusion_logit", "reasoning_count"],
    "u_token": ["neg_mean_ll", "neg_top1_logp", "neg_margin_logp"],
    "u_gap": ["n_hops", "reasoning_claim_count", "reasoning_tokens_per_claim", "reasoning_active_tokens_per_claim",
              "reasoning_feature_gap_l2", "reasoning_feature_gap_mean_abs", "reasoning_feature_gap_max_abs",
              "reasoning_feature_gap_cosine", "sidecar_reasoning_tokens", "sidecar_reasoning_active_tokens",
              "sidecar_reasoning_mean_l2", "sidecar_reasoning_std_l2", "sidecar_reasoning_mean_abs",
              "sidecar_reasoning_std_abs"],
    "u_judge": ["reasoning_label_mean", "reasoning_label_std", "reasoning_label_majority_agreement",
                "reasoning_label_majority_margin", "reasoning_label_valid_count"],
}
FEATURE_NAMES = [n for g in GROUPS.values() for n in g]
JUDGE_COLUMNS = [FEATURE_NAMES.index(n) for n in GROUPS["u_judge"]]
JUDGE_FREE_COLUMNS = [i for i in range(len(FEATURE_NAMES)) if i not in JUDGE_COLUMNS]
SPAN_FEATURE_NAMES = ["concl_logprob_sum", "concl_logprob_mean", "concl_logprob_min", "concl_entropy_mean",
                      "step_min_mean_ll", "step_max_mean_entropy", "step_ll_spread"]
FORBIDDEN_SUBSTRINGS = ("match_rate", "label_gap", "majority_gap", "correct", "gold", "ground_truth", "sample_label")


def columns(names: List[str]) -> List[int]:
    return [FEATURE_NAMES.index(n) for n in names]


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def static_features(*, steps: list, conclusion: dict, n_hops: int, feat: dict, hidden_dim: int) -> Dict[str, float]:
    """u_token and u_gap: depend only on the generated text and backbone features (cacheable per response)."""
    ll, top1, margin = (feat[k].float().numpy() for k in ("ll", "top1", "margin"))
    f = {"neg_mean_ll": float(-ll.mean()), "neg_top1_logp": float(-top1.mean()), "neg_margin_logp": float(-margin.mean())}

    tok = feat["tok"][:, :hidden_dim].float()
    reasoning = tok[steps[0]["tok"][0]: steps[-1]["tok"][1]]
    c0, c1 = conclusion["tok"]
    concl_mean, reas_mean = tok[c0:c1].mean(0).numpy(), reasoning.mean(0).numpy()
    diff = reas_mean - concl_mean
    cos = float(np.dot(reas_mean, concl_mean) / (np.linalg.norm(reas_mean) * np.linalg.norm(concl_mean) + 1e-12))
    l2 = reasoning.norm(dim=1).numpy()
    absv = reasoning.abs().reshape(-1).numpy()
    K, T_r = float(len(steps)), float(reasoning.shape[0])
    f.update({
        "n_hops": float(n_hops), "reasoning_claim_count": K,
        "reasoning_tokens_per_claim": T_r / K, "reasoning_active_tokens_per_claim": T_r / K,
        "reasoning_feature_gap_l2": float(np.linalg.norm(diff)), "reasoning_feature_gap_mean_abs": float(np.abs(diff).mean()),
        "reasoning_feature_gap_max_abs": float(np.abs(diff).max()), "reasoning_feature_gap_cosine": cos,
        "sidecar_reasoning_tokens": T_r, "sidecar_reasoning_active_tokens": T_r,
        "sidecar_reasoning_mean_l2": float(l2.mean()), "sidecar_reasoning_std_l2": float(l2.std()),
        "sidecar_reasoning_mean_abs": float(absv.mean()), "sidecar_reasoning_std_abs": float(absv.std()),
    })
    return f


def sample_features(*, logit: float, step_logits: Optional[List[float]], steps: list, conclusion: dict,
                    n_hops: int, feat: dict, hidden_dim: int, step_labels: Optional[list],
                    static: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """u(r) of one response. Heads without step scores fall back to the conclusion score for u_reason."""
    z = float(logit)
    pc = float(_sigmoid(z))
    rl = np.asarray(step_logits if step_logits else [z], dtype=np.float64)
    rp = _sigmoid(rl)
    f = {
        "base_logit": z, "conclusion_logit": z, "conclusion_prob": pc,
        "reasoning_mean_logit": rl.mean(), "reasoning_min_logit": rl.min(), "reasoning_max_logit": rl.max(),
        "reasoning_std_logit": rl.std(), "reasoning_mean_prob": rp.mean(), "reasoning_min_prob": rp.min(),
        "reasoning_max_prob": rp.max(), "reasoning_std_prob": rp.std(), "reasoning_var_prob": rp.var(),
        "reasoning_gap_prob": abs(pc - rp.mean()), "base_minus_conclusion_logit": 0.0,
        "reasoning_count": float(len(rl)),
    }
    f.update(static if static is not None else
             static_features(steps=steps, conclusion=conclusion, n_hops=n_hops, feat=feat, hidden_dim=hidden_dim))

    labels = [int(v) for v in (step_labels or []) if v in (0, 1)]
    if labels:
        arr = np.asarray(labels, dtype=np.float64)
        mean = arr.mean()
        f.update({"reasoning_label_mean": mean, "reasoning_label_std": arr.std(),
                  "reasoning_label_majority_agreement": float(np.mean(arr == float(mean >= 0.5))),
                  "reasoning_label_majority_margin": abs(mean - 0.5), "reasoning_label_valid_count": float(len(arr))})
    else:
        f.update({k: np.nan for k in GROUPS["u_judge"]})
        f["reasoning_label_valid_count"] = 0.0
    return f


def span_features(*, steps: list, conclusion: dict, feat: dict) -> Dict[str, float]:
    """Token log-likelihood and entropy of the conclusion span and of the weakest reasoning step."""
    ll, ent = feat["ll"].float().numpy(), feat["ent"].float().numpy()
    c0, c1 = conclusion["tok"]
    cl, ce = ll[c0:c1], ent[c0:c1]
    step_ll = np.asarray([ll[a:b].mean() for a, b in (s["tok"] for s in steps)])
    step_ent = np.asarray([ent[a:b].mean() for a, b in (s["tok"] for s in steps)])
    return {"concl_logprob_sum": float(cl.sum()), "concl_logprob_mean": float(cl.mean()),
            "concl_logprob_min": float(cl.min()), "concl_entropy_mean": float(ce.mean()),
            "step_min_mean_ll": float(step_ll.min()), "step_max_mean_entropy": float(step_ent.max()),
            "step_ll_spread": float(step_ll.max() - step_ll.min())}


def matrix(rows: List[Dict[str, float]], names: Optional[List[str]] = None) -> np.ndarray:
    names = names or FEATURE_NAMES
    return np.asarray([[r[n] for n in names] for r in rows], dtype=np.float64)


def assert_no_label_features(names: List[str]) -> None:
    bad = [n for n in names if any(s in n for s in FORBIDDEN_SUBSTRINGS)]
    if bad:
        raise AssertionError(f"label-derived calibration features: {bad}")
