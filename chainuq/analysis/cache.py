"""Cache the head-independent evidence of every evaluation cell for the Stage-2 analyses.

For each (backbone, dataset, split) this stores the evaluation-set UIDs, correctness labels, the static part of
u(r) (u_token and u_gap, from backbone features), span token features, the judge's chain verdict, P(true) and step
labels. Head-dependent evidence (u_base, u_reason, predicted verdict) is rebuilt from prediction files in
milliseconds, so Stage-2 variants are scored without reloading the multi-GB token features.

    python -m chainuq.analysis.cache --backbones llama,qwen8b,ministral --workers 12
"""

from __future__ import annotations

import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from chainuq import config

ANALYSIS = config.ANALYSIS_ROOT
SPLITS = ("validation", "test", "test2")
STATIC_NAMES = ["neg_mean_ll", "neg_top1_logp", "neg_margin_logp"] + [
    "n_hops",
    "reasoning_claim_count",
    "reasoning_tokens_per_claim",
    "reasoning_active_tokens_per_claim",
    "reasoning_feature_gap_l2",
    "reasoning_feature_gap_mean_abs",
    "reasoning_feature_gap_max_abs",
    "reasoning_feature_gap_cosine",
    "sidecar_reasoning_tokens",
    "sidecar_reasoning_active_tokens",
    "sidecar_reasoning_mean_l2",
    "sidecar_reasoning_std_l2",
    "sidecar_reasoning_mean_abs",
    "sidecar_reasoning_std_abs",
]


def cache_path(bb: str, ds: str, split: str) -> Path:
    return ANALYSIS / "cache" / bb / f"{ds}_{split}.pkl"


def build(job):
    bb, ds, split = job
    out = cache_path(bb, ds, split)
    if out.exists():
        return f"skip {out}"
    from chainuq import evidence as EV
    from chainuq import registry
    from chainuq.artifacts import feat_dir, feature_info
    from chainuq.evaluation import FEATURE_KEYS, verdict_score
    from chainuq.samples import load_judge_task, load_samples

    if not (feat_dir(bb, ds, split) / "DONE.json").exists():
        return f"missing {bb}/{ds}/{split}"
    hidden = feature_info(bb)["hidden_size"]
    samples = load_samples(bb, ds, split, feature_keys=FEATURE_KEYS)
    rec = load_judge_task(registry.MAIN_JUDGE, "final_blind", bb, ds, split)
    static = [EV.static_features(steps=s.steps, conclusion=s.conclusion, n_hops=s.gen["n_hops"], feat=s.feat, hidden_dim=hidden) for s in samples]
    data = {
        "uids": [s.uid for s in samples],
        "y": np.asarray([s.y for s in samples]),
        "static": np.asarray([[d[n] for n in STATIC_NAMES] for d in static], dtype=np.float64),
        "span": EV.matrix([EV.span_features(steps=s.steps, conclusion=s.conclusion, feat=s.feat) for s in samples], EV.SPAN_FEATURE_NAMES),
        "verdict": np.asarray([verdict_score(rec.get(s.uid, {})) for s in samples], dtype=np.float64),
        "p_true": np.asarray([(rec.get(s.uid, {}).get("p_true") or np.nan) for s in samples], dtype=np.float64),
        "step_labels": [s.step_labels for s in samples],
        "n_steps": np.asarray([len(s.steps) for s in samples]),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(data, f)
    tmp.rename(out)
    return f"wrote {out} n={len(samples)}"


def main():
    from chainuq.registry import DATASETS

    p = argparse.ArgumentParser()
    p.add_argument("--backbones", default="llama,qwen8b,ministral")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--splits", default=",".join(SPLITS))
    p.add_argument("--workers", type=int, default=8)
    a = p.parse_args()
    jobs = [(b, d, s) for b in a.backbones.split(",") for d in a.datasets.split(",") for s in a.splits.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        for msg in ex.map(build, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
