"""Label-free target adaptation of the trained ChainUQ head (see ``chainuq.adaptation``).

For each OOD target dataset, the head of every seed is fine-tuned with the unlabeled pool of target responses
(the ``train`` split of that dataset: questions outside every evaluation split), then predicts the validation,
test, and test2 splits. Predictions are written under ``--run`` so that ``experiments.adapt_eval`` scores them
with the same R-Log as the source head.

Runs of the paper (predictions key: options):
  chainuq-ta                 default (consistency targets of the main judge, 2,000-question pools)
  chainuq-ta-src             --lam 0 (source-only control)
  chainuq-ta-pseudo          --target_mode pseudo (verdict as pseudo-label)
  chainuq-ta-steps           --target_mode steps (step labels only)
  chainuq-ta-both            --target_mode both
  chainuq-ta-mid[-pseudo]    --pool_judge qwen4b [--target_mode pseudo] (mid-strength pool verifier)
  chainuq-ta-weak[-pseudo]   --pool_judge qwen1b [--target_mode pseudo] (weak pool verifier)
  chainuq-ta-n<N>            --pool_n N (pool-size curve) or --pool_root <6,000-question pool artifacts>

Example:
    CUDA_VISIBLE_DEVICES=0 python -m chainuq.experiments.adapt --backbone llama
"""

from __future__ import annotations

import argparse
import hashlib
import json
from contextlib import nullcontext

import torch

from chainuq import heads as H
from chainuq.adaptation import MODES, adapt
from chainuq.artifacts import feat_dir, pred_path, use_root, write_jsonl
from chainuq.registry import FINAL_HEAD, OOD_DATASETS, SOURCE_DATASET
from chainuq.samples import load_samples, load_verdicts
from chainuq.training import load_trained, prediction_records, to_items


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--datasets", default=",".join(OOD_DATASETS))
    p.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--lam", type=float, default=1.0, help="weight of the target loss (0 = source-only control)")
    p.add_argument("--max_src", type=int, default=8000, help="source responses kept on the GPU for replay batches")
    p.add_argument("--target_mode", default="pcs", choices=MODES)
    p.add_argument("--pool_n", type=int, default=0, help="use a fixed random subset of this many pool responses")
    p.add_argument("--pool_judge", default="", help="judge whose labels supervise the pool (default: main judge)")
    p.add_argument("--pool_root", default="", help="artifact tree holding the pool (default: CHAINUQ_ARTIFACTS)")
    p.add_argument("--run", default="chainuq-ta", help="predictions key of the adapted heads")
    args = p.parse_args()
    bb, device = args.backbone, torch.device("cuda")
    keys = list(H.HEADS[FINAL_HEAD].features)
    models = load_trained(FINAL_HEAD, bb, [int(s) for s in args.seeds.split(",")], device)
    tr = load_samples(bb, SOURCE_DATASET, "train", feature_keys=keys, require_steps=False, device=device, max_n=args.max_src)
    src = to_items(tr, device=device, verdicts=load_verdicts(bb, SOURCE_DATASET, "train"))
    del tr
    judge = args.pool_judge or None
    for ds in args.datasets.split(","):
        with use_root(args.pool_root) if args.pool_root else nullcontext():
            pool = load_samples(bb, ds, "train", feature_keys=keys, device=device, unlabeled=True, judge=judge)
            pool = to_items(pool, device=device, verdicts=load_verdicts(bb, ds, "train", judge=judge))
        if args.pool_n:
            pool = sorted(pool, key=lambda it: hashlib.md5(it["uid"].encode()).hexdigest())[: args.pool_n]
        evals = {sp: to_items(load_samples(bb, ds, sp, feature_keys=keys, device=device), device=device)
                 for sp in ("validation", "test", "test2") if (feat_dir(bb, ds, sp) / "DONE.json").exists()}
        for seed, base in models.items():
            m = adapt(base, src, pool, device, args.steps, args.lr, args.lam, seed, mode=args.target_mode)
            for sp, items in evals.items():
                write_jsonl(pred_path(args.run, bb, seed, ds, sp), prediction_records(m, items, device))
            print(json.dumps({"run": args.run, "backbone": bb, "dataset": ds, "seed": seed, "n_pool": len(pool)}), flush=True)
        del pool, evals
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
