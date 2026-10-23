"""Paired bootstrap of the test AUROC difference between ChainUQ (and ChainUQ-A) and the strongest baseline.

For one (backbone, dataset): refit R-Log on validation for the first ``--seeds`` head seeds, score the test
split, and compare with the strongest baseline of the same setting (highest mean test AUROC in ``eval.json``;
multi-seed baselines use the same seeds). Each bootstrap resample averages AUROC over seeds.
Writes ``results/paper/<backbone>/<dataset>_signif.json``.

Example:
    python -m chainuq.experiments.significance --backbone llama --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from chainuq import calibration as C
from chainuq.artifacts import feature_info, results_dir
from chainuq.evaluation import Cell, HeadEvidence, available_methods, load_preds, paired_bootstrap_auroc
from chainuq.registry import BASELINE_GROUPS, FINAL_HEAD, TIER_A_BASELINES

TIER_OF = {k: t for _, ms in BASELINE_GROUPS for k, t, _ in ms}
POOLS = {"B": list(TIER_OF), "A": TIER_A_BASELINES}  # ChainUQ competes with every baseline, ChainUQ-A with tier A


def strongest_baseline(R, pool):
    cand = {k: R[TIER_OF[k]][k]["auroc"]["mean"] for k in pool if k in R[TIER_OF[k]]}
    return max(cand, key=cand.get)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()
    bb, ds = args.backbone, args.dataset
    methods = available_methods(bb)
    cell = Cell(bb, ds, "test", feature_info(bb)["hidden_size"])
    yv, yt = cell.labels("validation"), cell.labels("test")
    ours = {"B": [], "A": []}
    for seed in methods[FINAL_HEAD][: args.seeds]:
        ev = HeadEvidence(cell, load_preds(FINAL_HEAD, bb, seed, ds, "validation"), load_preds(FINAL_HEAD, bb, seed, ds, "test"))
        for tier in ours:
            Xv, Xt = ev.matrices(tier)
            ours[tier].append(C.RLog().fit(ev.sv, Xv, yv).predict(ev.st, Xt))
    R = json.loads((results_dir() / bb / "eval.json").read_text())["datasets"][ds]
    out = {}
    for tier, pool in POOLS.items():
        best = strongest_baseline(R, pool)
        base = [cell.scores("test", load_preds(best, bb, sd, ds, "test")) for sd in methods[best][: args.seeds]]
        ok = np.all([np.isfinite(b) for b in base], axis=0) & np.all([np.isfinite(o) for o in ours[tier]], axis=0)
        r = paired_bootstrap_auroc(yt[ok], np.stack(ours[tier])[:, ok], np.stack(base)[:, ok], n_boot=args.n_boot)
        out[tier] = {"baseline": best, "n": int(ok.sum()), **r}
    path = results_dir() / "paper" / bb / f"{ds}_signif.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
