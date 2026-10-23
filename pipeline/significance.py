#!/usr/bin/env python3
"""Paired bootstrap of the AUROC difference between ChainUQ (and ChainUQ-A) and the strongest baseline.

For one (backbone, dataset): refit the final R-Log on validation for the first ``--seeds`` head seeds,
score the test split, and compare with the strongest baseline of the same setting (by mean test AUROC in
eval.json; for multi-seed baselines the same seeds are used). Each bootstrap resample averages AUROC
over seeds. Writes ``results/paper/<backbone>/<dataset>_signif.json``.

Example:
    python pipeline/significance.py --backbone llama --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq import calibration as C
from cuq import ufeatures as UF
from cuq.registry import FINAL_HEAD, SOURCE_DATASET, feat_dir, results_dir

_here = Path(__file__).resolve().parent


def _module(name):
    spec = importlib.util.spec_from_file_location(name, _here / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


E = _module("evaluate")
PL = _module("paper_latex")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--n_boot", type=int, default=1000)
    args = p.parse_args()
    bb, ds = args.backbone, args.dataset
    hidden = json.loads((feat_dir(bb, SOURCE_DATASET, "train") / "DONE.json").read_text())["hidden_size"]
    methods = E.available_methods(bb)
    cell = E.Cell(bb, ds, "step", hidden)
    yv, yt = cell.labels("validation"), cell.labels("test")
    Ev0, Et0 = cell.extra("validation"), cell.extra("test")
    jv, jt = cell.verdict("validation"), cell.verdict("test")
    J = E.cols(UF.GROUPS["u_judge"])
    notJ = [i for i in range(len(UF.FEATURE_NAMES)) if i not in J]
    seeds = methods[FINAL_HEAD][: args.seeds]
    ours = {"B": [], "A": []}
    for seed in seeds:
        pv, pt = E.load_preds(FINAL_HEAD, bb, seed, ds, "validation"), E.load_preds(FINAL_HEAD, bb, seed, ds, "test")
        sv, st = cell.scores("validation", pv), cell.scores("test", pt)
        Uv, Ut = cell.u_matrix("validation", pv), cell.u_matrix("test", pt)
        Ev, Et = cell.with_vhat(Ev0, "validation", pv), cell.with_vhat(Et0, "test", pt)
        for tier, Xv, Xt in (("B", np.column_stack([Uv, jv, Ev]), np.column_stack([Ut, jt, Et])),
                             ("A", np.column_stack([Uv[:, notJ], Ev]), np.column_stack([Ut[:, notJ], Et]))):
            ours[tier].append(C.RLog2c().fit(sv, Xv, yv).predict(st, Xt))
    R = json.loads((results_dir() / bb / "eval.json").read_text())["datasets"][ds]
    tier_of = {k: t for _, ms in PL.GROUPS for k, t, _ in ms}
    out = {}
    for tier, pool in (("B", PL.BASELINES), ("A", PL.TIER_A_BASELINES)):
        cand = {k: PL.value(R, k, tier_of[k], "auroc") for k in pool}
        best = max({k: v for k, v in cand.items() if v is not None}.items(), key=lambda kv: kv[1])[0]
        bseeds = methods[best][: args.seeds]
        base = [cell.scores("test", E.load_preds(best, bb, sd, ds, "test")) for sd in bseeds]
        ok = np.all([np.isfinite(b) for b in base], axis=0) & np.all([np.isfinite(o) for o in ours[tier]], axis=0)
        r = E.paired_bootstrap_auroc(yt[ok], np.stack(ours[tier])[:, ok], np.stack(base)[:, ok], n_boot=args.n_boot)
        out[tier] = {"baseline": best, "n": int(ok.sum()), **r}
    path = results_dir() / "paper" / bb / f"{ds}_signif.json"
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
