"""Post-hoc calibrators on the same final head score, per (backbone, dataset), mean/std over the head seeds.

PS / TS / ISO see the head score only; BWH and R-Log (with evidence-set selection, i.e. ChainUQ) also see the
evidence and the judge outputs. Results merge into ``results/paper/<backbone>/<dataset>_<split>.json``.

Example:
    python -m chainuq.analysis.calibrators --backbone llama --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from chainuq import calibration as C
from chainuq.artifacts import feature_info, results_dir
from chainuq.evaluation import Cell, agg, available_methods, eval_scores, final_calibrator, final_evidence, fit_eval, scalar_calibrated
from chainuq.registry import BACKBONES, DATASETS, FINAL_HEAD


def run(backbone, ds, split):
    methods = available_methods(backbone)
    cell = Cell(backbone, ds, split, feature_info(backbone)["hidden_size"])
    yv, yt = cell.labels("validation"), cell.labels(split)
    rows = defaultdict(list)
    for seed in methods[FINAL_HEAD]:
        ev = final_evidence(cell, backbone, seed)
        if ev is None:
            continue
        Xv, Xt = ev.matrices("B")
        for name in ("ps", "ts", "iso"):
            rows[name].append(eval_scores(yv, ev.sv, yt, ev.st, scalar_calibrated(name, yv, ev.sv, ev.st)))
        rows["bwh"].append(fit_eval(C.BWH, yv, ev.sv, Xv, yt, ev.st, Xt)[0])
        rows["rlog"].append(fit_eval(final_calibrator(ev, "B"), yv, ev.sv, Xv, yt, ev.st, Xt)[0])
    return {"calibrators": {k: agg(v) for k, v in rows.items()}, "n_val": len(cell.val), "n_test": len(cell.test)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--split", default="test")
    args = p.parse_args()
    out_dir = results_dir() / "paper" / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.dataset}_{args.split}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    old.update(run(args.backbone, args.dataset, args.split))
    path.write_text(json.dumps(old, indent=1, default=float))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
