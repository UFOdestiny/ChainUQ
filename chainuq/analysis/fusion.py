"""Probe + judge fusion baselines of the main comparison: a logistic regression (C chosen by inner cross-validation)
on a supervised probe's score and the judge's two outputs (chain verdict, P(true) log-odds), fit on the validation
split of each dataset. Reads the per-response predictions of ``analysis.predictions``.

Writes ``results/<backbone>/eval_extra.json`` (``eval_extra_test2.json``) in the layout of ``eval.json``.

    python -m chainuq.analysis.fusion
"""

from __future__ import annotations

import json

import numpy as np

from chainuq.analysis import variants as D
from chainuq.analysis.predictions import out_path
from chainuq.artifacts import results_dir
from chainuq.evaluation import agg, eval_scores

FUSION = {"fusion-saplma": "S+verdict+ptrue@lrsel#saplma", "fusion-uhead": "S+verdict+ptrue@lrsel#uhead"}


def main():
    for bb in D.BACKBONES:
        for split in ("test", "test2"):
            out = {"datasets": {}}
            for ds in D.DATASETS:
                if not out_path(next(iter(FUSION.values())), bb, ds, split).exists():
                    continue
                cell = out["datasets"].setdefault(ds, {"tier_b": {}})
                for key, variant in FUSION.items():
                    zv = np.load(out_path(variant, bb, ds, "validation"))
                    zt = np.load(out_path(variant, bb, ds, split))
                    rows = [eval_scores(zv["y"], pv, zt["y"], pt, pt) for pv, pt in zip(zv["p"], zt["p"])]
                    cell["tier_b"][key] = agg(rows)
            path = results_dir() / bb / ("eval_extra.json" if split == "test" else f"eval_extra_{split}.json")
            path.write_text(json.dumps(out, indent=1))
            print(path)


if __name__ == "__main__":
    main()
