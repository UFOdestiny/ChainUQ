"""Per-response probabilities of Stage-2 variants (fit on validation, applied once to test and test2).

Writes one npz per (variant, backbone, dataset, split) under ``predictions/`` with the UIDs, labels, judge verdict,
raw head score and the calibrated probability of every seed, for the fusion baselines (``analysis.fusion``) and the
response-level analyses (``analysis.responses``, ``analysis.subtasks``).

    python -m chainuq.analysis.predictions --workers 30 \\
        --variants 'B-u_reason-u_token-u_judge+ptrue+head:saplma+vq@rlogs#chainuq,S+verdict+ptrue@lrsel#saplma'
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chainuq.analysis import variants as D
from chainuq.analysis.cache import ANALYSIS, cache_path

OUT = ANALYSIS / "predictions"
SEEDS = list(range(2026, 2036))


def out_path(variant, bb, ds, split):
    safe = variant.replace("#", "__").replace("@", "_at_").replace("+", "p_").replace("-", "m_")
    return OUT / safe / bb / f"{ds}_{split}.npz"


def run(job):
    variant, bb, ds = job
    names, cal, head, extra = D.parse_variant(variant)
    res = {}
    for seed in SEEDS:
        bv = D.blocks(bb, ds, "validation", head, seed, tuple(extra))
        if bv is None:
            continue
        m = D.make_cal(cal, names, head, extra, bb, ds, seed)().fit(bv["s"], D.design(bv, names), bv["y"])
        r = res.setdefault("validation", {"p": [], "s": [], "seeds": []})
        r["p"].append(m.predict(bv["s"], D.design(bv, names)))
        r["s"].append(bv["s"])
        r["seeds"].append(seed)
        for split in ("test", "test2"):
            if not cache_path(bb, ds, split).exists():
                continue
            bt = D.blocks(bb, ds, split, head, seed, tuple(extra))
            if bt is None:
                continue
            r = res.setdefault(split, {"p": [], "s": [], "seeds": []})
            r["p"].append(m.predict(bt["s"], D.design(bt, names)))
            r["s"].append(bt["s"])
            r["seeds"].append(seed)
    for split, r in res.items():
        cell = D.load_cell(bb, ds, split)
        path = out_path(variant, bb, ds, split)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, uids=np.asarray(cell["uids"]), y=cell["y"], verdict=cell["verdict"], p=np.asarray(r["p"]), s=np.asarray(r["s"]), seeds=np.asarray(r["seeds"])
        )
    return job, {k: len(v["seeds"]) for k, v in res.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", required=True)
    p.add_argument("--backbones", default=",".join(D.BACKBONES))
    p.add_argument("--datasets", default=",".join(D.DATASETS))
    p.add_argument("--workers", type=int, default=30)
    a = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [(v, b, d) for v in a.variants.split(",") for b in a.backbones.split(",") for d in a.datasets.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        for job, n in ex.map(run, jobs):
            print(job, n, flush=True)


if __name__ == "__main__":
    main()
