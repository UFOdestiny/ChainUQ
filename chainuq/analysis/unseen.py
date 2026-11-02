"""Unseen-domain control: the leave-one-domain-out head (``pipeline.train_lodo``) on its held-out OOD domain, with
R-Log fit on HotpotQA validation, so neither stage sees any information from the target domain.

Writes ``information/unseen_domain.json`` under ``CHAINUQ_ANALYSIS``.

    OMP_NUM_THREADS=1 python -m chainuq.analysis.unseen --workers 30
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

from sklearn.metrics import roc_auc_score

from chainuq.analysis import variants as D
from chainuq.analysis.cache import ANALYSIS
from chainuq.registry import FINAL_HEAD, OOD_DATASETS, SOURCE_DATASET


def unseen(job):
    variant, bb, ds, seed = job
    names, cal, head, extra = D.parse_variant(variant)
    bs, bt = D.blocks(bb, SOURCE_DATASET, "validation", head, seed, tuple(extra)), D.blocks(bb, ds, "test", head, seed, tuple(extra))
    if bs is None or bt is None:
        return None
    m = D.make_cal(cal, names, head, extra, bb, SOURCE_DATASET, seed)().fit(bs["s"], D.design(bs, names), bs["y"])
    return {"variant": variant, "bb": bb, "ds": ds, "seed": seed, "auroc": roc_auc_score(bt["y"], m.predict(bt["s"], D.design(bt, names)))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="2026,2027,2028")
    p.add_argument("--workers", type=int, default=30)
    a = p.parse_args()
    jobs = [
        (f"{ev}@rlogs#{FINAL_HEAD}-lodo-{ds}", bb, ds, int(s))
        for ev in (D.EVIDENCE_A, D.EVIDENCE_B)
        for bb in D.BACKBONES
        for ds in OOD_DATASETS
        for s in a.seeds.split(",")
    ]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(unseen, jobs) if r]
    path = ANALYSIS / "information" / "unseen_domain.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"unseen": rows}))
    print(path)


if __name__ == "__main__":
    main()
