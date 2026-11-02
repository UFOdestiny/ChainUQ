"""bAbI within-subtask analysis: pooled vs within-task AUROC, and a subtask-prior baseline (validation accuracy of the
question's task as the score). Writes ``information/babi_subtasks.json`` under ``CHAINUQ_ANALYSIS``.

    python -m chainuq.analysis.subtasks
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import roc_auc_score

from datasets import load_from_disk

from chainuq import config
from chainuq.analysis import variants as D
from chainuq.analysis.cache import ANALYSIS
from chainuq.analysis.fusion import FUSION
from chainuq.analysis.predictions import out_path
from chainuq.registry import DATASETS

V = {"ChainUQ": D.CHAINUQ, "ChainUQ-A": D.CHAINUQ_A, "SAPLMA + J": FUSION["fusion-saplma"]}


def task_map():
    ds = load_from_disk(str(config.DATASETS_ROOT / DATASETS["babi"]["dir"] / "hf_dataset"))
    out = {}
    for split in ("validation", "test"):
        for r in ds[split]:
            t = r.get("task", 1)
            out[r["split_uid"]] = int(t) if str(t).isdigit() else 1
    return out


def within(y, p, tasks):
    num = den = 0.0
    for t in np.unique(tasks):
        m = tasks == t
        if 0 < y[m].sum() < m.sum():
            num += m.sum() * roc_auc_score(y[m], p[m])
            den += m.sum()
    return num / den


def main():
    T = task_map()
    res = {}
    for bb in D.BACKBONES:
        r = {}
        for name, v in V.items():
            zt = np.load(out_path(v, bb, "babi", "test"), allow_pickle=True)
            tasks = np.array([T.get(u, -1) for u in zt["uids"]])
            y = zt["y"].astype(int)
            r[name] = {"pooled": float(np.mean([roc_auc_score(y, p) for p in zt["p"]])), "within": float(np.mean([within(y, p, tasks) for p in zt["p"]]))}
            if name == "SAPLMA + J":
                vv = np.nan_to_num(zt["verdict"], nan=0.5)
                r["Judge-Final"] = {"pooled": roc_auc_score(y, vv), "within": within(y, vv, tasks)}
        zv = np.load(out_path(V["ChainUQ"], bb, "babi", "validation"), allow_pickle=True)
        tv = np.array([T.get(u, -1) for u in zv["uids"]])
        acc = {t: zv["y"][tv == t].mean() for t in np.unique(tv)}
        prior = np.array([acc.get(t, zv["y"].mean()) for t in tasks])
        r["Task prior"] = {"pooled": roc_auc_score(y, prior), "within": 0.5}
        res[bb] = r
    path = ANALYSIS / "information" / "babi_subtasks.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(res, indent=1))
    for bb, r in res.items():
        print(bb, {k: (round(v["pooled"], 3), round(v["within"], 3)) for k, v in r.items()})


if __name__ == "__main__":
    main()
