"""Judge-light ChainUQ: route a budget of responses to the judge by expected information gain (EIG).

For every cell and head seed, R-Log is fit on validation twice: ChainUQ-A on the judge-free evidence A and ChainUQ on
(A, J). A multinomial logistic model q(J | A) of the judge's verdict level is fit on the validation responses without
their correctness labels. For a test response with evidence a,

    p_j   = ChainUQ(a, J = j)  (judge columns set to the validation mean of level j)
    EIG(a) = H2(sum_j q(j|a) p_j) - sum_j q(j|a) H2(p_j)   = I(y; J | A = a) under the fitted models,

and the k responses with the largest EIG receive the judge call (ChainUQ), the rest keep ChainUQ-A. Baselines route
by ChainUQ-A confidence (least certain first) and at random. Reports AUROC and log-loss per call budget.

Writes ``information/routing_<head>.json`` under ``CHAINUQ_ANALYSIS``.

    OMP_NUM_THREADS=1 python -m chainuq.analysis.routing --head chainuq --workers 32
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from chainuq import calibration as C
from chainuq import metrics as M
from chainuq.analysis.cache import ANALYSIS
from chainuq.analysis.information import BUDGETS, LR, h2, jlevel
from chainuq.analysis.variants import BACKBONES, DATASETS, blocks, design, small_columns

A_BLOCKS = ["u_base", "u_gap", "span", "vhat", "head:saplma", "vq"]
B_BLOCKS = A_BLOCKS + ["verdict", "ptrue"]
EXTRA = ("saplma",)


def run(job):
    head, bb, ds, seed = job
    bv, bt = blocks(bb, ds, "validation", head, seed, EXTRA), blocks(bb, ds, "test", head, seed, EXTRA)
    if bv is None or bt is None:
        return None
    yv, yt = bv["y"], bt["y"]
    XAv, XAt, XBv, XBt = design(bv, A_BLOCKS), design(bt, A_BLOCKS), design(bv, B_BLOCKS), design(bt, B_BLOCKS)
    mA = C.RLogSel(small_columns(bv, A_BLOCKS)).fit(bv["s"], XAv, yv)
    mB = C.RLogSel(small_columns(bv, B_BLOCKS)).fit(bv["s"], XBv, yv)
    pA, pB = mA.predict(bt["s"], XAt), mB.predict(bt["s"], XBt)
    jv = jlevel(bv["verdict"][:, 0])
    K = int(jv.max()) + 1
    if np.isfinite(bt["vlogq"]).all() and K == bt["vlogq"].shape[1]:
        q = np.exp(bt["vlogq"])  # the head's own predictive distribution of the judge (categorical imitation)
        q = q / q.sum(1, keepdims=True)
    else:
        q = LR().fit(np.column_stack([bv["s"], XAv]), jv).proba(np.column_stack([bt["s"], XAt]), K)
    nA = XAt.shape[1]  # judge columns follow the judge-free blocks in design order
    pj = []
    for j in range(K):
        X = XBt.copy()
        sel = jv == j
        X[:, nA] = np.nanmean(bv["verdict"][sel, 0]) if sel.any() else 0.5
        X[:, nA + 1] = np.nanmean(bv["ptrue"][sel, 0]) if sel.any() else 0.0
        pj.append(mB.predict(bt["s"], X))
    pj = np.column_stack(pj)
    pbar = (q * pj).sum(1)  # model-implied judge-free prediction: p(y|a) = sum_j q(j|a) p(y|a, j)
    eig = h2(pbar) - (q * h2(pj)).sum(1)  # >= 0 by concavity of the binary entropy
    rng = np.random.default_rng(seed)
    orders = {"eig": np.argsort(-eig), "conf": np.argsort(np.abs(pA - 0.5)), "random": rng.permutation(len(yt)), "eig-mix": np.argsort(-eig)}
    out = {"head": head, "bb": bb, "ds": ds, "seed": seed, "auroc": {}, "nll": {}}
    for name, o in orders.items():
        base = pbar if name == "eig-mix" else pA  # eig-mix: the routing of Proposition 3 with model-consistent scores
        au, ll = [], []
        for f in BUDGETS:
            k = int(round(f * len(yt)))
            m = np.zeros(len(yt), bool)
            m[o[:k]] = True
            p = np.where(m, pB, base)
            au.append(M.auroc(yt, p))
            ll.append(C.nll(yt, p))
        out["auroc"][name], out["nll"][name] = au, ll
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="chainuq")
    p.add_argument("--seeds", default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035")
    p.add_argument("--backbones", default=",".join(BACKBONES))
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--out", default="")
    a = p.parse_args()
    jobs = [(a.head, b, d, int(s)) for b in a.backbones.split(",") for d in DATASETS for s in a.seeds.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(run, jobs) if r is not None]
    out = a.out or str(ANALYSIS / "information" / f"routing_{a.head}.json")
    json.dump(rows, open(out, "w"))
    print(f"{len(rows)} rows -> {out}")
    for metric in ("auroc", "nll"):
        print(metric, "budgets", BUDGETS)
        for k in ("eig", "eig-mix", "conf", "random"):
            R = np.mean([r[metric][k] for r in rows], 0)
            lo, hi = R[0], R[-1]
            print(f"  {k:7s}", " ".join(f"{x:.4f}" for x in R), f" recovered@50%={(R[5] - lo) / (hi - lo):.3f}")


if __name__ == "__main__":
    main()
