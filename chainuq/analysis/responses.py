"""Response-level analyses of the final method on frozen test predictions (``analysis.predictions``).

* silent errors: wrong answers the judge accepts (every step valid and a chain verdict >= 0.5), and the AUROC of
  the judge, the raw head score, ChainUQ-A, ChainUQ and the probe + judge fusion baselines on the accepted and the
  flagged responses;
* fusion routing: the probe + judge fusion baselines under a judge-call budget, with the same rule as ChainUQ's
  routing (``analysis.routing``): the routed responses get the fusion score and the others the probe score alone
  (Platt-scaled on the validation split), routed by probe confidence (least certain first) or at random;
* selective risk: area under the risk-coverage curve (AURC) of ChainUQ, ChainUQ-A and every baseline;
* fusion bootstrap: paired bootstrap over test questions of the AUROC difference between ChainUQ and the stronger
  probe + judge fusion baseline (per-seed AUROCs averaged inside every resample);
* second labeler: AUROC of ChainUQ and the fusion baselines against the correctness labels of Qwen3-4B.

Writes ``responses.json`` under ``CHAINUQ_ANALYSIS``.

    python -m chainuq.analysis.responses --workers 18
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression

from chainuq import metrics as M
from chainuq.analysis import variants as D
from chainuq.analysis.cache import ANALYSIS
from chainuq.analysis.fusion import FUSION
from chainuq.analysis.information import BUDGETS
from chainuq.analysis.predictions import out_path
from chainuq.artifacts import judge_path, pred_path, read_jsonl, root
from chainuq.registry import SECOND_LABEL_JUDGE, TIER_A_BASELINES, TIER_B_BASELINES

BASELINES = TIER_A_BASELINES + TIER_B_BASELINES
JUDGE_LABELED = ["hotpotqa", "musique", "babi"]
N_BOOT = 1000


def load(variant, bb, ds, split="test"):
    z = np.load(out_path(variant, bb, ds, split))
    return {k: z[k] for k in z.files}


def baseline_scores(method, bb, ds, uids, split="test"):
    """Raw scores of a baseline, one row per available seed, aligned with ``uids``; NaN where missing."""
    mdir = root() / "preds" / method / bb
    seeds = sorted(int(p.name[4:]) for p in mdir.glob("seed*")) if mdir.exists() else []
    out = []
    for seed in seeds:
        path = pred_path(method, bb, seed, ds, split)
        if not path.exists():
            continue
        pr = {r["uid"]: r.get("logit") for r in read_jsonl(path)}
        out.append([np.nan if pr.get(u) is None else float(pr[u]) for u in uids])
    return np.asarray(out, dtype=np.float64) if out else None


def aurc(y, conf):
    """Area under the selective risk-coverage curve when answers are kept in order of decreasing confidence."""
    ok = np.isfinite(conf)
    y, conf = np.asarray(y)[ok], np.asarray(conf)[ok]
    order = np.argsort(-conf, kind="stable")
    err = 1.0 - y[order]
    return float(np.mean(np.cumsum(err) / np.arange(1, len(err) + 1)))


def accepted_mask(cell):
    """The judge accepts a response when it marks every step valid and gives a verdict >= 0.5."""
    clean = np.array([bool(l) and all(int(x) == 1 for x in l if x in (0, 1)) for l in cell["step_labels"]])
    v = np.nan_to_num(cell["verdict"], nan=0.0)
    return clean & (v >= 0.5)


def silent_errors(bb, ds):
    B, A = load(D.CHAINUQ, bb, ds), load(D.CHAINUQ_A, bb, ds)
    cell = D.load_cell(bb, ds, "test")
    y, v = B["y"], np.nan_to_num(B["verdict"], nan=0.5)
    acc = accepted_mask(cell)
    out = {
        "n": int(len(y)),
        "n_err": int((y == 0).sum()),
        "silent_frac": float(((y == 0) & acc).sum() / max(1, (y == 0).sum())),
        "accepted_frac": float(acc.mean()),
    }
    for name, m in (("accepted", acc), ("flagged", ~acc)):
        if len(np.unique(y[m])) < 2:
            continue
        out[name] = {
            "n": int(m.sum()),
            "err_rate": float(1 - y[m].mean()),
            "judge": M.auroc(y[m], v[m]),
            "head": float(np.mean([M.auroc(y[m], s[m]) for s in B["s"]])),
            "chainuq_a": float(np.mean([M.auroc(y[m], p[m]) for p in A["p"]])),
            "chainuq": float(np.mean([M.auroc(y[m], p[m]) for p in B["p"]])),
        }
        for k, variant in FUSION.items():
            out[name][k] = float(np.mean([M.auroc(y[m], p[m]) for p in load(variant, bb, ds)["p"]]))
    return out


def fusion_routing(bb, ds):
    """AUROC per judge-call budget of every fusion baseline (mean over head seeds), routed by probe confidence and
    at random."""
    out = {}
    for k, v in FUSION.items():
        zv, zt = load(v, bb, ds, "validation"), load(v, bb, ds)
        y = zt["y"]
        curves = {"conf": [], "random": []}
        for i, seed in enumerate(zt["seeds"]):
            platt = LogisticRegression().fit(zv["s"][i][:, None], zv["y"])
            p0, p1 = platt.predict_proba(zt["s"][i][:, None])[:, 1], zt["p"][i]
            orders = {"conf": np.argsort(np.abs(p0 - 0.5), kind="stable"), "random": np.random.default_rng(int(seed)).permutation(len(y))}
            for name, o in orders.items():
                au = []
                for f in BUDGETS:
                    m = np.zeros(len(y), bool)
                    m[o[: int(round(f * len(y)))]] = True
                    au.append(M.auroc(y, np.where(m, p1, p0)))
                curves[name].append(au)
        out[k] = {name: np.mean(c, 0).tolist() for name, c in curves.items()}
    return out


def selective(bb, ds):
    B, A = load(D.CHAINUQ, bb, ds), load(D.CHAINUQ_A, bb, ds)
    y, uids = B["y"], [str(u) for u in B["uids"]]
    out = {
        "chainuq": float(np.mean([aurc(y, p) for p in B["p"]])),
        "chainuq_a": float(np.mean([aurc(y, p) for p in A["p"]])),
        "oracle": aurc(y, y.astype(float)),
    }
    for m in BASELINES:
        S = baseline_scores(m, bb, ds, uids)
        if S is not None:
            out[m] = float(np.mean([aurc(y, s) for s in S]))
    return out


def fast_auroc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def fusion_bootstrap(job):
    bb, ds = job
    B = load(D.CHAINUQ, bb, ds)
    y = B["y"]
    fus = {k: load(v, bb, ds) for k, v in FUSION.items()}
    best = max(fus, key=lambda k: np.mean([M.auroc(y, p) for p in fus[k]["p"]]))
    F = fus[best]["p"]
    rng = np.random.default_rng(2026)
    diffs = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(y), len(y))
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        diffs.append(np.mean([fast_auroc(yy, pb[idx]) - fast_auroc(yy, pf[idx]) for pb, pf in zip(B["p"], F)]))
    point = float(np.mean([M.auroc(y, pb) - M.auroc(y, pf) for pb, pf in zip(B["p"], F)]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return job, {"best_fusion": best, "diff": point, "lo": float(lo), "hi": float(hi), "sig": "+" if lo > 0 else ("-" if hi < 0 else "0")}


def second_labeler(bb, ds):
    path = judge_path(SECOND_LABEL_JUDGE, "correct", bb, ds, "test")
    if not path.exists():
        return None
    alt = {r["uid"]: r.get("value") for r in read_jsonl(path)}
    out = {}
    for k, v in [("chainuq", D.CHAINUQ)] + list(FUSION.items()):
        z = load(v, bb, ds)
        y2 = np.array([alt.get(str(u)) if alt.get(str(u)) in (0, 1) else -1 for u in z["uids"]])
        ok = y2 >= 0
        out[k] = float(np.mean([M.auroc(y2[ok], p[ok]) for p in z["p"]]))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=18)
    a = p.parse_args()
    jobs = [(bb, ds) for bb in D.BACKBONES for ds in D.DATASETS]
    res = {"silent": {}, "aurc": {}, "bootstrap": {}, "relabel": {}, "fusion_routing": {}}
    with ProcessPoolExecutor(a.workers) as ex:
        for (bb, ds), r in ex.map(fusion_bootstrap, jobs):
            res["bootstrap"][f"{bb}/{ds}"] = r
    for bb, ds in jobs:
        key = f"{bb}/{ds}"
        res["silent"][key] = silent_errors(bb, ds)
        res["aurc"][key] = selective(bb, ds)
        res["fusion_routing"][key] = fusion_routing(bb, ds)
        if ds in JUDGE_LABELED:
            res["relabel"][key] = second_labeler(bb, ds)
        print(key, flush=True)
    path = ANALYSIS / "responses.json"
    path.write_text(json.dumps(res, indent=1))
    print(path)


if __name__ == "__main__":
    main()
