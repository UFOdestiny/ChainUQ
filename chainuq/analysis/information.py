"""Information decomposition between the internal-state student and the answer-blind judge.

For one evaluation cell, A = judge-free evidence of the student (head logit s, predicted verdict v_hat, span token
features, chain structure) and J = the judge's verdict (one call, discretized to its verbalized levels). With
cross-entropies of logistic models fit on the validation split and scored on the test split:

    I(y; A | J) ~ CE(y | J) - CE(y | A, J)        what the student knows beyond the judge (fusion gain)
    I(y; J | A) ~ CE(y | A) - CE(y | A, J)        what the judge knows beyond the student (value of a judge call)
    CE(y | J) - CE(y | A) = I(y; A | J) - I(y; J | A)   (student beats teacher iff the first term is larger)
    H(J | A)    ~ CE(J | A)                       imitation residual, an upper bound on I(y; J | A)

It also evaluates judge routing under a call budget: pointwise expected information gain
EIG(r) = H(sum_j q(j|A) p(y|A,j)) - sum_j q(j|A) H(p(y|A,j)) against confidence and random routing.

Writes ``information/decomposition_<head>.json`` under ``CHAINUQ_ANALYSIS`` (one row per cell and seed);
``CHAINUQ_ANALYSIS_JUDGE`` swaps the judge as in ``analysis.variants``.

    OMP_NUM_THREADS=1 python -m chainuq.analysis.information --head chainuq --workers 32
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from chainuq.analysis.cache import ANALYSIS
from chainuq.analysis.variants import BACKBONES, DATASETS, blocks

LEVELS = (0.25, 0.85, 0.97)  # verbalized verdicts cluster at {0, .05, .1}, {.9, .95}, {.99, 1}
BUDGETS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
EPS = 1e-6
EXTRA = ("saplma",)  # the second internal-state probe of the judge-free evidence


def jlevel(v):
    v = np.nan_to_num(v, nan=0.5)
    return np.digitize(v, LEVELS)


def onehot(j, k=len(LEVELS) + 1):
    return np.eye(k)[j]


def jcols(b):
    """The judge's output J beyond its level: the verbalized verdict and its token log-odds of P(true)."""
    return np.column_stack([b["verdict"], np.nan_to_num(b["ptrue"])])


def judge_free(b):
    extra = [np.nan_to_num(b[k]) for k in b if k.startswith("head:")]
    if np.isfinite(b["vq"]).all():  # categorical heads: the imitation distribution q(J) is part of A
        extra.append(b["vq"])
    return np.column_stack([b["s"], np.nan_to_num(b["vhat"][:, 0]), np.nan_to_num(b["span"]), np.nan_to_num(b["u_gap"])] + extra)


class LR:
    def __init__(self, C=0.5):
        self.C = C

    def fit(self, X, y):
        self.sc = StandardScaler().fit(X)
        Z = np.nan_to_num(self.sc.transform(X))
        self.classes = np.unique(y)
        if len(self.classes) == 1:
            self.const = self.classes[0]
            return self
        self.const = None
        self.m = LogisticRegression(C=self.C, max_iter=2000).fit(Z, y)
        return self

    def proba(self, X, k=None):
        """Class-probability matrix over classes 0..k-1 (or over the fitted classes)."""
        n = len(X)
        k = k or int(self.classes.max()) + 1
        P = np.full((n, k), EPS)
        if self.const is not None:
            P[:, int(self.const)] = 1.0
        else:
            P[:, self.m.classes_.astype(int)] = self.m.predict_proba(np.nan_to_num(self.sc.transform(X)))
        P = np.clip(P, EPS, 1)
        return P / P.sum(1, keepdims=True)


class GBM(LR):
    """Gradient-boosted trees in place of the logistic model (estimator-sensitivity check)."""

    def fit(self, X, y):
        from sklearn.ensemble import HistGradientBoostingClassifier

        self.sc = StandardScaler().fit(X)
        self.classes = np.unique(y)
        if len(self.classes) == 1:
            self.const = self.classes[0]
            return self
        self.const = None
        self.m = HistGradientBoostingClassifier(
            max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, min_samples_leaf=40, random_state=2026
        ).fit(np.nan_to_num(self.sc.transform(X)), y)
        return self


MODEL = LR


def ce(P, y):
    return float(-np.mean(np.log(P[np.arange(len(y)), y])))


def h2(p):
    p = np.clip(p, EPS, 1 - EPS)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def route_auc(y, p_a, p_b, order, budgets=BUDGETS):
    out = []
    for f in budgets:
        k = int(round(f * len(y)))
        sel = np.zeros(len(y), bool)
        sel[order[:k]] = True
        out.append(roc_auc_score(y, np.where(sel, p_b, p_a)))
    return out


def cell(job):
    head, bb, ds, seed = job
    bv, bt = blocks(bb, ds, "validation", head, seed, EXTRA), blocks(bb, ds, "test", head, seed, EXTRA)
    if bv is None or bt is None:
        return None
    yv, yt = bv["y"].astype(int), bt["y"].astype(int)
    jv, jt = jlevel(bv["verdict"][:, 0]), jlevel(bt["verdict"][:, 0])
    Av, At = judge_free(bv), judge_free(bt)
    K = len(LEVELS) + 1
    XJv, XJt = np.column_stack([onehot(jv), jcols(bv)]), np.column_stack([onehot(jt), jcols(bt)])
    mJ = MODEL().fit(np.nan_to_num(XJv), yv)
    mA = MODEL().fit(Av, yv)
    mAJ = MODEL().fit(np.column_stack([Av, np.nan_to_num(XJv)]), yv)
    mq = LR().fit(Av, jv)  # student's predictive distribution over judge levels (needs no correctness label)
    pJ, pA = mJ.proba(np.nan_to_num(XJt), 2)[:, 1], mA.proba(At, 2)[:, 1]
    pAJ = mAJ.proba(np.column_stack([At, np.nan_to_num(XJt)]), 2)[:, 1]
    q = mq.proba(At, K)
    # p(y | A, J = j) for every level j (the verdict column set to the level's mean validation verdict)
    Jv = jcols(bv)
    vbar = [np.nan_to_num(np.nanmean(Jv[jv == j], 0), nan=0.5) if np.any(jv == j) else np.r_[0.5, np.zeros(Jv.shape[1] - 1)] for j in range(K)]
    pj = np.column_stack([mAJ.proba(np.column_stack([At, np.tile(np.r_[onehot(np.array([j]))[0], vbar[j]], (len(At), 1))]), 2)[:, 1] for j in range(K)])
    pbar = (q * pj).sum(1)
    eig = h2(pbar) - (q * h2(pj)).sum(1)
    two = lambda p: np.column_stack([1 - p, p])
    res = {
        "head": head,
        "bb": bb,
        "ds": ds,
        "seed": seed,
        "n": len(yt),
        "ce_J": ce(two(pJ), yt),
        "ce_A": ce(two(pA), yt),
        "ce_AJ": ce(two(pAJ), yt),
        "ce_Jq": ce(q, jt),
        "auc_J": roc_auc_score(yt, pJ),
        "auc_A": roc_auc_score(yt, pA),
        "auc_AJ": roc_auc_score(yt, pAJ),
        "eig_mean": float(eig.mean()),
        # imitation loss of the head itself (categorical heads): -log q_head(J | A) on the test split
        "ce_Jhead": (ce(np.clip(np.exp(bt["vlogq"]), EPS, 1), jt) if np.isfinite(bt["vlogq"]).all() else float("nan")),
    }
    # the same accounting estimated on the validation split alone (5-fold cross-fitting): a prediction made before
    # any test label is seen
    from sklearn.model_selection import StratifiedKFold

    oof = {"J": np.zeros(len(yv)), "A": np.zeros(len(yv)), "AJ": np.zeros(len(yv))}
    XAJv = np.column_stack([Av, np.nan_to_num(XJv)])
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(Av, yv):
        oof["J"][te] = LR().fit(np.nan_to_num(XJv[tr]), yv[tr]).proba(np.nan_to_num(XJv[te]), 2)[:, 1]
        oof["A"][te] = LR().fit(Av[tr], yv[tr]).proba(Av[te], 2)[:, 1]
        oof["AJ"][te] = LR().fit(XAJv[tr], yv[tr]).proba(XAJv[te], 2)[:, 1]
    for k, v in oof.items():
        res[f"val_ce_{k}"] = ce(two(np.clip(v, EPS, 1 - EPS)), yv)
    # region decomposition of I(y; A | J): accepted (v >= 0.5) vs rejected responses
    acc = np.nan_to_num(bt["verdict"][:, 0], nan=0.5) >= 0.5
    for name, m in (("acc", acc), ("rej", ~acc)):
        res[f"frac_{name}"] = float(m.mean())
        res[f"I_AgJ_{name}"] = (ce(two(pJ[m]), yt[m]) - ce(two(pAJ[m]), yt[m])) if m.any() else 0.0
        res[f"ce_J_{name}"] = ce(two(pJ[m]), yt[m]) if m.any() else 0.0  # the judge's remaining uncertainty there
    # Proposition 3: residual usable information <= E[(u_J - sum_j q_j u_j)^2] / 8 for the logistic model on (A, onehot J)
    if np.isfinite(bt["vq"]).all():
        mq = LR().fit(np.column_stack([Av, onehot(jv)]), yv)
        k = Av.shape[1]
        u = mq.m.coef_[0][k:] / mq.sc.scale_[k:] if mq.const is None else np.zeros(K)
        qt = bt["vq"] / bt["vq"].sum(1, keepdims=True)
        res["quad_bound"] = float(np.mean((u[jt] - (qt * u[None, :]).sum(1)) ** 2) / 8)
        # calibrated-q forms: E[Var_q(u)]/8 and the label-free (B^2/4) E[1 - sum_l q_l^2] with B the half-range of u
        ubar = (qt * u[None, :]).sum(1)
        res["quad_cal"] = float(np.mean((qt * (u[None, :] - ubar[:, None]) ** 2).sum(1)) / 8)
        res["half_range_B"] = float((u.max() - u.min()) / 2)
        res["gini_bound"] = float(res["half_range_B"] ** 2 / 4 * np.mean(1 - (qt**2).sum(1)))
        res["q_calib_ece"] = float(np.mean(np.abs(qt.mean(0) - np.bincount(jt, minlength=K) / len(jt))))
        # level-wise calibration of q (one-vs-rest ECE with 10 equal-width bins, averaged over the levels)
        eces = []
        for l in range(K):
            pl, yl = qt[:, l], (jt == l).astype(float)
            b = np.minimum((pl * 10).astype(int), 9)
            eces.append(sum(abs(pl[b == i].mean() - yl[b == i].mean()) * (b == i).mean() for i in range(10) if (b == i).any()))
        res["q_level_ece"] = float(np.mean(eces))
        res["I_JgA_onehot"] = ce(two(mA.proba(At, 2)[:, 1]), yt) - ce(two(mq.proba(np.column_stack([At, onehot(jt)]), 2)[:, 1]), yt)
    # bootstrap over test responses of the two accounting terms (per-response log-loss differences, fitted models fixed)
    nll = lambda p, y: -(y * np.log(np.clip(p, EPS, 1)) + (1 - y) * np.log(np.clip(1 - p, EPS, 1)))
    lJ, lA, lAJ = nll(pJ, yt), nll(pA, yt), nll(pAJ, yt)
    brng = np.random.default_rng(seed)
    idx = brng.integers(0, len(yt), (200, len(yt)))
    res["boot_I_AgJ"] = (lJ - lAJ)[idx].mean(1).tolist()
    res["boot_I_JgA"] = (lA - lAJ)[idx].mean(1).tolist()
    rng = np.random.default_rng(seed)
    orders = {"eig": np.argsort(-eig), "conf": np.argsort(np.abs(pA - 0.5)), "random": rng.permutation(len(yt))}
    res["route"] = {k: route_auc(yt, pA, pAJ, o) for k, o in orders.items()}
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head", default="chainuq")
    p.add_argument("--seeds", default="2026,2027,2028,2029,2030,2031,2032,2033,2034,2035")
    p.add_argument("--backbones", default=",".join(BACKBONES))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--out", default="")
    p.add_argument("--model", default="lr", choices=("lr", "gbm"), help="gbm: estimator-sensitivity check")
    a = p.parse_args()
    global MODEL
    MODEL = GBM if a.model == "gbm" else LR
    jobs = [(a.head, b, d, int(s)) for b in a.backbones.split(",") for d in DATASETS for s in a.seeds.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        rows = [r for r in ex.map(cell, jobs) if r is not None]
    out = a.out or str(ANALYSIS / "information" / f"decomposition_{a.head}.json")
    json.dump(rows, open(out, "w"))
    keys = ["ce_J", "ce_A", "ce_AJ", "ce_Jq", "auc_J", "auc_A", "auc_AJ", "eig_mean"]
    print(f"{len(rows)} rows -> {out}")
    print("bb        ds          I(y;A|J) I(y;J|A)  H(J|A)  dCE(J-A)  aucJ   aucA   aucAJ")
    for b in a.backbones.split(","):
        for d in DATASETS:
            rr = [r for r in rows if r["bb"] == b and r["ds"] == d]
            if not rr:
                continue
            m = {k: np.mean([r[k] for r in rr]) for k in keys}
            print(
                f"{b:9s} {d:11s} {m['ce_J'] - m['ce_AJ']:8.4f} {m['ce_A'] - m['ce_AJ']:8.4f} {m['ce_Jq']:7.3f} "
                f"{m['ce_J'] - m['ce_A']:8.4f}  {m['auc_J']:.3f}  {m['auc_A']:.3f}  {m['auc_AJ']:.3f}"
            )
    for k in ("eig", "conf", "random"):
        R = np.mean([r["route"][k] for r in rows], 0)
        print(k.ljust(7), " ".join(f"{x:.4f}" for x in R))


if __name__ == "__main__":
    main()
