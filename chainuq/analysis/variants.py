"""Stage-2 variants: (evidence blocks, calibrator, head) triples scored on the cached evidence of every cell.

A variant is written ``<blockset>[+block...][-block...][@calibrator][#head]``, e.g.
``B-u_reason-u_token-u_judge+ptrue+head:saplma+vq@rlogs#chainuq`` (ChainUQ) or ``S@iso#saplma`` (a probe with
isotonic regression). Blocksets: ``B`` with the judge, ``A`` judge-free, ``S`` the head score only.

By default the calibrator is fit on the validation split of the evaluated dataset and scored once on the held-out
splits (``CHAINUQ_ANALYSIS_SPLITS``, default ``test,test2``); results go to ``runs/<tag>.jsonl`` under
``CHAINUQ_ANALYSIS``. With ``--cv`` the score is instead an outer stratified 5-fold cross-validation inside the
validation split (no test access; used to select variants), written to ``runs/<tag>_cv.jsonl``.
``CHAINUQ_ANALYSIS_JUDGE=<judge>`` swaps the chain verdict and P(true) of the main judge for another judge's.

    python -m chainuq.analysis.variants --tag attribution --seeds 2026,2027,2028 \\
        --variants 'B-u_reason-u_token-u_judge+ptrue+head:saplma+vq@rlogs#chainuq,S+verdict+ptrue@lrsel#saplma'
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np
from sklearn.model_selection import StratifiedKFold

from chainuq import calibration as C
from chainuq import evidence as EV
from chainuq import metrics as M
from chainuq.analysis.cache import ANALYSIS, STATIC_NAMES, cache_path
from chainuq.artifacts import pred_path, read_jsonl
from chainuq.registry import FINAL_HEAD

BACKBONES = ["llama", "qwen8b", "ministral"]
DATASETS = ["hotpotqa", "musique", "strategyqa", "babi", "gsm8k", "math500"]

# Evidence of the final method (one judge call, the SAPLMA score as a second probe view, the head's q(J)).
EVIDENCE_B = "B-u_reason-u_token-u_judge+ptrue+head:saplma+vq"
EVIDENCE_A = "A-u_reason-u_token+head:saplma+vq"
CHAINUQ = f"{EVIDENCE_B}@rlogs#{FINAL_HEAD}"
CHAINUQ_A = f"{EVIDENCE_A}@rlogs#{FINAL_HEAD}"

# Judge swap: the chain verdict and P(true) of another judge in every cell (step labels stay the main judge's).
SWAP_JUDGE = os.environ.get("CHAINUQ_ANALYSIS_JUDGE", "")
HELD_OUT_SPLITS = tuple(os.environ.get("CHAINUQ_ANALYSIS_SPLITS", "test,test2").split(","))


@lru_cache(maxsize=64)
def load_cell(bb, ds, split):
    with open(cache_path(bb, ds, split), "rb") as f:
        cell = pickle.load(f)
    if SWAP_JUDGE:
        from chainuq.evaluation import verdict_score
        from chainuq.samples import load_judge_task

        rec = load_judge_task(SWAP_JUDGE, "final_blind", bb, ds, split)
        cell["verdict"] = np.asarray([verdict_score(rec.get(u, {})) for u in cell["uids"]], dtype=np.float64)
        cell["p_true"] = np.asarray([(rec.get(u, {}).get("p_true") or np.nan) for u in cell["uids"]], dtype=np.float64)
    return cell


@lru_cache(maxsize=256)
def load_head(method, bb, seed, ds, split):
    """(logit, u_base+u_reason matrix, predicted verdict, log q(verdict level)) aligned with the cell's UIDs; None if
    the head has no predictions for every response of the cell."""
    path = pred_path(method, bb, seed, ds, split)
    if not path.exists():
        return None
    preds = {r["uid"]: r for r in read_jsonl(path)}
    cell = load_cell(bb, ds, split)
    s, rows, vhat, logq = [], [], [], []
    for u in cell["uids"]:
        pr = preds.get(u)
        if pr is None:
            return None
        z = float(pr["logit"])
        st = pr.get("step_logits")
        rl = np.asarray(st if st else [z], dtype=np.float64)
        rp, pc = 1 / (1 + np.exp(-rl)), 1 / (1 + np.exp(-z))
        rows.append(
            [z, z, pc, rl.mean(), rl.min(), rl.max(), rl.std(), rp.mean(), rp.min(), rp.max(), rp.std(), rp.var(), abs(pc - rp.mean()), 0.0, float(len(rl))]
        )
        s.append(z)
        vhat.append(pr.get("verdict_hat", np.nan))
        logq.append(pr.get("verdict_logq") or [np.nan] * 4)
    return np.asarray(s), np.asarray(rows), np.asarray(vhat, dtype=np.float64), np.asarray(logq, dtype=np.float64)


def judge_matrix(cell):
    out = []
    for labels in cell["step_labels"]:
        arr = np.asarray([int(v) for v in (labels or []) if v in (0, 1)], dtype=np.float64)
        if len(arr):
            m = arr.mean()
            out.append([m, arr.std(), float(np.mean(arr == float(m >= 0.5))), abs(m - 0.5), float(len(arr))])
        else:
            out.append([np.nan] * 4 + [0.0])
    return np.asarray(out)


U_TOKEN = [STATIC_NAMES.index(n) for n in EV.GROUPS["u_token"]]
U_GAP = [STATIC_NAMES.index(n) for n in EV.GROUPS["u_gap"]]
U_BASE = list(range(3))
U_REASON = list(range(3, 15))


def blocks(bb, ds, split, head, seed, extra_heads=()):
    """All evidence blocks of one cell for one head seed (None if the head has no predictions)."""
    cell = load_cell(bb, ds, split)
    h = load_head(head, bb, seed, ds, split)
    if h is None:
        return None
    s, HB, vhat, logq = h
    b = {
        "s": s,
        "y": cell["y"],
        "u_base": HB[:, U_BASE],
        "u_reason": HB[:, U_REASON],
        "u_token": cell["static"][:, U_TOKEN],
        "u_gap": cell["static"][:, U_GAP],
        "u_judge": judge_matrix(cell),
        "verdict": cell["verdict"][:, None],
        "span": cell["span"],
        "vhat": vhat[:, None],
        "vlogq": logq,
    }
    b["vq"] = np.exp(b["vlogq"])  # the head's imitation distribution q(J) (probabilities enter the evidence)
    pt = np.clip(np.nan_to_num(cell["p_true"], nan=0.5), 1e-12, 1 - 1e-12)
    b["ptrue"] = np.log(pt / (1 - pt))[:, None]  # judge's token log-odds of "correct" (finer than the verbal verdict)
    for name in extra_heads:
        eh, _, eseed = name.partition("@")
        e = load_head(eh, bb, int(eseed) if eseed else seed, ds, split)
        if e is None:
            return None
        b[f"head:{name}"] = e[0][:, None]
    return b


# Evidence blocksets: "B" with the judge, "A" judge-free, "S" the head score only (with @iso: the raw head).
BLOCKSETS = {
    "B": ["u_base", "u_reason", "u_token", "u_gap", "u_judge", "verdict", "span", "vhat"],
    "A": ["u_base", "u_reason", "u_token", "u_gap", "span", "vhat"],
    "S": [],  # head score only (use with @iso for the raw head)
}


def design(b, names):
    cols = [b[n] for n in names if b[n].shape[1] and not np.all(np.isnan(b[n]))]
    return np.column_stack(cols) if cols else np.zeros((len(b["s"]), 0))


class LRSel:
    """Logistic regression with C chosen by inner 5-fold log-loss; nothing else."""

    GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)

    def fit(self, s, U, y):
        X = np.column_stack([s, U])
        best = None
        for c in self.GRID:
            oof = C.out_of_fold(lambda tr, te: C.LogReg(c).fit(X[tr], y[tr]).predict(X[te]), len(y), y)
            loss = C.nll(y, oof)
            if best is None or loss < best[0]:
                best = (loss, c)
        self.choice = best[1]
        self.m = C.LogReg(best[1]).fit(X, y)
        return self

    def predict(self, s, U):
        return self.m.predict(np.column_stack([s, U]))


class GBMCal:
    def fit(self, s, U, y):
        self.m = C.CrossFitGBM().fit(np.column_stack([s, U]), y)
        return self

    def predict(self, s, U):
        return self.m.predict(np.column_stack([s, U]))


class IsoS:
    def fit(self, s, U, y):
        self.fill = float(np.nanmedian(s))
        self.m = C.Isotonic().fit(np.nan_to_num(s, nan=self.fill), y)
        return self

    def predict(self, s, U):
        return self.m.predict(np.nan_to_num(s, nan=self.fill))


SMALL_BLOCKS = ("verdict", "ptrue", "vhat", "vlogq", "vq")


def small_columns(b, names):
    """Column indices of the compact evidence (judge outputs, predicted verdict, extra probe scores) in design()."""
    cols, off = [], 0
    for n in names:
        w = b[n].shape[1] if b[n].shape[1] and not np.all(np.isnan(b[n])) else 0
        if n in SMALL_BLOCKS or n.startswith("head:"):
            cols += list(range(off, off + w))
        off += w
    return cols


# rlogs: R-Log with evidence-set selection (the final calibrator); rlog: R-Log on the full evidence;
# lrsel: logistic regression with C chosen by inner CV; gbm: cross-fitted gradient boosting; iso: isotonic on s.
CALIBRATORS = {"rlogs": C.RLogSel, "rlog": C.RLog, "lrsel": LRSel, "gbm": GBMCal, "iso": IsoS}


def make_cal(cal, names, head, extra, bb, ds, seed):
    if cal == "rlogs":
        cols = small_columns(blocks(bb, ds, "validation", head, seed, tuple(extra)), names)
        return lambda: C.RLogSel(cols)
    return CALIBRATORS[cal]


def parse_variant(v):
    """'<blockset>[+block...][-block...][@cal][#head]' e.g. 'B-u_judge@lrsel#chainuq' or 'A+head:saplma-lf05'.
    A ``head:<run>`` block extends to the next '+', so run names may contain '-'."""
    head = "chainuq"
    if "#" in v:
        v, head = v.split("#")
    cal = "rlog"
    if "@" in v and v.rpartition("@")[2] in CALIBRATORS:
        v, _, cal = v.rpartition("@")
    tokens, cur, sign = [], "", "+"
    for ch in v:
        if ch == "-" and cur.startswith("head:"):
            cur += ch
        elif ch in "+-" and cur:
            tokens.append((sign, cur))
            cur, sign = "", ch
        elif ch in "+-":
            sign = ch
        else:
            cur += ch
    tokens.append((sign, cur))
    names = list(BLOCKSETS[tokens[0][1]])
    for sign, name in tokens[1:]:
        if sign == "+":
            names.append(name)
        else:
            names = [n for n in names if n != name]
    extra = [n.split(":", 1)[1] for n in names if n.startswith("head:")]
    return names, cal, head, extra


def scores(y, p):
    return {"auroc": M.auroc(y, p), "nll": C.nll(y, p), **M.calibration(y, p), "prauc": M.prauc(y, p)}


def run_cv(job):
    variant, bb, ds, seed = job
    names, cal, head, extra = parse_variant(variant)
    b = blocks(bb, ds, "validation", head, seed, tuple(extra))
    if b is None:
        return job, None
    y, s, X = b["y"], b["s"], design(b, names)
    oof = np.zeros(len(y))
    factory = make_cal(cal, names, head, extra, bb, ds, seed)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        m = factory().fit(s[tr], X[tr], y[tr])
        oof[te] = m.predict(s[te], X[te])
    return job, scores(y, oof)


def run_held_out(job):
    variant, bb, ds, seed = job
    names, cal, head, extra = parse_variant(variant)
    bv = blocks(bb, ds, "validation", head, seed, tuple(extra))
    if bv is None:
        return job, None
    m = make_cal(cal, names, head, extra, bb, ds, seed)().fit(bv["s"], design(bv, names), bv["y"])
    out = {}
    for split in HELD_OUT_SPLITS:
        if not cache_path(bb, ds, split).exists():
            continue
        bt = blocks(bb, ds, split, head, seed, tuple(extra))
        if bt is not None:
            out[split] = scores(bt["y"], m.predict(bt["s"], design(bt, names)))
    return job, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants", required=True)
    p.add_argument("--backbones", default=",".join(BACKBONES))
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--seeds", default="2026,2027,2028")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--tag", required=True, help="name of the run file")
    p.add_argument("--cv", action="store_true", help="cross-validate inside the validation split (no test access)")
    a = p.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    jobs = [(v, b, d, int(s)) for v in a.variants.split(",") for b in a.backbones.split(",") for d in a.datasets.split(",") for s in a.seeds.split(",")]
    out_path = ANALYSIS / "runs" / f"{a.tag}{'_cv' if a.cv else ''}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        done = {tuple(json.loads(l)["job"]) for l in open(out_path)}
    jobs = [j for j in jobs if j not in done]
    t0 = time.time()
    fn = run_cv if a.cv else run_held_out
    with ProcessPoolExecutor(a.workers) as ex, open(out_path, "a") as f:
        for i, (job, res) in enumerate(ex.map(fn, jobs, chunksize=1)):
            f.write(json.dumps({"job": list(job), "res": res}) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} {time.time() - t0:.0f}s", flush=True)
    print(f"done {len(jobs)} jobs in {time.time() - t0:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
