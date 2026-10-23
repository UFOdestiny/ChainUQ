#!/usr/bin/env python3
"""Method development for Stage 2 on VALIDATION DATA ONLY (test splits are never read).

Every candidate is scored by 5-fold cross-fitting inside each dataset's validation
split: fit on 4 folds, predict the held-out fold, and compute the metrics on the
pooled out-of-fold predictions. This is the only signal used to choose the final
design; test is evaluated once afterwards by pipeline/evaluate.py.

Example:
    python pipeline/dev_stage2.py --backbone llama --heads chainuq,saplma --seeds 2026,2027,2028
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq import calibration as C
from cuq import metrics as M
from cuq import ufeatures as UF
from cuq.io import index_by_uid, read_jsonl
from cuq.registry import ARTIFACTS, BACKBONES, DATASETS, MAIN_JUDGE, SOURCE_DATASET, feat_dir, judge_path, pred_path
from cuq.store import load_samples

U_KEYS = ["tok", "ll", "top1", "margin", "ent"]


class ValCell:
    def __init__(self, backbone, ds, hidden):
        self.samples = load_samples(backbone, ds, "validation", feature_keys=U_KEYS)
        self.y = np.asarray([s.y for s in self.samples])
        self.hidden = hidden
        self._static = {}
        fb = judge_path(MAIN_JUDGE, "final_blind", backbone, ds, "validation")
        fbr = index_by_uid(read_jsonl(fb)) if fb.exists() else {}
        self.jf = np.asarray([_jf(fbr.get(s.uid, {})) for s in self.samples])
        nc = judge_path(MAIN_JUDGE, "steps_noconcl", backbone, ds, "validation")
        ncr = index_by_uid(read_jsonl(nc)) if nc.exists() else {}
        self.noconcl = {u: r["value"] for u, r in ncr.items() if isinstance(r["value"], list)}

    def U(self, preds, step_override=None):
        rows = []
        for s in self.samples:
            if s.uid not in self._static:
                self._static[s.uid] = UF.static_features(steps=s.steps, conclusion=s.conclusion, n_hops=s.gen["n_hops"],
                                                         feat=s.feat, hidden_dim=self.hidden)
            sl = s.step_labels if step_override is None else step_override.get(s.uid)
            p = preds[s.uid]
            rows.append(UF.sample_features(logit=p["logit"], step_logits=p.get("step_logits"), steps=s.steps,
                                           conclusion=s.conclusion, n_hops=s.gen["n_hops"], feat=s.feat,
                                           hidden_dim=self.hidden, step_labels=sl, static=self._static[s.uid]))
        return UF.matrix(rows)

    def extra(self):
        if not hasattr(self, "_extra"):
            self._extra = UF.matrix([UF.extra_features(steps=x.steps, conclusion=x.conclusion, feat=x.feat)
                                     for x in self.samples], UF.EXTRA_NAMES)
        return self._extra

    def s(self, preds):
        return np.asarray([preds[x.uid]["logit"] for x in self.samples], dtype=np.float64)

    def branches(self, preds):
        """Token-only / layer-only scores of PCS heads; None when absent."""
        if "branch_tok" not in next(iter(preds.values())):
            return None
        return np.asarray([[preds[x.uid]["branch_tok"], preds[x.uid]["branch_lay"]] for x in self.samples], dtype=np.float64)

    def vhat(self, preds):
        """Head-predicted chain verdict (PCS heads only); None when the head has no such output."""
        if "verdict_hat" not in next(iter(preds.values())):
            return None
        return np.asarray([preds[x.uid]["verdict_hat"] for x in self.samples], dtype=np.float64)


def _ensemble(pred_list):
    """Average of per-head standardised logits (and step logits) -- a fixed, untuned combination."""
    if len(pred_list) == 1:
        return pred_list[0]
    stats = []
    for pr in pred_list:
        v = np.asarray([r["logit"] for r in pr.values()])
        stats.append((v.mean(), v.std() + 1e-8))
    out = {}
    for uid in pred_list[0]:
        zs = [(pr[uid]["logit"] - m) / sd for pr, (m, sd) in zip(pred_list, stats)]
        steps = [pr[uid].get("step_logits") for pr in pred_list]
        st = None
        if all(steps):
            st = list(np.mean([[(x - m) / sd for x in sl] for sl, (m, sd) in zip(steps, stats)], axis=0))
        out[uid] = {"logit": float(np.mean(zs)), "step_logits": st}
    return out


def _jf(r):
    pv, pt = r.get("p_verbal"), r.get("p_true")
    if pv is None and pt is None:
        return np.nan
    return (pv if pv is not None else pt) + 1e-3 * (pt if pt is not None else 0.0)


def crossfit(fit_predict, y, n_splits=5, seed=2026):
    """Out-of-fold predictions; fit_predict(train_idx, test_idx) -> probabilities for test_idx."""
    out = np.full(len(y), np.nan)
    for tr, te in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        out[te] = fit_predict(tr, te)
    return out


def scores(y, p):
    return {"auroc": M.auroc(y, p), "prauc": M.prauc(y, p), **M.calibration(y, p)}


def candidates(s, U, jf, y, E=None):
    J = [UF.FEATURE_NAMES.index(n) for n in UF.GROUPS["u_judge"]]
    notJ = [i for i in range(U.shape[1]) if i not in J]
    jfc = np.nan_to_num(jf, nan=np.nanmean(jf) if np.isfinite(jf).any() else 0.5)
    X_full = np.column_stack([U, jfc])
    c = {
        "iso(s)": lambda tr, te: C.Isotonic().fit(s[tr], y[tr]).predict(s[te]),
        "rlog(s,u) [current]": lambda tr, te: C.RLog().fit(s[tr], U[tr], y[tr]).predict(s[te], U[te]),
        "iso(judge_final)": lambda tr, te: C.Isotonic().fit(jfc[tr], y[tr]).predict(jfc[te]),
        "rlog(s,u+jf)": lambda tr, te: C.RLog().fit(s[tr], X_full[tr], y[tr]).predict(s[te], X_full[te]),
        "rlog(s,u+jf) C=0.05": lambda tr, te: C.RLog(C=0.05).fit(s[tr], X_full[tr], y[tr]).predict(s[te], X_full[te]),
        "lr(s,u_judge,jf)": lambda tr, te: C.LogReg(0.25).fit(np.column_stack([s, U[:, J], jfc])[tr], y[tr]).predict(
            np.column_stack([s, U[:, J], jfc])[te]),
        "gbm(s,u+jf)+iso": lambda tr, te: _gbm(np.column_stack([s, X_full]), y, tr, te),
        "RLOG2(s,u+jf) [v2]": lambda tr, te: C.RLog2().fit(s[tr], X_full[tr], y[tr]).predict(s[te], X_full[te]),
        "RLOG2(s,u+jf+extra) [v2+]": lambda tr, te: C.RLog2().fit(s[tr], np.column_stack([X_full, E])[tr], y[tr]).predict(
            s[te], np.column_stack([X_full, E])[te]),
        "RLOG2(s,u-judge+extra) [v2-A+]": lambda tr, te: C.RLog2().fit(s[tr], np.column_stack([U[:, notJ], E])[tr], y[tr]).predict(
            s[te], np.column_stack([U[:, notJ], E])[te]),
        "RLOG2b(s,u+jf+extra) [v2b+]": lambda tr, te: C.RLog2b().fit(s[tr], np.column_stack([X_full, E])[tr], y[tr]).predict(
            s[te], np.column_stack([X_full, E])[te]),
        "RLOG2b(s,u-judge+extra) [v2b-A+]": lambda tr, te: C.RLog2b().fit(s[tr], np.column_stack([U[:, notJ], E])[tr], y[tr]).predict(
            s[te], np.column_stack([U[:, notJ], E])[te]),
        "RLOG2c(s,u+jf+extra) [v2c+]": lambda tr, te: C.RLog2c().fit(s[tr], np.column_stack([X_full, E])[tr], y[tr]).predict(
            s[te], np.column_stack([X_full, E])[te]),
        "RLOG2c(s,u-judge+extra) [v2c-A+]": lambda tr, te: C.RLog2c().fit(s[tr], np.column_stack([U[:, notJ], E])[tr], y[tr]).predict(
            s[te], np.column_stack([U[:, notJ], E])[te]),
        "RLOG2d(s,u+jf+extra) [v2d+]": lambda tr, te: C.RLog2d().fit(s[tr], np.column_stack([X_full, E])[tr], y[tr]).predict(
            s[te], np.column_stack([X_full, E])[te]),
        "RLOG2d(s,u-judge+extra) [v2d-A+]": lambda tr, te: C.RLog2d().fit(s[tr], np.column_stack([U[:, notJ], E])[tr], y[tr]).predict(
            s[te], np.column_stack([U[:, notJ], E])[te]),
        "SMOOTH(RLOG2c)(s,u+jf+extra) [v2cs+]": lambda tr, te: C.Smoothed(C.RLog2c).fit(s[tr], np.column_stack([X_full, E])[tr], y[tr]).predict(
            s[te], np.column_stack([X_full, E])[te]),
        "SMOOTH(RLOG2c)(s,u-judge+extra) [v2cs-A+]": lambda tr, te: C.Smoothed(C.RLog2c).fit(s[tr], np.column_stack([U[:, notJ], E])[tr], y[tr]).predict(
            s[te], np.column_stack([U[:, notJ], E])[te]),
        # Fixed-capacity R-Log (the configuration most often selected by RLog2c): cheap head comparisons.
        "RLOGCF05p(s,u+jf+extra) [fix+]": lambda tr, te: C.RLogCF(C=0.05, final="platt").fit(s[tr], np.column_stack([X_full, E])[tr], y[tr]).predict(
            s[te], np.column_stack([X_full, E])[te]),
        "RLOGCF05p(s,u-judge+extra) [fix-A+]": lambda tr, te: C.RLogCF(C=0.05, final="platt").fit(s[tr], np.column_stack([U[:, notJ], E])[tr], y[tr]).predict(
            s[te], np.column_stack([U[:, notJ], E])[te]),
        # Tier A (no judge at inference): drop the step-judge features, no chain-level verdict.
        "RLOG2(s,u-judge) [v2-A]": lambda tr, te: C.RLog2().fit(s[tr], U[tr][:, notJ], y[tr]).predict(s[te], U[te][:, notJ]),
    }
    return c


def _gbm(X, y, tr, te):
    Xi = C._Imputer().fit(X[tr])
    m = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0,
                                       min_samples_leaf=40, random_state=2026)
    m.fit(Xi.transform(X[tr]), y[tr])
    return C._clip(m.predict_proba(Xi.transform(X[te]))[:, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--heads", default="chainuq,saplma")
    p.add_argument("--seeds", default="2026,2027,2028")
    p.add_argument("--datasets", default=",".join(DATASETS))
    p.add_argument("--out", default="")
    p.add_argument("--cands", default="", help="comma-separated substrings; keep only matching candidates")
    args = p.parse_args()
    hidden = json.loads((feat_dir(args.backbone, SOURCE_DATASET, "train") / "DONE.json").read_text())["hidden_size"]
    seeds = [int(x) for x in args.seeds.split(",")]
    report = {}
    for ds in args.datasets.split(","):
        cell = ValCell(args.backbone, ds, hidden)
        y = cell.y
        for head in args.heads.split(","):
            agg = {}
            for seed in seeds:
                parts = head.split(":", 1)[1].split("+") if head.startswith("ens:") else [head]
                pths = [pred_path(h, args.backbone, seed, ds, "validation") for h in parts]
                if not all(p.exists() for p in pths):
                    continue
                preds = _ensemble([index_by_uid(read_jsonl(p)) for p in pths])
                s, U = cell.s(preds), cell.U(preds)
                ok = np.isfinite(s)  # sampling baselines only cover the first 1000 questions
                E = cell.extra()
                vh = cell.vhat(preds)
                if vh is not None:  # judge-free feature: the head's own estimate of the judge's verdict
                    E = np.column_stack([E, vh])
                if not ok.all():
                    s, U, jf_, y_, E = s[ok], U[ok], cell.jf[ok], y[ok], E[ok]
                else:
                    jf_, y_ = cell.jf, y
                cands = candidates(s, U, jf_, y_, E)
                br = cell.branches(preds)
                if br is not None:  # same calibrators with the two branch scores as extra evidence: "[br...]"
                    Eb = np.column_stack([cell.extra(), vh, br]) if vh is not None else np.column_stack([cell.extra(), br])
                    if not ok.all():
                        Eb = Eb[ok]
                    cands.update({k.replace("[fix", "[br"): v for k, v in candidates(s, U, jf_, y_, Eb).items() if "[fix" in k})
                if args.cands:
                    cands = {k: v for k, v in cands.items() if any(c in k for c in args.cands.split(","))}
                for name, fp in cands.items():
                    agg.setdefault(name, []).append(scores(y_, crossfit(fp, y_)))
            if not agg:
                continue
            report[f"{ds}/{head}"] = {k: {m: float(np.mean([r[m] for r in v])) for m in v[0]} for k, v in agg.items()}
            print(f"\n== {ds} / {head}  (val n={len(y)}, pos={y.mean():.3f}, {len(seeds)} seeds, 5-fold cross-fit)")
            for k, v in report[f"{ds}/{head}"].items():
                print(f"  {k:24s} auroc={v['auroc']:.3f} prauc={v['prauc']:.3f} ece={v['ece']:.3f} "
                      f"brier={v['brier']:.3f} risk90={v['risk90']:.3f}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
