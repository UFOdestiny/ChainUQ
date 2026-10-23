#!/usr/bin/env python3
"""Stage 5: calibration + every table/analysis of the experiment plan, per backbone.

Protocol: every calibrator / threshold is fit on the validation split and
applied once to test. Two OOD protocols are reported:
  target  calibrator fit on the target dataset's validation split (original paper);
  strict  calibrator fit on HotpotQA validation only and transferred unchanged.

Outputs ``results/<backbone>/eval.json`` (all numbers, mean/std over seeds) and
``results/<backbone>/tables.md`` (human-readable).

Example:
    python pipeline/evaluate.py --backbone llama
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq import calibration as C
from cuq import metrics as M
from cuq import ufeatures as UF
from cuq.io import index_by_uid, read_jsonl
from cuq.registry import (ARTIFACTS, BACKBONES, DATASETS, FINAL_HEAD, HEAD_SEEDS, JUDGES, MAIN_JUDGE, SOURCE_DATASET, feat_dir,
                          gen_path, judge_path, pred_path, results_dir)
from cuq.store import load_samples

TIER_A_UNSUP = ["mcp", "ppl", "mte", "ccp", "cotuq", "se", "selfcheck"]
TIER_A_SUP = ["saplma", "factoscope", "lookback", "uhead", "meanpool-mlp", "attnpool-mlp"]
CHAINUQ_VARIANTS = ["chainuq", "chainuq-lb", "chainuq-cs", "chainuq-cf", "chainuq-fr", "chainuq2bd", "chainuq2bd-core"]
TIER_B_SCORES = ["judge_final", "judge_step", "rm"]
LABEL_JUDGE_2 = "qwen4b"
TEST = "test"  # held-out split scored by this run; "test2" = fresh set built after two looks at "test"
PAPER_ONLY = False  # --paper_only: baselines, final ChainUQ / ChainUQ-A, label robustness, judge agreement
U_KEYS = ["tok", "ll", "top1", "margin", "ent"]


def available_methods(backbone):
    root = ARTIFACTS / "preds"
    out = {}
    if not root.exists():
        return out
    for mdir in sorted(root.iterdir()):
        seeds = sorted(int(p.name[4:]) for p in (mdir / backbone).glob("seed*")
                       if p.is_dir() and p.name[4:].isdigit() and any(p.rglob("*.jsonl")))
        if seeds:
            out[mdir.name] = seeds
    return out


def load_preds(method, backbone, seed, ds, split, fmt="step"):
    path = pred_path(method, backbone, seed, ds, split, fmt)
    return index_by_uid(read_jsonl(path)) if path.exists() else None


class Cell:
    """Validation + test samples of one dataset with cached u(r) building blocks."""

    def __init__(self, backbone, ds, fmt, hidden_dim):
        self.ds, self.fmt, self.hidden_dim = ds, fmt, hidden_dim
        self._static_cache = {}
        self.val = load_samples(backbone, ds, "validation", fmt, feature_keys=U_KEYS) if fmt == "step" else []
        self.test = load_samples(backbone, ds, TEST, fmt, feature_keys=U_KEYS)
        self.backbone = backbone

    def labels(self, split):
        return np.asarray([s.y for s in (self.val if split == "validation" else self.test)])

    def extra(self, split):
        """R-Log v2 extension: conclusion-span and weakest-step token uncertainty."""
        samples = self.val if split == "validation" else self.test
        return UF.matrix([UF.extra_features(steps=x.steps, conclusion=x.conclusion, feat=x.feat) for x in samples],
                         UF.EXTRA_NAMES)

    def verdict(self, split):
        """Chain-level judge verdict (answer-blind Judge-Final probability), main judge."""
        samples = self.val if split == "validation" else self.test
        path = judge_path(MAIN_JUDGE, "final_blind", self.backbone, self.ds, split, self.fmt)
        rec = index_by_uid(read_jsonl(path)) if path.exists() else {}
        out = []
        for x in samples:
            r = rec.get(x.uid, {})
            pv, pt = r.get("p_verbal"), r.get("p_true")
            out.append(np.nan if pv is None and pt is None else (pv if pv is not None else pt) + 1e-3 * (pt or 0.0))
        return np.asarray(out, dtype=np.float64)

    def _static(self, s):
        if s.uid not in self._static_cache:
            self._static_cache[s.uid] = UF.static_features(steps=s.steps, conclusion=s.conclusion,
                                                           n_hops=s.gen["n_hops"], feat=s.feat, hidden_dim=self.hidden_dim)
        return self._static_cache[s.uid]

    def u_matrix(self, split, preds, step_labels_override=None):
        samples = self.val if split == "validation" else self.test
        rows = []
        for s in samples:
            pr = preds[s.uid]
            sl = s.step_labels if step_labels_override is None else step_labels_override.get(s.uid)
            rows.append(UF.sample_features(logit=pr["logit"], step_logits=pr.get("step_logits"), steps=s.steps,
                                           conclusion=s.conclusion, n_hops=s.gen["n_hops"], feat=s.feat,
                                           hidden_dim=self.hidden_dim, step_labels=sl, static=self._static(s)))
        return UF.matrix(rows)

    def with_vhat(self, E, split, preds):
        """Append the head-predicted chain verdict (PCS heads) to the judge-free evidence E."""
        samples = self.val if split == "validation" else self.test
        if not preds or "verdict_hat" not in next(iter(preds.values())):
            return E
        return np.column_stack([E, [preds[s.uid].get("verdict_hat", np.nan) for s in samples]])

    def scores(self, split, preds):
        samples = self.val if split == "validation" else self.test
        return np.asarray([preds[s.uid]["logit"] if s.uid in preds else np.nan for s in samples], dtype=np.float64)


def cols(names):
    return [UF.FEATURE_NAMES.index(n) for n in names]


def eval_scores(y_val, s_val, y_te, s_te, calibrated_te=None):
    """Discrimination on test (threshold from val) + calibration of ``calibrated_te`` if given."""
    ok_v, ok_t = np.isfinite(s_val), np.isfinite(s_te)
    thr = M.best_threshold(y_val[ok_v], s_val[ok_v])
    out = M.discrimination(y_te[ok_t], s_te[ok_t], thr)
    out["n_test"] = int(ok_t.sum())
    if calibrated_te is not None:
        out.update(M.calibration(y_te[ok_t], calibrated_te[ok_t]))
    return out


def scalar_calibrated(name, y_val, s_val, s_te):
    ok = np.isfinite(s_val)
    m = C.SCALAR[name]().fit(s_val[ok], y_val[ok])
    out = np.full_like(s_te, np.nan)
    good = np.isfinite(s_te)
    out[good] = m.predict(s_te[good])
    return out


def agg(rows):
    """Mean/std over seeds for a list of metric dicts."""
    keys = [k for k in rows[0] if isinstance(rows[0][k], (int, float))]
    return {k: {"mean": float(np.nanmean([r[k] for r in rows])), "std": float(np.nanstd([r[k] for r in rows])),
                "n_seeds": len(rows)} for k in keys}


def paired_bootstrap_auroc(y, scores_a, scores_b, n_boot=1000, seed=2026):
    """95% CI of mean-over-seeds AUROC(A) - AUROC(B) on test. scores_*: [n_seeds, n]."""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        a = np.mean([M.auroc(y[idx], s[idx]) for s in scores_a])
        b = np.mean([M.auroc(y[idx], s[idx]) for s in scores_b])
        diffs.append(a - b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff_mean": float(np.mean(diffs)), "ci95": [float(lo), float(hi)]}


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = np.mean(a) * np.mean(b) + (1 - np.mean(a)) * (1 - np.mean(b))
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def evaluate_backbone(backbone, n_perm=20):
    methods = available_methods(backbone)
    hidden = json.loads((feat_dir(backbone, SOURCE_DATASET, "train") / "DONE.json").read_text())["hidden_size"]
    res = {"backbone": backbone, "methods": methods, "datasets": {}}
    src_cell = Cell(backbone, SOURCE_DATASET, "step", hidden)

    for ds in DATASETS:
        if not gen_path(backbone, ds, TEST).exists():  # StrategyQA has no test2
            continue
        cell = src_cell if ds == SOURCE_DATASET else Cell(backbone, ds, "step", hidden)
        if not cell.test or not cell.val:
            continue
        yv, yt = cell.labels("validation"), cell.labels(TEST)
        R = {"n_val": len(yv), "n_test": len(yt), "test_pos_rate": float(yt.mean()),
             "tier_a": {}, "tier_b": {}, "calibration": {}, "decomposition": {}, "loo": {},
             "strict_ood": {}, "permutation": {}, "judge_swap": {}, "quadrants": {}, "rlog_ablation": {}}
        outs = defaultdict(list)  # method -> [(raw test score, calibrated test prob)] per seed, for label robustness

        # ---- Tier A: raw scores, plus ISO-calibrated calibration metrics -------------------------
        for m in TIER_A_UNSUP + TIER_A_SUP + CHAINUQ_VARIANTS:
            if m not in methods:
                continue
            rows = []
            for seed in methods[m]:
                pv, pt = load_preds(m, backbone, seed, ds, "validation"), load_preds(m, backbone, seed, ds, TEST)
                if pv is None or pt is None:
                    continue
                sv, st = cell.scores("validation", pv), cell.scores(TEST, pt)
                cal = scalar_calibrated("iso", yv, sv, st)
                rows.append(eval_scores(yv, sv, yt, st, cal))
                outs[m].append((st, cal))
            if rows:
                R["tier_a"][m] = agg(rows)

        # ---- Tier B: judge-access scores ---------------------------------------------------------
        for m in TIER_B_SCORES:
            if m in methods:
                pv, pt = load_preds(m, backbone, methods[m][0], ds, "validation"), load_preds(m, backbone, methods[m][0], ds, TEST)
                if pv is not None and pt is not None:
                    sv, st = cell.scores("validation", pv), cell.scores(TEST, pt)
                    cal = scalar_calibrated("iso", yv, sv, st)
                    R["tier_b"][m] = agg([eval_scores(yv, sv, yt, st, cal)])
                    outs[m].append((st, cal))

        # ---- Heads + R-Log (Tier B), calibration table, decomposition, permutation --------------
        for head in ([] if PAPER_ONLY else ["chainuq", "uhead", "saplma"]):
            if head not in methods:
                continue
            full, lr_u, cal_rows = [], [], defaultdict(list)
            dec_rows, loo_rows, perm_rows, strict_rows, abl_rows = defaultdict(list), defaultdict(list), [], [], defaultdict(list)
            for seed in methods[head]:
                pv, pt = load_preds(head, backbone, seed, ds, "validation"), load_preds(head, backbone, seed, ds, TEST)
                if pv is None or pt is None:
                    continue
                sv, st = cell.scores("validation", pv), cell.scores(TEST, pt)
                Uv, Ut = cell.u_matrix("validation", pv), cell.u_matrix(TEST, pt)
                rl = C.RLog().fit(sv, Uv, yv)
                pr_v, pr_t = rl.predict(sv, Uv), rl.predict(st, Ut)
                full.append(eval_scores(yv, pr_v, yt, pr_t, pr_t))
                outs[f"{head}+rlog"].append((pr_t, pr_t))
                # LR on evidence that does not come from the head (u_token, u_gap, u_judge).
                H = cols(UF.GROUPS["u_token"] + UF.GROUPS["u_gap"] + UF.GROUPS["u_judge"])
                lu = C.LogReg().fit(Uv[:, H], yv)
                lr_u.append(eval_scores(yv, lu.predict(Uv[:, H]), yt, lu.predict(Ut[:, H]), lu.predict(Ut[:, H])))
                if head != "chainuq":
                    continue
                for name in ("ps", "ts", "iso"):
                    cal = scalar_calibrated(name, yv, sv, st)
                    cal_rows[name].append(eval_scores(yv, sv, yt, st, cal))
                bwh = C.BWH().fit(sv, Uv, yv)
                cal_rows["bwh"].append(eval_scores(yv, bwh.predict(sv, Uv), yt, bwh.predict(st, Ut), bwh.predict(st, Ut)))
                cal_rows["rlog"].append(full[-1])

                # E2: what does each evidence source add?
                J = cols(UF.GROUPS["u_judge"])
                notJ = [i for i in range(Uv.shape[1]) if i not in J]
                for name, fit in {
                    "iso(s)": lambda: (scalar_calibrated("iso", yv, sv, sv), scalar_calibrated("iso", yv, sv, st)),
                    "lr(u_judge)": lambda: _lr(Uv[:, J], Ut[:, J], yv),
                    "lr(s,u_judge)": lambda: _lr(np.column_stack([sv, Uv[:, J]]), np.column_stack([st, Ut[:, J]]), yv),
                    "rlog(s,u-judge)": lambda: _rlog(sv, st, Uv[:, notJ], Ut[:, notJ], yv),
                    "rlog(s,u)": lambda: (pr_v, pr_t),
                }.items():
                    a, b = fit()
                    dec_rows[name].append(eval_scores(yv, a, yt, b, b))
                for g, names in UF.GROUPS.items():
                    keep = [i for i in range(Uv.shape[1]) if i not in cols(names)]
                    a, b = _rlog(sv, st, Uv[:, keep], Ut[:, keep], yv)
                    loo_rows[f"-{g}"].append(eval_scores(yv, a, yt, b, b))

                # R-Log internal ablations (E6).
                for name, kw in {"single_alpha": dict(single_alpha=True), "no_final_iso": dict(final_iso=False)}.items():
                    v = C.RLogVariant(**kw).fit(sv, Uv, yv)
                    abl_rows[name].append(eval_scores(yv, v.predict(sv, Uv), yt, v.predict(st, Ut), v.predict(st, Ut)))

                # Permutation test: break the link between u_judge and the sample.
                rng = np.random.default_rng(seed)
                for _ in range(n_perm):
                    Up = Ut.copy()
                    Up[:, J] = Ut[rng.permutation(len(yt))][:, J]
                    p = rl.predict(st, Up)
                    perm_rows.append(eval_scores(yv, pr_v, yt, p, p))

                # Strict OOD: calibrator fit on HotpotQA validation only.
                if ds != SOURCE_DATASET:
                    sp = load_preds(head, backbone, seed, SOURCE_DATASET, "validation")
                    s_src = src_cell.scores("validation", sp)
                    U_src = src_cell.u_matrix("validation", sp)
                    y_src = src_cell.labels("validation")
                    rl_s = C.RLog().fit(s_src, U_src, y_src)
                    iso_s = C.Isotonic().fit(s_src, y_src)
                    strict_rows.append({
                        **{f"rlog_{k}": v for k, v in eval_scores(y_src, rl_s.predict(s_src, U_src), yt, rl_s.predict(st, Ut), rl_s.predict(st, Ut)).items()},
                        **{f"iso_{k}": v for k, v in eval_scores(y_src, iso_s.predict(s_src), yt, iso_s.predict(st), iso_s.predict(st)).items()},
                    })

                # E3: swap the step judge (validation + test labels only).
                # "gptoss-noconcl" is the control whose step judge never sees the model's conclusion.
                for name, judge, task in [(j, j, "steps") for j in sorted(set(list(JUDGES) + ["nli"]))] + \
                        [("gptoss-noconcl", MAIN_JUDGE, "steps_noconcl")]:
                    jv = _judge_steps(judge, backbone, ds, "validation", task)
                    jt = _judge_steps(judge, backbone, ds, TEST, task)
                    if not jv or not jt:
                        continue
                    Uv2, Ut2 = cell.u_matrix("validation", pv, jv), cell.u_matrix(TEST, pt, jt)
                    r2 = C.RLog().fit(sv, Uv2, yv)
                    R["judge_swap"].setdefault(name, []).append(
                        eval_scores(yv, r2.predict(sv, Uv2), yt, r2.predict(st, Ut2), r2.predict(st, Ut2)))

                # E7: quadrants (answer correct x all steps judged consistent).
                if seed == methods[head][0]:
                    cons = np.asarray([1 if s.step_labels and all(v == 1 for v in s.step_labels) else 0 for s in cell.test])
                    for yy in (0, 1):
                        for cc in (0, 1):
                            m = (yt == yy) & (cons == cc)
                            if m.any():
                                R["quadrants"][f"y{yy}_consistent{cc}"] = {
                                    "share": float(m.mean()), "n": int(m.sum()),
                                    "mean_conf_iso": float(scalar_calibrated("iso", yv, sv, st)[m].mean()),
                                    "mean_conf_rlog": float(pr_t[m].mean()),
                                }
            if full:
                R["tier_b"][f"{head}+rlog"] = agg(full)
                if head == "chainuq":  # head-independent evidence only: identical for every head
                    R["tier_b"]["lr(u)"] = agg(lr_u)
            if head == "chainuq" and cal_rows:
                R["calibration"] = {k: agg(v) for k, v in cal_rows.items()}
                R["decomposition"] = {k: agg(v) for k, v in dec_rows.items()}
                R["loo"] = {k: agg(v) for k, v in loo_rows.items()}
                R["rlog_ablation"] = {k: agg(v) for k, v in abl_rows.items()}
                R["permutation"] = agg(perm_rows) if perm_rows else {}
                R["strict_ood"] = agg(strict_rows) if strict_rows else {}
                R["judge_swap"] = {k: agg(v) for k, v in R["judge_swap"].items()}

        # ---- ChainUQ v2 (designed on validation only; see pipeline/dev_stage2.py) ------------------
        Ev, Et = cell.extra("validation"), cell.extra(TEST)
        jv, jt = cell.verdict("validation"), cell.verdict(TEST)
        J = cols(UF.GROUPS["u_judge"])
        notJ = [i for i in range(len(UF.FEATURE_NAMES)) if i not in J]
        v2_rows = defaultdict(list)
        v2_variants = {
            # name: (head, tier, builder of the evidence matrix from (U, E, verdict), calibrator class)
            "chainuq-v2-A": ("chainuq2bd", "tier_a", lambda U, E, v: np.column_stack([U[:, notJ], E]), C.RLog2b),
            "chainuq-v2": ("chainuq2bd", "tier_b", lambda U, E, v: np.column_stack([U, v, E]), C.RLog2b),
            # v3: identical inputs, every calibration step cross-fitted (R-Log v2c). Adopted on validation
            # after the v2 test evaluation; both are reported.
            "chainuq-v3-A": (FINAL_HEAD, "tier_a", lambda U, E, v: np.column_stack([U[:, notJ], E]), C.RLog2c),
            "chainuq-v3": (FINAL_HEAD, "tier_b", lambda U, E, v: np.column_stack([U, v, E]), C.RLog2c),
            "abl: v1 head + R-Log v2": ("chainuq", "ablation", lambda U, E, v: np.column_stack([U, v, E]), C.RLog2b),
            "abl: v2 head + R-Log v1": ("chainuq2bd", "ablation", lambda U, E, v: U, C.RLog),
            "abl: v2 - chain verdict": ("chainuq2bd", "ablation", lambda U, E, v: np.column_stack([U, E]), C.RLog2b),
            "abl: v2 - extra features": ("chainuq2bd", "ablation", lambda U, E, v: np.column_stack([U, v]), C.RLog2b),
        }
        if PAPER_ONLY:
            v2_variants = {k: v for k, v in v2_variants.items() if k in ("chainuq-v3", "chainuq-v3-A")}
        for name, (head, tier, build, cls) in v2_variants.items():
            if head not in methods:
                continue
            for seed in methods[head]:
                pv, pt = load_preds(head, backbone, seed, ds, "validation"), load_preds(head, backbone, seed, ds, TEST)
                if pv is None or pt is None:
                    continue
                sv, st = cell.scores("validation", pv), cell.scores(TEST, pt)
                Xv = build(cell.u_matrix("validation", pv), cell.with_vhat(Ev, "validation", pv), jv)
                Xt = build(cell.u_matrix(TEST, pt), cell.with_vhat(Et, TEST, pt), jt)
                m = cls().fit(sv, Xv, yv)
                pr_v, pr_t = m.predict(sv, Xv), m.predict(st, Xt)
                v2_rows[name].append(eval_scores(yv, pr_v, yt, pr_t, pr_t))
                outs[name].append((pr_t, pr_t))
            if v2_rows[name]:
                if tier == "ablation":
                    R.setdefault("v2_ablation", {})[name] = agg(v2_rows[name])
                else:
                    R[tier][name] = agg(v2_rows[name])

        # ---- Judge agreement with the main judge (step labels) -------------------------------------
        main_t = {s.uid: s.step_labels for s in cell.test}
        agree = {}
        for judge in sorted(set(list(JUDGES) + ["nli"]) - {MAIN_JUDGE}):
            other = _judge_steps(judge, backbone, ds, TEST)
            if not other:
                continue
            a, b = [], []
            for u, lab in main_t.items():
                if u in other and other[u] and lab and len(other[u]) == len(lab):
                    a.extend(lab)
                    b.extend(other[u])
            if a:
                agree[judge] = {"step_agreement": float(np.mean(np.asarray(a) == np.asarray(b))),
                                "step_kappa": cohen_kappa(a, b), "n_steps": len(a)}
        R["judge_agreement"] = agree

        # ---- Label robustness: second response-label judge on test (judge-labelled datasets only) -
        if DATASETS[ds]["label"] == "judge":
            alt_path = judge_path(LABEL_JUDGE_2, "correct", backbone, ds, TEST)
            if alt_path.exists():
                alt = {r["uid"]: r["value"] for r in read_jsonl(alt_path) if r["value"] in (0, 1)}
                y2 = np.asarray([alt.get(s.uid, -1) for s in cell.test])
                has = y2 >= 0
                agree_mask = has & (y2 == yt)
                lr = {"n_with_second_label": int(has.sum()),
                      "label_agreement": float(np.mean(y2[has] == yt[has])) if has.any() else float("nan"),
                      "label_kappa": cohen_kappa(yt[has], y2[has]) if has.any() else float("nan"),
                      "agreement_subset_share": float(agree_mask.mean()), "methods": {}}
                for m, pairs in outs.items():
                    rows_agree, rows_alt = [], []
                    for st, cal in pairs:
                        ok = np.isfinite(st)
                        a, b = agree_mask & ok, has & ok
                        rows_agree.append({"auroc": M.auroc(yt[a], st[a]), **M.calibration(yt[a], cal[a])})
                        rows_alt.append({"auroc": M.auroc(y2[b], st[b]), **M.calibration(y2[b], cal[b])})
                    lr["methods"][m] = {"agree_subset": agg(rows_agree), "second_judge_labels": agg(rows_alt)}
                R["label_robustness"] = lr

        # ---- E5: reasoning-format robustness (head + R-Log fit on the step-format validation) -----
        if TEST == "test" and not PAPER_ONLY:  # format variants exist only for the original test split
            fmt_rows = {}
            for fmt in ("numbered", "bullet", "free"):
                gpath = ARTIFACTS / "gen" / backbone / ds / fmt / "test.jsonl"
                if not gpath.exists() or "chainuq" not in methods:
                    continue
                gens = read_jsonl(gpath)
                parse_rate = float(np.mean([g.get("parse_ok", False) for g in gens])) if gens else float("nan")
                try:
                    fcell = Cell(backbone, ds, fmt, hidden)
                except FileNotFoundError:
                    continue
                if not fcell.test:
                    continue
                yf = fcell.labels(TEST)
                rows = []
                for seed in methods["chainuq"]:
                    pv = load_preds("chainuq", backbone, seed, ds, "validation")
                    pf = load_preds("chainuq", backbone, seed, ds, TEST, fmt)
                    if pv is None or pf is None:
                        continue
                    sv, sf = cell.scores("validation", pv), fcell.scores(TEST, pf)
                    Uv, Uf = cell.u_matrix("validation", pv), fcell.u_matrix(TEST, pf)
                    rl = C.RLog().fit(sv, Uv, yv)
                    iso = scalar_calibrated("iso", yv, sv, sf)
                    rows.append({**{f"head_{k}": v for k, v in eval_scores(yv, sv, yf, sf, iso).items()},
                                 **{f"rlog_{k}": v for k, v in eval_scores(yv, rl.predict(sv, Uv), yf, rl.predict(sf, Uf), rl.predict(sf, Uf)).items()}})
                if rows:
                    fmt_rows[fmt] = {"parse_rate": parse_rate, **agg(rows)}
            R["formats"] = fmt_rows

        # ---- Paired bootstrap: ChainUQ-Head vs best Tier-A supervised baseline -------------------
        sup = {} if PAPER_ONLY else {m: R["tier_a"][m]["auroc"]["mean"] for m in TIER_A_SUP if m in R["tier_a"]}
        if "chainuq" in methods and sup:
            best = max(sup, key=sup.get)
            sa = np.stack([cell.scores(TEST, load_preds("chainuq", backbone, sd, ds, TEST)) for sd in methods["chainuq"]])
            sb = np.stack([cell.scores(TEST, load_preds(best, backbone, sd, ds, TEST)) for sd in methods[best]])
            R["bootstrap_vs_best_sup"] = {"baseline": best, **paired_bootstrap_auroc(yt, sa, sb)}
        res["datasets"][ds] = R
    return res


def _lr(Xv, Xt, yv):
    m = C.LogReg().fit(Xv, yv)
    return m.predict(Xv), m.predict(Xt)


def _rlog(sv, st, Uv, Ut, yv):
    m = C.RLog().fit(sv, Uv, yv)
    return m.predict(sv, Uv), m.predict(st, Ut)


def _judge_steps(judge, backbone, ds, split, task="steps"):
    if judge.startswith("self-") and judge != f"self-{backbone}":
        return {}
    path = judge_path(judge, task, backbone, ds, split)
    if not path.exists():
        return {}
    return {r["uid"]: r["value"] for r in read_jsonl(path) if isinstance(r["value"], list)}


def fmt_table(res):
    lines = [f"# {res['backbone']}\n"]
    for ds, R in res["datasets"].items():
        lines.append(f"## {ds} (val {R['n_val']}, test {R['n_test']}, pos rate {R['test_pos_rate']:.3f})\n")
        lines.append("| tier | method | Acc | AUROC | PRAUC | ECE | Brier | Risk@90 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for tier in ("tier_a", "tier_b"):
            for m, v in R[tier].items():
                g = lambda k: f"{v[k]['mean']:.3f}±{v[k]['std']:.3f}" if k in v else "-"
                lines.append(f"| {tier[-1].upper()} | {m} | {g('acc')} | {g('auroc')} | {g('prauc')} | {g('ece')} | {g('brier')} | {g('risk90')} |")
        for sec in ("calibration", "decomposition", "loo", "rlog_ablation", "judge_swap", "v2_ablation"):
            if R.get(sec):
                lines.append(f"\n**{sec}**\n\n| variant | AUROC | ECE | Brier | Risk@90 |\n|---|---|---|---|---|")
                for m, v in R[sec].items():
                    g = lambda k: f"{v[k]['mean']:.3f}" if k in v else "-"
                    lines.append(f"| {m} | {g('auroc')} | {g('ece')} | {g('brier')} | {g('risk90')} |")
        for sec in ("permutation", "strict_ood", "quadrants", "judge_agreement", "label_robustness", "formats",
                    "bootstrap_vs_best_sup"):
            if R.get(sec):
                lines.append(f"\n**{sec}**: `{json.dumps(R[sec], default=float)[:1500]}`")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--n_perm", type=int, default=20)
    p.add_argument("--paper_only", action="store_true", help="skip analyses the paper takes from paper_tables.py")
    p.add_argument("--test_split", default="test")
    args = p.parse_args()
    global TEST, PAPER_ONLY
    TEST = args.test_split
    PAPER_ONLY = args.paper_only
    res = evaluate_backbone(args.backbone, args.n_perm)
    out = results_dir() / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if TEST == "test" else f"_{TEST}"
    (out / f"eval{suffix}.json").write_text(json.dumps(res, indent=2, default=float))
    (out / f"tables{suffix}.md").write_text(fmt_table(res))
    print(f"wrote {out / f'tables{suffix}.md'}")


if __name__ == "__main__":
    main()
