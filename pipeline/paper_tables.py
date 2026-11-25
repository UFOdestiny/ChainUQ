#!/usr/bin/env python3
"""Numbers for the paper that ``eval.json`` does not hold, all computed with the final method.

Final method: ChainUQ head (``chainuq2bd``) + R-Log (``RLog2c``); full ChainUQ uses the evidence
[u(r); chain verdict; conclusion/weakest-step token features], ChainUQ-A drops every judge input.
Every calibrator is fit on the validation split and applied once to the evaluated split.

Per (backbone, dataset), mean/std over the 10 head seeds:
  calibrators     PS / TS / ISO on s, BWH and R-Log on the full evidence     (RQ2, RQ3 figure)
  stage2          R-Log with one evidence block removed, or R-Log at fixed capacity (RQ4)
  heads           full ChainUQ / ChainUQ-A with each head ablation (RQ4; heads trained for llama)
  judge_swap      ChainUQ without the chain verdict, step labels from another verifier (appendix)
  choice          how often each R-Log capacity was selected on validation (appendix)
and, for llama seed 2026, per-sample s / calibrated confidence for the case studies.

Example:
    python pipeline/paper_tables.py --backbone llama --dataset hotpotqa
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq import calibration as C
from cuq import ufeatures as UF
from cuq.registry import FINAL_HEAD, SOURCE_DATASET, feat_dir, results_dir

_spec = importlib.util.spec_from_file_location("evaluate", Path(__file__).resolve().parent / "evaluate.py")
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

HEAD = FINAL_HEAD
HEAD_ABLATIONS = [h for h in ["chainuq2bd-lb", "chainuq2bd-cs", "chainuq2bd-cf", "chainuq2bd-fr", "chainuq2bd-tok",
                               "chainuq2bd-lay", "chainuq2", "chainuq2bd", "chainuq2bd-core-tok", "chainuq2-core",
                               "chainuq2bd-core", "chainuq2pcs-steps", "chainuq2pcs-verdict",
                               "uhead", "saplma"]  # last two: head replaced by a baseline probe, same R-Log
                  if h != HEAD]
STEP_VERIFIERS = ["gptoss", "qwen1b", "self", "nli", "gptoss-noconcl"]


def fit_eval(cls, yv, sv, Xv, yt, st, Xt):
    m = cls().fit(sv, Xv, yv)
    pv, pt = m.predict(sv, Xv), m.predict(st, Xt)
    return E.eval_scores(yv, pv, yt, pt, pt), m, pt


def run(backbone, ds, split, sections=("main", "heads"), heads=None, head_seeds=0):
    E.TEST = split
    hidden = json.loads((feat_dir(backbone, SOURCE_DATASET, "train") / "DONE.json").read_text())["hidden_size"]
    methods = E.available_methods(backbone)
    cell = E.Cell(backbone, ds, "step", hidden)
    yv, yt = cell.labels("validation"), cell.labels(split)
    Ev0, Et0 = cell.extra("validation"), cell.extra(split)
    jv, jt = cell.verdict("validation"), cell.verdict(split)
    J = E.cols(UF.GROUPS["u_judge"])
    notJ = [i for i in range(len(UF.FEATURE_NAMES)) if i not in J]
    full = lambda U, v, X: np.column_stack([U, v, X])
    tier_a = lambda U, v, X: np.column_stack([U[:, notJ], X])

    rows = defaultdict(lambda: defaultdict(list))
    choice = Counter()
    cases = []
    for seed in (methods[HEAD] if "main" in sections else []):
        pv, pt = E.load_preds(HEAD, backbone, seed, ds, "validation"), E.load_preds(HEAD, backbone, seed, ds, split)
        sv, st = cell.scores("validation", pv), cell.scores(split, pt)
        Uv, Ut = cell.u_matrix("validation", pv), cell.u_matrix(split, pt)
        Ev, Et = cell.with_vhat(Ev0, "validation", pv), cell.with_vhat(Et0, split, pt)

        # Calibrators on the same head score (RQ2 table, RQ3 figure).
        for name in ("ps", "ts", "iso"):
            cal = E.scalar_calibrated(name, yv, sv, st)
            rows["calibrators"][name].append(E.eval_scores(yv, sv, yt, st, cal))
        r, _, _ = fit_eval(C.BWH, yv, sv, full(Uv, jv, Ev), yt, st, full(Ut, jt, Et))
        rows["calibrators"]["bwh"].append(r)
        r, m, main_t = fit_eval(C.RLog2c, yv, sv, full(Uv, jv, Ev), yt, st, full(Ut, jt, Et))
        rows["calibrators"]["rlog"].append(r)
        choice[m.choice] += 1
        r, _, _ = fit_eval(C.RLog2c, yv, sv, tier_a(Uv, jv, Ev), yt, st, tier_a(Ut, jt, Et))
        rows["calibrators"]["rlog_A"].append(r)

        # Stage-2 ablations (full ChainUQ with one evidence block removed).
        blocks = {f"-{g}": [i for i in range(Uv.shape[1]) if i not in E.cols(n)] for g, n in UF.GROUPS.items()}
        for name, keep in blocks.items():
            r, _, _ = fit_eval(C.RLog2c, yv, sv, full(Uv[:, keep], jv, Ev), yt, st, full(Ut[:, keep], jt, Et))
            rows["stage2"][name].append(r)
        r, _, _ = fit_eval(C.RLog2c, yv, sv, np.column_stack([Uv, Ev]), yt, st, np.column_stack([Ut, Et]))
        rows["stage2"]["-verdict"].append(r)
        r, _, _ = fit_eval(C.RLog2c, yv, sv, np.column_stack([Uv, jv]), yt, st, np.column_stack([Ut, jt]))
        rows["stage2"]["-token_extra"].append(r)
        r, _, _ = fit_eval(C.RLog, yv, sv, full(Uv, jv, Ev), yt, st, full(Ut, jt, Et))
        rows["stage2"]["fixed_capacity"].append(r)

        # Step-label source swap, without the chain verdict so the step labels carry the evidence.
        for name in STEP_VERIFIERS:
            judge, task = ("gptoss", "steps_noconcl") if name == "gptoss-noconcl" else \
                (f"self-{backbone}" if name == "self" else name, "steps")
            lv, lt = E._judge_steps(judge, backbone, ds, "validation", task), E._judge_steps(judge, backbone, ds, split, task)
            if not lv or not lt:
                continue
            Uv2, Ut2 = cell.u_matrix("validation", pv, lv), cell.u_matrix(split, pt, lt)
            r, _, _ = fit_eval(C.RLog2c, yv, sv, np.column_stack([Uv2, Ev]), yt, st, np.column_stack([Ut2, Et]))
            rows["judge_swap"][name].append(r)

        if backbone == "llama" and seed == methods[HEAD][0]:
            for x, s_, c_, v_ in zip(cell.test, st, main_t, jt):
                cases.append({"uid": x.uid, "y": int(x.y), "s": float(1 / (1 + np.exp(-s_))), "conf": float(c_),
                              "verdict": None if not np.isfinite(v_) else float(v_), "step_labels": x.step_labels})

    # Head ablations inside the full pipeline (heads trained for llama only).
    for head in ((heads or HEAD_ABLATIONS) if "heads" in sections else []):
        for seed in (methods.get(head, [])[:head_seeds] if head_seeds else methods.get(head, [])):
            pv, pt = E.load_preds(head, backbone, seed, ds, "validation"), E.load_preds(head, backbone, seed, ds, split)
            if pv is None or pt is None:
                continue
            sv, st = cell.scores("validation", pv), cell.scores(split, pt)
            Uv, Ut = cell.u_matrix("validation", pv), cell.u_matrix(split, pt)
            # The steps-only PCS ablation has no trained verdict output: keep its evidence judge-free of it.
            Ev, Et = (Ev0, Et0) if head == "chainuq2pcs-steps" else \
                (cell.with_vhat(Ev0, "validation", pv), cell.with_vhat(Et0, split, pt))
            rows["heads_raw"][head].append(E.eval_scores(yv, sv, yt, st, E.scalar_calibrated("iso", yv, sv, st)))
            r, _, _ = fit_eval(C.RLog2c, yv, sv, full(Uv, jv, Ev), yt, st, full(Ut, jt, Et))
            rows["heads_B"][head].append(r)
            r, _, _ = fit_eval(C.RLog2c, yv, sv, tier_a(Uv, jv, Ev), yt, st, tier_a(Ut, jt, Et))
            rows["heads_A"][head].append(r)

    out = {sec: {k: E.agg(v) for k, v in d.items()} for sec, d in rows.items()}
    out["choice"] = dict(choice)
    out["n_val"], out["n_test"] = len(yv), len(yt)
    return out, cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--sections", default="main,heads", help="main and/or heads; results merge into the existing file")
    p.add_argument("--heads", default="", help="comma-separated subset of the head ablations (default: all)")
    p.add_argument("--head_seeds", type=int, default=0, help="use only the first N seeds for the head ablations")
    args = p.parse_args()
    out_dir = results_dir() / "paper" / args.backbone
    out_dir.mkdir(parents=True, exist_ok=True)
    res, cases = run(args.backbone, args.dataset, args.split, tuple(args.sections.split(",")),
                     [h for h in args.heads.split(",") if h] or None, args.head_seeds)
    path = out_dir / f"{args.dataset}_{args.split}.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    for sec, val in res.items():
        if isinstance(val, dict) and isinstance(old.get(sec), dict):
            old[sec].update(val)
        elif val or sec not in old:
            old[sec] = val
    path.write_text(json.dumps(old, indent=1, default=float))
    if cases:
        (out_dir / f"{args.dataset}_{args.split}_cases.json").write_text(json.dumps(cases))
    print(f"wrote {out_dir / f'{args.dataset}_{args.split}.json'}")


if __name__ == "__main__":
    main()
