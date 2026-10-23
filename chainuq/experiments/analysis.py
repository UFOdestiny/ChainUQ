"""Per-cell analyses of the paper beyond the main comparison (``pipeline.evaluate``).

For one (backbone, dataset), mean/std over the head seeds:
  calibrators  PS / TS / ISO on the head score, BWH and R-Log on the full evidence, R-Log on judge-free evidence
  stage2       ChainUQ with one evidence block removed, and with the single-capacity R-Log
  judge_swap   ChainUQ without the chain verdict, with step labels from another verifier
  choice       how often each R-Log capacity was selected on validation
  heads_*      ChainUQ and ChainUQ-A with the head replaced by an ablation or a baseline probe (same R-Log)
and, for Llama seed 2026, per-response head scores and calibrated confidences for the case studies.
Results merge into ``results/paper/<backbone>/<dataset>_<split>.json``.

Example:
    python -m chainuq.experiments.analysis --backbone llama --dataset hotpotqa
    python -m chainuq.experiments.analysis --backbone qwen8b --dataset musique --sections heads --head_seeds 5
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

import numpy as np

from chainuq import calibration as C
from chainuq import evidence as EV
from chainuq.artifacts import feature_info, results_dir
from chainuq.evaluation import Cell, HeadEvidence, agg, available_methods, eval_scores, fit_eval, judge_step_labels, load_preds, scalar_calibrated
from chainuq.heads import predicts_verdict
from chainuq.registry import FINAL_HEAD, MAIN_JUDGE

HEAD_ABLATIONS = ["chainuq-wo-step", "chainuq-wo-verdict", "chainuq-wo-pcs", "chainuq-wo-pcs-layer",
                  "chainuq-wo-pcs-token", "chainuq-wo-pcs-bdrop", "uhead", "uhead-pcs", "saplma"]
STEP_VERIFIERS = {"gptoss": (MAIN_JUDGE, "steps"), "gptoss-noconcl": (MAIN_JUDGE, "steps_noconcl"),
                  "qwen1b": ("qwen1b", "steps"), "self": ("self-{backbone}", "steps"), "nli": ("nli", "steps")}


def main_section(cell, methods, backbone, rows, choice, cases):
    yv, yt, sp, ds = cell.labels("validation"), cell.labels(cell.split), cell.split, cell.dataset
    for seed in methods[FINAL_HEAD]:
        pv, pt = load_preds(FINAL_HEAD, backbone, seed, ds, "validation"), load_preds(FINAL_HEAD, backbone, seed, ds, sp)
        ev = HeadEvidence(cell, pv, pt)
        (Xv, Xt), (Av, At) = ev.matrices("B"), ev.matrices("A")

        for name in ("ps", "ts", "iso"):
            rows["calibrators"][name].append(eval_scores(yv, ev.sv, yt, ev.st, scalar_calibrated(name, yv, ev.sv, ev.st)))
        rows["calibrators"]["bwh"].append(fit_eval(C.BWH, yv, ev.sv, Xv, yt, ev.st, Xt)[0])
        r, m, conf = fit_eval(C.RLog, yv, ev.sv, Xv, yt, ev.st, Xt)
        rows["calibrators"]["rlog"].append(r)
        choice[m.choice] += 1
        rows["calibrators"]["rlog_A"].append(fit_eval(C.RLog, yv, ev.sv, Av, yt, ev.st, At)[0])

        # Stage-2 ablations: one evidence block removed, or the single-capacity R-Log.
        full = lambda U, v, E: np.column_stack([U, v, E])
        for g, names in EV.GROUPS.items():
            keep = [i for i in range(ev.Uv.shape[1]) if i not in EV.columns(names)]
            rows["stage2"][f"-{g}"].append(fit_eval(C.RLog, yv, ev.sv, full(ev.Uv[:, keep], ev.jv, ev.Ev), yt, ev.st,
                                                    full(ev.Ut[:, keep], ev.jt, ev.Et))[0])
        rows["stage2"]["-verdict"].append(fit_eval(C.RLog, yv, ev.sv, np.column_stack([ev.Uv, ev.Ev]), yt, ev.st,
                                                   np.column_stack([ev.Ut, ev.Et]))[0])
        rows["stage2"]["-token_extra"].append(fit_eval(C.RLog, yv, ev.sv, np.column_stack([ev.Uv, ev.jv]), yt, ev.st,
                                                       np.column_stack([ev.Ut, ev.jt]))[0])
        rows["stage2"]["fixed_capacity"].append(fit_eval(C.RLogCore, yv, ev.sv, Xv, yt, ev.st, Xt)[0])

        # Step labels from another verifier, without the chain verdict so the step labels carry the evidence.
        for name, (judge, task) in STEP_VERIFIERS.items():
            judge = judge.format(backbone=backbone)
            lv, lt = judge_step_labels(judge, backbone, ds, "validation", task), judge_step_labels(judge, backbone, ds, sp, task)
            if lv and lt:
                Uv2, Ut2 = cell.u_matrix("validation", pv, lv), cell.u_matrix(sp, pt, lt)
                rows["judge_swap"][name].append(fit_eval(C.RLog, yv, ev.sv, np.column_stack([Uv2, ev.Ev]), yt, ev.st,
                                                         np.column_stack([Ut2, ev.Et]))[0])

        if backbone == "llama" and seed == methods[FINAL_HEAD][0]:
            for x, s_, c_, v_ in zip(cell.test, ev.st, conf, ev.jt):
                cases.append({"uid": x.uid, "y": int(x.y), "s": float(1 / (1 + np.exp(-s_))), "conf": float(c_),
                              "verdict": None if not np.isfinite(v_) else float(v_), "step_labels": x.step_labels})


def heads_section(cell, methods, backbone, rows, heads, head_seeds):
    yv, yt, sp, ds = cell.labels("validation"), cell.labels(cell.split), cell.split, cell.dataset
    for head in heads:
        for seed in methods.get(head, [])[: head_seeds or None]:
            pv, pt = load_preds(head, backbone, seed, ds, "validation"), load_preds(head, backbone, seed, ds, sp)
            if pv is None or pt is None:
                continue
            ev = HeadEvidence(cell, pv, pt, use_predicted_verdict=predicts_verdict(head))
            rows["heads_raw"][head].append(eval_scores(yv, ev.sv, yt, ev.st, scalar_calibrated("iso", yv, ev.sv, ev.st)))
            for tier in ("B", "A"):
                Xv, Xt = ev.matrices(tier)
                rows[f"heads_{tier}"][head].append(fit_eval(C.RLog, yv, ev.sv, Xv, yt, ev.st, Xt)[0])


def run(backbone, ds, split, sections=("main", "heads"), heads=None, head_seeds=0):
    methods = available_methods(backbone)
    cell = Cell(backbone, ds, split, feature_info(backbone)["hidden_size"])
    rows, choice, cases = defaultdict(lambda: defaultdict(list)), Counter(), []
    if "main" in sections:
        main_section(cell, methods, backbone, rows, choice, cases)
    if "heads" in sections:
        heads_section(cell, methods, backbone, rows, heads or HEAD_ABLATIONS, head_seeds)
    out = {sec: {k: agg(v) for k, v in d.items()} for sec, d in rows.items()}
    out["choice"] = dict(choice)
    out["n_val"], out["n_test"] = len(cell.val), len(cell.test)
    return out, cases


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--sections", default="main,heads", help="main and/or heads; results merge into the existing file")
    p.add_argument("--heads", default="", help="comma-separated heads for the heads section (default: HEAD_ABLATIONS)")
    p.add_argument("--head_seeds", type=int, default=0, help="use only the first N seeds in the heads section")
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
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
