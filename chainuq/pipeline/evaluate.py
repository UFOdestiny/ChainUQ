"""Stage 5: main comparison on the held-out split, per backbone.

For every dataset: tier-A baselines (no external model at inference; calibration metrics after isotonic
regression fit on validation), tier-B baselines (judge or reward model), ChainUQ (head + R-Log on the evidence
and one judge call) and ChainUQ-A (judge-free evidence), each as mean/std over the head seeds; plus every method
re-scored against a second correctness labeler.

Outputs ``results/<backbone>/eval.json`` (``eval_<split>.json`` for other splits) and a readable ``tables.md``.

Example:
    python -m chainuq.pipeline.evaluate --backbone llama            # test
    python -m chainuq.pipeline.evaluate --backbone llama --split test2
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from chainuq import metrics as M
from chainuq.artifacts import feature_info, gen_path, judge_path, read_jsonl, results_dir
from chainuq.evaluation import (
    Cell,
    agg,
    available_methods,
    cohen_kappa,
    eval_scores,
    final_calibrator,
    final_evidence,
    fit_eval,
    load_preds,
    scalar_calibrated,
)
from chainuq.registry import (
    BACKBONES,
    DATASETS,
    FINAL_HEAD,
    OURS,
    OURS_A,
    SECOND_LABEL_JUDGE,
    TIER_A_BASELINES,
    TIER_B_BASELINES,
)


def baseline_rows(cell, methods, backbone, outs):
    """Raw-score discrimination and isotonic-calibrated calibration metrics of every baseline."""
    yv, yt, sp = cell.labels("validation"), cell.labels(cell.split), cell.split
    rows = {"tier_a": {}, "tier_b": {}}
    for tier, pool in (("tier_a", TIER_A_BASELINES), ("tier_b", TIER_B_BASELINES)):
        for m in pool:
            res = []
            for seed in methods.get(m, []):
                pv, pt = load_preds(m, backbone, seed, cell.dataset, "validation"), load_preds(m, backbone, seed, cell.dataset, sp)
                if pv is None or pt is None:
                    continue
                sv, st = cell.scores("validation", pv), cell.scores(sp, pt)
                cal = scalar_calibrated("iso", yv, sv, st)
                res.append(eval_scores(yv, sv, yt, st, cal))
                outs[m].append((st, cal))
            if res:
                rows[tier][m] = agg(res)
    return rows


def chainuq_rows(cell, methods, backbone, outs):
    """ChainUQ (tier B) and ChainUQ-A (tier A): R-Log on the final head of every seed."""
    yv, yt = cell.labels("validation"), cell.labels(cell.split)
    res = defaultdict(list)
    for seed in methods.get(FINAL_HEAD, []):
        ev = final_evidence(cell, backbone, seed)
        if ev is None:
            continue
        for key, tier in ((OURS, "B"), (OURS_A, "A")):
            Xv, Xt = ev.matrices(tier)
            r, _, p_te = fit_eval(final_calibrator(ev, tier), yv, ev.sv, Xv, yt, ev.st, Xt)
            res[key].append(r)
            outs[key].append((p_te, p_te))
    return {k: agg(v) for k, v in res.items()}


def label_robustness(cell, backbone, outs):
    """Agreement with a second correctness labeler, and every method re-scored on (i) responses where both
    labelers agree and (ii) the second labeler's labels."""
    path = judge_path(SECOND_LABEL_JUDGE, "correct", backbone, cell.dataset, cell.split)
    if DATASETS[cell.dataset]["label"] != "judge" or not path.exists():
        return None
    yt = cell.labels(cell.split)
    alt = {r["uid"]: r["value"] for r in read_jsonl(path) if r["value"] in (0, 1)}
    y2 = np.asarray([alt.get(s.uid, -1) for s in cell.test])
    has = y2 >= 0
    agree = has & (y2 == yt)
    res = {
        "n_with_second_label": int(has.sum()),
        "label_agreement": float(np.mean(y2[has] == yt[has])) if has.any() else float("nan"),
        "label_kappa": cohen_kappa(yt[has], y2[has]) if has.any() else float("nan"),
        "agreement_subset_share": float(agree.mean()),
        "methods": {},
    }
    for m, pairs in outs.items():
        on_agree, on_alt = [], []
        for st, cal in pairs:
            ok = np.isfinite(st)
            a, b = agree & ok, has & ok
            on_agree.append({"auroc": M.auroc(yt[a], st[a]), **M.calibration(yt[a], cal[a])})
            on_alt.append({"auroc": M.auroc(y2[b], st[b]), **M.calibration(y2[b], cal[b])})
        res["methods"][m] = {"agree_subset": agg(on_agree), "second_judge_labels": agg(on_alt)}
    return res


def evaluate_backbone(backbone, split):
    methods = available_methods(backbone)
    hidden = feature_info(backbone)["hidden_size"]
    res = {"backbone": backbone, "split": split, "final_head": FINAL_HEAD, "methods": methods, "datasets": {}}
    for ds in DATASETS:
        if not gen_path(backbone, ds, split).exists():  # StrategyQA has no test2
            continue
        cell = Cell(backbone, ds, split, hidden)
        if not cell.test or not cell.val:
            continue
        yt = cell.labels(split)
        outs = defaultdict(list)  # method -> [(raw score, calibrated probability)] per seed
        R = {"n_val": len(cell.val), "n_test": len(yt), "test_pos_rate": float(yt.mean()), **baseline_rows(cell, methods, backbone, outs)}
        ours = chainuq_rows(cell, methods, backbone, outs)
        if OURS in ours:
            R["tier_b"][OURS], R["tier_a"][OURS_A] = ours[OURS], ours[OURS_A]
        lr = label_robustness(cell, backbone, outs)
        if lr:
            R["label_robustness"] = lr
        res["datasets"][ds] = R
        print(f"[evaluate] {backbone} {ds} {split}: {len(yt)} responses", flush=True)
    return res


def tables_md(res):
    lines = [f"# {res['backbone']} ({res.get('split', 'test')})\n"]
    for ds, R in res["datasets"].items():
        lines.append(f"## {ds} (val {R['n_val']}, test {R['n_test']}, positive rate {R['test_pos_rate']:.3f})\n")
        lines.append("| tier | method | Acc | AUROC | PRAUC | ECE | Brier | Risk@90 |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for tier in ("tier_a", "tier_b"):
            for m, v in R[tier].items():
                g = lambda k: f"{v[k]['mean']:.3f}±{v[k]['std']:.3f}" if k in v else "-"
                lines.append(f"| {tier[-1].upper()} | {m} | {g('acc')} | {g('auroc')} | {g('prauc')} | {g('ece')} | {g('brier')} | {g('risk90')} |")
        lines.append("")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(BACKBONES))
    p.add_argument("--split", default="test", help="held-out split: test or test2")
    args = p.parse_args()
    res = evaluate_backbone(args.backbone, args.split)
    out = results_dir() / args.backbone
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.split == "test" else f"_{args.split}"
    (out / f"eval{suffix}.json").write_text(json.dumps(res, indent=2, default=float))
    (out / f"tables{suffix}.md").write_text(tables_md(res))
    print(f"wrote {out / f'eval{suffix}.json'}")


if __name__ == "__main__":
    main()
