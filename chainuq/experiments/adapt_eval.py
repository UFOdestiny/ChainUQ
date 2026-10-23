"""Score label-free target adaptation: the source head vs. the adapted head (``experiments.adapt``).

Per (backbone, OOD dataset, seed) and tier (B = ChainUQ, A = judge-free ChainUQ-A), R-Log is fit on validation
for the source head and for the adapted head and applied to the held-out splits. ``selected`` uses the adapted
head only when it gives the lower out-of-fold log-loss of R-Log (C=0.05, Platt map, 5 folds) on the labeled
validation split that R-Log is fit on anyway, so no test information enters the choice.
Writes ``results/paper/<backbone>/<dataset>_adapt[<tag>].json``.

Example:
    python -m chainuq.experiments.adapt_eval --backbone llama --dataset babi
    python -m chainuq.experiments.adapt_eval --backbone llama --dataset babi --adapted chainuq-ta-src --tag _src --splits test
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

from chainuq import calibration as C
from chainuq.artifacts import feat_dir, feature_info, results_dir
from chainuq.evaluation import Cell, HeadEvidence, agg, eval_scores, load_preds
from chainuq.registry import FINAL_HEAD


def oof_nll(s, X, y):
    p = C.out_of_fold(lambda tr, te: C.RLogCore(C=0.05, final="platt", cross_fit=True).fit(s[tr], X[tr], y[tr])
                      .predict(s[te], X[te]), len(y), y)
    return C.nll(y, p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True)
    p.add_argument("--dataset", required=True)
    p.add_argument("--seeds", default="2026,2027,2028,2029,2030")
    p.add_argument("--splits", default="test,test2")
    p.add_argument("--adapted", default="chainuq-ta", help="predictions key of the adapted head")
    p.add_argument("--tag", default="", help="suffix of the output file, e.g. _src")
    args = p.parse_args()
    bb, ds = args.backbone, args.dataset
    hidden = feature_info(bb)["hidden_size"]
    cells = {sp: Cell(bb, ds, sp, hidden) for sp in args.splits.split(",") if (feat_dir(bb, ds, sp) / "DONE.json").exists()}
    yv = next(iter(cells.values())).labels("validation")
    rows, picks, n_seeds = defaultdict(lambda: defaultdict(list)), defaultdict(int), 0
    for seed in [int(s) for s in args.seeds.split(",")]:
        res = {}
        for head in (FINAL_HEAD, args.adapted):
            pv = load_preds(head, bb, seed, ds, "validation")
            if pv is None:
                break
            for sp, cell in cells.items():
                ev = HeadEvidence(cell, pv, load_preds(head, bb, seed, ds, sp))
                for tier in ("B", "A"):
                    Xv, Xt = ev.matrices(tier)
                    if (head, tier) not in res:
                        m = C.RLog().fit(ev.sv, Xv, yv)
                        res[head, tier] = {"model": m, "nll": oof_nll(ev.sv, Xv, yv), "p_val": m.predict(ev.sv, Xv)}
                    r = res[head, tier]
                    q = r["model"].predict(ev.st, Xt)
                    r[sp] = eval_scores(yv, r["p_val"], cell.labels(sp), q, q)
        if len(res) < 4:
            continue
        n_seeds += 1
        for tier in ("B", "A"):
            b, a = res[FINAL_HEAD, tier], res[args.adapted, tier]
            use = a["nll"] < b["nll"]
            picks[tier] += int(use)
            for sp in cells:
                rows[f"{tier}_{sp}"]["base"].append(b[sp])
                rows[f"{tier}_{sp}"]["adapted"].append(a[sp])
                rows[f"{tier}_{sp}"]["selected"].append(a[sp] if use else b[sp])
        print(json.dumps({"seed": seed, **{f"{h}_{t}": round(v["test"]["auroc"], 4) for (h, t), v in res.items() if "test" in v},
                          "pick_A": bool(res[args.adapted, "A"]["nll"] < res[FINAL_HEAD, "A"]["nll"])}), flush=True)
    out = {k: {kk: agg(vv) for kk, vv in d.items()} for k, d in rows.items()}
    out["picks"], out["n_seeds"] = dict(picks), n_seeds
    path = results_dir() / "paper" / bb / f"{ds}_adapt{args.tag}.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
