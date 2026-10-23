#!/usr/bin/env python3
"""Final verdict on TEST: does ChainUQ v2 beat the strongest baseline per (backbone, dataset, metric)?

Rules (fixed before looking at test results):
  * ChainUQ v2-A (tier A, no judge at inference) is compared with every tier-A baseline.
  * ChainUQ v2   (tier B, judge evidence)       is compared with every baseline of both tiers.
  * Ablations and the original ChainUQ (v1) are not competitors; v1 is reported for reference.
  * A cell is a win when our mean over seeds is at least as good as the best baseline mean.

Example:
    python pipeline/sota_report.py > results/sota_report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq.registry import BACKBONES, results_dir

METRICS = ["acc", "auroc", "prauc", "ece", "brier", "risk90"]
LOWER = {"ece", "brier", "risk90"}
VERSION = sys.argv[1] if len(sys.argv) > 1 else "v2"
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "test"
OURS = {"A": f"chainuq-{VERSION}-A", "B": f"chainuq-{VERSION}"}
NOT_COMPETITORS = {"chainuq", "chainuq-lb", "chainuq-cs", "chainuq-cf", "chainuq-fr", "chainuq2bd",
                   "chainuq-v2-A", "chainuq-v2", "chainuq-v3-A", "chainuq-v3", "chainuq+rlog"}


def better(m, a, b):
    return a <= b if m in LOWER else a >= b


def main():
    lines = [f"# ChainUQ {VERSION} vs strongest baseline ({SPLIT}, mean over seeds)\n"]
    totals = {"A": [0, 0], "B": [0, 0]}
    per_metric = {t: {m: [0, 0] for m in METRICS} for t in "AB"}
    for bb in BACKBONES:
        path = results_dir() / bb / ("eval.json" if SPLIT == "test" else f"eval_{SPLIT}.json")
        if not path.exists():
            lines.append(f"\n**{bb}: eval.json missing**\n")
            continue
        res = json.loads(path.read_text())
        for ds, R in res["datasets"].items():
            a_base = {k: v for k, v in R["tier_a"].items() if k not in NOT_COMPETITORS}
            b_base = {**a_base, **{k: v for k, v in R["tier_b"].items() if k not in NOT_COMPETITORS}}
            lines.append(f"\n## {bb} / {ds} (n_test={R['n_test']})\n")
            lines.append("| tier | metric | ours | best baseline | value | win |")
            lines.append("|---|---|---|---|---|---|")
            for tier, pool, table in (("A", a_base, R["tier_a"]), ("B", b_base, R["tier_b"])):
                ours = table.get(OURS[tier])
                if ours is None:
                    continue
                for m in METRICS:
                    cand = {k: v[m]["mean"] for k, v in pool.items() if m in v}
                    if not cand or m not in ours:
                        continue
                    best = min(cand, key=cand.get) if m in LOWER else max(cand, key=cand.get)
                    o = ours[m]["mean"]
                    win = better(m, o, cand[best])
                    totals[tier][0] += win
                    totals[tier][1] += 1
                    per_metric[tier][m][0] += win
                    per_metric[tier][m][1] += 1
                    lines.append(f"| {tier} | {m} | {o:.3f}±{ours[m]['std']:.3f} | {best} | {cand[best]:.3f} | "
                                 f"{'**win**' if win else 'loss'} |")
            v1 = R["tier_a"].get("chainuq")
            if v1:
                lines.append(f"\noriginal ChainUQ head (v1): auroc {v1['auroc']['mean']:.3f}, ece {v1['ece']['mean']:.3f}")
    summary = ["\n# Summary\n"]
    for tier in "AB":
        w, n = totals[tier]
        summary.append(f"- **Tier {tier}** ({OURS[tier]}): wins {w}/{n} cells ({100 * w / max(n, 1):.0f}%)")
        summary.append("  - per metric: " + ", ".join(f"{m} {a}/{b}" for m, (a, b) in per_metric[tier].items()))
    print("\n".join(summary + lines))


if __name__ == "__main__":
    main()
