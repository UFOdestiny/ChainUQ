#!/usr/bin/env python3
"""Build the test-split eval.json of the final method without recomputing it.

Baseline rows, judge agreement, and label robustness do not depend on the ChainUQ head and are taken
from an earlier full evaluation (``eval_refinehead.json``). The final ChainUQ / ChainUQ-A rows are the
``calibrators.rlog`` / ``calibrators.rlog_A`` entries of pipeline/paper_tables.py, which run the same
R-Log on the same inputs and seeds as pipeline/evaluate.py.

Example:
    python pipeline/assemble_eval.py --source eval_refinehead.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq.registry import BACKBONES, FINAL_HEAD, results_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="eval_refinehead.json")
    args = p.parse_args()
    for bb in BACKBONES:
        res = json.loads((results_dir() / bb / args.source).read_text())
        for ds, R in res["datasets"].items():
            P = json.loads((results_dir() / "paper" / bb / f"{ds}_test.json").read_text())
            R["tier_b"]["chainuq-v3"] = P["calibrators"]["rlog"]
            R["tier_a"]["chainuq-v3-A"] = P["calibrators"]["rlog_A"]
            for stale in ("chainuq-v2", "chainuq-v2-A"):
                R["tier_a"].pop(stale, None)
                R["tier_b"].pop(stale, None)
        res["final_head"] = FINAL_HEAD
        res["assembled_from"] = args.source
        out = results_dir() / bb / "eval.json"
        if out.exists():
            out.rename(out.with_name("eval_before_assemble.json"))
        out.write_text(json.dumps(res, indent=2, default=float))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
