#!/usr/bin/env python3
"""E8: aggregate measured compute from the stage stats files (no new GPU work).

Reports, per backbone and stage, total seconds and tokens plus per-sample
averages. The judge rows are what the paper's "external 24B judge cost"
discussion needs: Stage-2 step verification and Judge-Final are the
inference-time overhead of Tier-B methods; Stage-1 labelling is offline only.

Example:
    python pipeline/costs.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cuq.registry import ARTIFACTS, results_dir


def main():
    rows = defaultdict(lambda: defaultdict(float))
    for f in (ARTIFACTS / "gen").rglob("*.stats.json"):
        d = json.loads(f.read_text())
        key = (d["backbone"], f"gen[{d['fmt']}{'/' + d['tag'] if d['tag'] else ''}]")
        rows[key]["seconds"] += d["gen_seconds"]
        rows[key]["samples"] += d["n_generated"]
        rows[key]["gen_tokens"] += d["gen_tokens"]
        rows[key]["prompt_tokens"] += d["prompt_tokens"]
    for f in (ARTIFACTS / "judge").rglob("*.stats.json"):
        d = json.loads(f.read_text())
        key = (d["backbone"], f"judge[{d['judge']}/{d['task']}]")
        rows[key]["seconds"] += d["seconds"]
        rows[key]["samples"] += d["n"]
        rows[key]["gen_tokens"] += d["completion_tokens"]
        rows[key]["prompt_tokens"] += d["prompt_tokens"]
    for f in (ARTIFACTS / "feat").rglob("DONE.json"):
        d = json.loads(f.read_text())
        key = (d["backbone"], f"features[{d['fmt']}]")
        rows[key]["seconds"] += d["seconds"]
        rows[key]["samples"] += d["n_new"]
        rows[key]["prompt_tokens"] += d["tokens_forwarded"]
    for f in (ARTIFACTS / "ckpt").rglob("meta.json"):
        d = json.loads(f.read_text())
        parts = f.relative_to(ARTIFACTS / "ckpt").parts
        key = (parts[1], f"train[{parts[0]}]")
        rows[key]["seconds"] += float(d.get("train_seconds", 0.0))
        rows[key]["runs"] += 1
        rows[key]["n_params"] = float(d.get("n_params", 0))

    out = []
    for (bb, stage), v in sorted(rows.items()):
        rec = {"backbone": bb, "stage": stage, **{k: round(x, 2) for k, x in v.items()}}
        if v.get("samples"):
            rec["sec_per_sample"] = round(v["seconds"] / v["samples"], 5)
            rec["tokens_per_sample"] = round((v.get("gen_tokens", 0) + v.get("prompt_tokens", 0)) / v["samples"], 1)
        out.append(rec)
    results_dir().mkdir(parents=True, exist_ok=True)
    (results_dir() / "costs.json").write_text(json.dumps(out, indent=2))
    for r in out:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
