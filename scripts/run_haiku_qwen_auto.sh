#!/usr/bin/env bash
# After the Qwen3 Haiku targets exist: train Haiku-imitating probes (one seed per GPU, four GPUs), then the judge-free
# evaluation of Haiku vs gpt-oss imitation vs no imitation; the summary goes to $LOG/qwen_summary.txt.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source <(grep -E '^(CHAINUQ_[A-Z_]+|HF_HUB_OFFLINE)=' .env); set +a
LOG="$CHAINUQ_LOGS/haiku"
until grep -q HAIKU_QWEN_final_blind_DONE "$LOG/qwen_final_blind.log" 2>/dev/null && grep -q HAIKU_QWEN_steps_DONE "$LOG/qwen_steps.log" 2>/dev/null; do sleep 5; done
for g in 0 1 2 3; do
  BB=qwen8b scripts/run_haiku_train.sh $g haiku $((2026 + g)) > "$LOG/qwen_train_haiku_$g.log" 2>&1 &
done
wait
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
EA="A-u_reason-u_token+head:saplma+vq@rlogs"
"$CHAINUQ_PYTHON" -m chainuq.analysis.variants --backbones qwen8b --workers 12 --seeds 2026,2027,2028,2029 --tag haiku_jf_qwen \
  --variants "$EA#chainuq-pool2k-n10k-haiku,$EA#chainuq-pool2k-n10k-gptoss,$EA#chainuq-wo-pcs-n10k-gptoss,$EA#chainuq" > "$LOG/qwen_eval.log" 2>&1
"$CHAINUQ_PYTHON" - > "$LOG/qwen_summary.txt" 2>&1 <<'PY'
import json, os, collections, numpy as np
A = os.environ["CHAINUQ_ANALYSIS"]
R = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(f"{A}/runs/haiku_jf_qwen.jsonl"):
    r = json.loads(l); v, bb, ds, sd = r["job"]
    for sp, m in (r["res"] or {}).items():
        R[(v.split("#")[-1], sp)][ds].append(m["auroc"])
for (h, sp), d in sorted(R.items()):
    print(f"{h:34s} {sp:5s} macro={np.mean([np.mean(x) for x in d.values()]):.4f} " + " ".join(f"{k[:5]}={np.mean(x):.3f}({len(x)})" for k, x in sorted(d.items())))
PY
echo QWEN_AUTO_DONE >> "$LOG/qwen_summary.txt"
