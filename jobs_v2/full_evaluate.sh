#!/usr/bin/env bash
# Tables for every backbone + measured costs (CPU only). Backbones run in parallel, 2 threads each.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
for bb in llama qwen8b ministral; do
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 "$CUQ_PY" pipeline/evaluate.py --backbone $bb > "$CUQ_ARTIFACTS/results/evaluate_$bb.log" 2>&1 &
done
wait
"$CUQ_PY" pipeline/costs.py > /dev/null
