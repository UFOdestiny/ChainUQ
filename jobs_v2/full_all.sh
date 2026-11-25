#!/usr/bin/env bash
# Whole main run (E0-E2, E5-E8) for the three backbones, overlapping the two GPUs:
#   GPU 2: gen+features of backbone k        GPU 3: judge server + judging/heads of backbone k-1
# gpt-oss-20b is served at 0.45 memory on GPU 3 so the heads (~30 GB) fit next to it.
# After this finishes run jobs_v2/full_e3_judges.sh (needs GPU 3 free) and jobs_v2/full_evaluate.sh.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
nohup jobs_v2/serve_model.sh gpt-oss-20b 8100 "$JUDGE_GPU" 0.45 > /dev/null 2>&1 &
until curl -sf localhost:8100/v1/models > /dev/null; do sleep 15; done
PREV=""
for bb in llama qwen8b ministral; do
  jobs_v2/full_backbone.sh $bb gen &
  GEN_PID=$!
  [ -n "$PREV" ] && jobs_v2/full_backbone.sh $PREV judge && jobs_v2/full_backbone.sh $PREV heads
  wait $GEN_PID
  PREV=$bb
done
jobs_v2/full_backbone.sh $PREV judge && jobs_v2/full_backbone.sh $PREV heads
pid=$(ss -ltnp 2>/dev/null | grep ":8100 " | grep -oP 'pid=\K[0-9]+' | head -1); [ -n "$pid" ] && kill "$pid"
echo "[$(date '+%F %T')] main run finished"
