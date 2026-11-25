#!/usr/bin/env bash
# Health snapshot of the full run: stages, liveness, GPUs, judge server, errors, disk, output counts.
source "$(dirname "$0")/env.sh" > /dev/null 2>&1
A=$CUQ_ARTIFACTS
echo "== $(date '+%F %T')"
echo "-- stages"
cat "$A/logs/run_everything.txt" 2>/dev/null
for bb in llama qwen8b ministral; do
  [ -f "$A/logs/full_$bb/timeline.txt" ] && tail -n 2 "$A/logs/full_$bb/timeline.txt"
done
echo "-- processes"
pgrep -f "jobs_v2/run_everything.sh" > /dev/null && echo "run_everything.sh: alive" || echo "run_everything.sh: NOT RUNNING"
for p in pipeline/gen.py pipeline/features.py pipeline/judge.py pipeline/train.py pipeline/scores.py \
         pipeline/cotuq.py pipeline/nli_steps.py pipeline/evaluate.py; do
  n=$(pgrep -fc "$CUQ_PY $p" 2>/dev/null)
  [ "${n:-0}" -gt 0 ] && echo "$p x$n: $(pgrep -fa "$CUQ_PY $p" | head -1 | grep -oE -- '--backbone [a-z0-9]+.*' | cut -c1-90)"
done
echo "-- gpus (2,3)"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '3,4p'
echo "-- judge server :8100"
curl -sf -m 5 localhost:8100/v1/models > /dev/null && echo up || echo "down/not started"
echo "-- errors in logs touched in the last 40 min"
find "$A/logs" -name "*.log" -mmin -40 -print0 2>/dev/null | xargs -0 -r grep -lE "Traceback|CUDA out of memory|RuntimeError|Killed" 2>/dev/null \
  | while read -r f; do echo "$f"; grep -E "Traceback|Error|out of memory|Killed" "$f" | tail -n 2 | cut -c1-200; done
echo "-- outputs"
for bb in llama qwen8b ministral; do
  g=$(timeout 20 find -L "$A/gen/$bb" -name "*.jsonl" 2>/dev/null | wc -l)
  j=$(timeout 20 find -L "$A/judge" -path "*/$bb/*" -name "*.jsonl" 2>/dev/null | wc -l)
  f=$(timeout 20 find -L "$A/feat/$bb" -name DONE.json 2>/dev/null | wc -l)
  c=$(timeout 20 find -L "$A/ckpt" -path "*/$bb/*" -name meta.json 2>/dev/null | wc -l)
  echo "$bb: gen=$g judge=$j feat_done=$f heads_done=$c"
done
echo "-- disk"
echo "projects overflow: $(timeout 20 du -sh /projects/bduw/kytrio/chainuq_overflow 2>/dev/null | cut -f1)"
echo "-- slurm time left: $(squeue -j 23020607 -h -o %L 2>/dev/null)"
