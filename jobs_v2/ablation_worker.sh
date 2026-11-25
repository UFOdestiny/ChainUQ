#!/usr/bin/env bash
# Head-ablation worker: claims tasks "<idx> <backbone> <method>" from a queue directory (mkdir lock) and trains on one GPU.
# Usage: ablation_worker.sh <gpu> [pid to wait for before starting]
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
GPU=$1; WAIT=${2:-}
L=/projects/bduw/kytrio/chainuq_overflow/logs/paper_ablation; Q=$L/queue
[ -n "$WAIT" ] && while kill -0 "$WAIT" 2>/dev/null; do sleep 30; done
for t in $(ls $Q | grep '^task_' | sort); do
  mkdir "$Q/lock_${t#task_}" 2>/dev/null || continue
  read -r idx bb m < "$Q/$t" 2>/dev/null || continue  # task file removed after the list was taken
  echo "[$(date '+%F %T')] gpu$GPU start $bb $m" >> $L/timeline.txt
  CUDA_VISIBLE_DEVICES=$GPU "$CUQ_PY" pipeline/train.py --backbone $bb --methods $m --seeds all > $L/train_${bb}_${m}.log 2>&1
  echo "[$(date '+%F %T')] $bb $m done rc=$?" >> $L/timeline.txt
done
