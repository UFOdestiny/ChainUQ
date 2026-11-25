#!/usr/bin/env bash
# Llama AttnPool-MLP died at seed 2028 on a /work/nvme quota error; finish seeds 2028-2035 after Llama's heads stage.
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
until grep -q "llama heads done" "$CUQ_ARTIFACTS/logs/full_llama/timeline.txt" 2>/dev/null; do sleep 60; done
env CUDA_VISIBLE_DEVICES=$JUDGE_GPU "$CUQ_PY" pipeline/train.py --backbone llama --methods attnpool-mlp --seeds all
echo "[$(date '+%F %T')] llama attnpool-mlp rerun done"
