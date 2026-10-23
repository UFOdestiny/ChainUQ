#!/usr/bin/env bash
# Qwen3-8B's heads stage started before the v2 head was added to it: train it on GPU 2 after Llama's v2 run.
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
while pgrep -f "train.py --backbone llama --methods chainuq2bd" > /dev/null; do sleep 60; done
env CUDA_VISIBLE_DEVICES=$GEN_GPU "$CUQ_PY" pipeline/train.py --backbone qwen8b --methods chainuq2bd --seeds all
echo "[$(date '+%F %T')] qwen8b chainuq v2 head done" >> "$CUQ_ARTIFACTS/logs/full_qwen8b/timeline.txt"
