#!/usr/bin/env bash
# ChainUQ v2 head for Llama (seeds 2029-2035 + format predictions) and Qwen3-8B, sequentially on GPU 2.
# (Ministral's heads stage trains it itself.)
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
for bb in llama qwen8b; do
  env CUDA_VISIBLE_DEVICES=$GEN_GPU "$CUQ_PY" pipeline/train.py --backbone $bb --methods chainuq2bd --seeds all
  echo "[$(date '+%F %T')] $bb chainuq v2 head done" >> "$CUQ_ARTIFACTS/logs/full_${bb}/timeline.txt"
done
