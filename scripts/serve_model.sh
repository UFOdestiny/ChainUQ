#!/usr/bin/env bash
# OpenAI-compatible vLLM server for a judge (or a backbone verifying its own steps).
# Usage: scripts/serve_model.sh <model-dir-name> <port> [gpu] [gpu-memory-fraction]
set -euo pipefail
source "$(dirname "$0")/common.sh"
MODEL=$1; PORT=$2; GPU=${3:-$JUDGE_GPU}; MEM=${4:-0.85}
EXTRA=()
case "$MODEL" in
  gpt-oss-*) EXTRA+=(--reasoning-parser openai_gptoss) ;;
  Qwen3-8B|Qwen3-1.7B) EXTRA+=(--reasoning-parser qwen3) ;;
esac
CUDA_VISIBLE_DEVICES=$GPU exec "$VLLM" serve "$CHAINUQ_MODELS/$MODEL" --port "$PORT" --host 127.0.0.1 \
  --gpu-memory-utilization "$MEM" --max-model-len 16384 --max-num-seqs 256 --seed 2026 "${EXTRA[@]}" \
  > "$CHAINUQ_LOGS/serve_${MODEL}_${PORT}.log" 2>&1
