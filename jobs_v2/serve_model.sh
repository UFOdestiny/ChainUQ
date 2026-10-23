#!/usr/bin/env bash
# Start an OpenAI-compatible vLLM server for a judge (or a self-judging backbone).
# Usage: jobs_v2/serve_model.sh <model-dir-name> <port> [gpu] [gpu_mem_util]
set -euo pipefail
source "$(dirname "$0")/env.sh"
MODEL=$1; PORT=$2; GPU=${3:-$JUDGE_GPU}; MEM=${4:-0.85}
EXTRA=()
case "$MODEL" in
  gpt-oss-*)        EXTRA+=(--reasoning-parser openai_gptoss) ;;
  Qwen3-8B|Qwen3-1.7B) EXTRA+=(--reasoning-parser qwen3) ;;
esac
CUDA_VISIBLE_DEVICES=$GPU exec "$CUQ_VLLM" serve "$CUQ_MODELS_ROOT/$MODEL" \
  --port "$PORT" --host 127.0.0.1 \
  --gpu-memory-utilization "$MEM" --max-model-len 16384 \
  --max-num-seqs 256 --seed 2026 \
  "${EXTRA[@]}" > "${CUQ_SERVE_LOGS:-/projects/bduw/kytrio/chainuq_overflow/logs}/serve_${MODEL}_${PORT}.log" 2>&1
