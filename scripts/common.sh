#!/usr/bin/env bash
# Shared setup for the driver scripts (source it, do not run it): loads .env without overriding variables that
# are already set, applies defaults, and defines helpers.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then
  while IFS='=' read -r key value; do
    key="${key#export }"; key="${key// /}"
    [[ -z "$key" || "$key" == \#* ]] && continue
    value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
    [ -z "${!key+x}" ] && export "$key=$(eval echo "$value")"
  done < "$REPO_ROOT/.env"
fi
export CHAINUQ_ARTIFACTS="${CHAINUQ_ARTIFACTS:-$REPO_ROOT/artifacts}"
export CHAINUQ_DATASETS="${CHAINUQ_DATASETS:-$REPO_ROOT/datasets}"
export CHAINUQ_MODELS="${CHAINUQ_MODELS:-$REPO_ROOT/models}"
export CHAINUQ_LOGS="${CHAINUQ_LOGS:-$CHAINUQ_ARTIFACTS/logs}"
PY="${CHAINUQ_PYTHON:-python}"
VLLM="${CHAINUQ_VLLM:-vllm}"
GEN_GPU="${CHAINUQ_GEN_GPU:-0}"       # generation, feature extraction, head training
JUDGE_GPU="${CHAINUQ_JUDGE_GPU:-1}"   # judge server
JUDGE_PORT="${CHAINUQ_JUDGE_PORT:-8100}"
BACKBONES="llama qwen8b ministral"
DATASETS="hotpotqa musique strategyqa babi gsm8k math500"
OOD="musique strategyqa babi gsm8k math500"
mkdir -p "$CHAINUQ_LOGS"
cd "$REPO_ROOT"

# run <gpu> <module> [args...]: python -m chainuq.<module> on one GPU
run() { local gpu=$1 mod=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu "$PY" -m "chainuq.$mod" "$@"; }
# cpu <module> [args...]: CPU-only module (R-Log fits), one thread per process
cpu() { local mod=$1; shift; OMP_NUM_THREADS=${OMP_NUM_THREADS:-2} "$PY" -m "chainuq.$mod" "$@"; }
stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAINUQ_LOGS/timeline.txt"; }
# serve <model-dir> <port> <gpu> [memory fraction]: start a judge server in the background, wait until it answers
serve() {
  bash "$REPO_ROOT/scripts/serve_model.sh" "$@" &
  SERVER_PID=$!
  until curl -sf "localhost:$2/v1/models" > /dev/null; do
    kill -0 "$SERVER_PID" 2>/dev/null || { echo "server $1 failed, see $CHAINUQ_LOGS" >&2; return 1; }
    sleep 10
  done
}
stop_server() { kill "$SERVER_PID" 2>/dev/null; wait "$SERVER_PID" 2>/dev/null; }
