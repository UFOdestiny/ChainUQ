#!/usr/bin/env bash
# Shared setup for the driver scripts (source it, do not run it): loads .env through chainuq.config without
# overriding variables that are already set, applies defaults, and defines helpers.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Bootstrap Python from the caller; .env may then select the driver interpreter.
CHAINUQ_CONFIG_EXPORTS="$(PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "${CHAINUQ_PYTHON:-python}" -m chainuq.config)" || return 1
eval "$CHAINUQ_CONFIG_EXPORTS"
unset CHAINUQ_CONFIG_EXPORTS
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${CHAINUQ_PYTHON:-python}"
VLLM="${CHAINUQ_VLLM:-vllm}"
GEN_GPU="${CHAINUQ_GEN_GPU:-0}"       # generation, feature extraction, head training
JUDGE_GPU="${CHAINUQ_JUDGE_GPU:-1}"   # judge server
JUDGE_PORT="${CHAINUQ_JUDGE_PORT:-8100}"
WORKERS="${CHAINUQ_WORKERS:-30}"      # CPU processes of the Stage-2 analyses
BACKBONES="llama qwen8b ministral"
DATASETS="hotpotqa musique strategyqa babi gsm8k math500"
OOD="musique strategyqa babi gsm8k math500"
SEEDS3=2026,2027,2028
SEEDS5=2026,2027,2028,2029,2030
SEEDS10=2026,2027,2028,2029,2030,2031,2032,2033,2034,2035
mkdir -p "$CHAINUQ_LOGS"
cd "$REPO_ROOT"

# run <gpu> <module> [args...]: python -m chainuq.<module> on one GPU
run() { local gpu=$1 mod=$2; shift 2; CUDA_VISIBLE_DEVICES=$gpu "$PY" -m "chainuq.$mod" "$@"; }
# cpu <module> [args...]: CPU-only module, one BLAS thread per process (the analyses parallelize over processes)
cpu() { local mod=$1; shift; OMP_NUM_THREADS=${OMP_NUM_THREADS:-1} "$PY" -m "chainuq.$mod" "$@"; }
stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAINUQ_LOGS/timeline.txt"; }
# split lists: StrategyQA has no train/test2 split of its own (its train questions form the unlabeled pool)
splits_of() { case $1 in hotpotqa) echo train,validation,test,test2 ;; strategyqa) echo validation,test ;; *) echo validation,test,test2 ;; esac; }
eval_splits() { [ "$1" = strategyqa ] && echo validation,test || echo validation,test,test2; }
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
