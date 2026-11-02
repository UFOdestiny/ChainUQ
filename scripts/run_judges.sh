#!/usr/bin/env bash
# Further judges, used at inference only (PCS keeps the main judge): gpt-oss-120b at low and high reasoning effort,
# Qwen3-8B in thinking mode (another model family), and a second correctness labeler (Qwen3-4B) for label
# robustness. Run after scripts/run_main.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"
LOG="$CHAINUQ_LOGS/judges"; mkdir -p "$LOG"
PORT=$((JUDGE_PORT + 1))
judge() { "$PY" -m chainuq.pipeline.judge --port $PORT "$@"; }

serve gpt-oss-120b $PORT $JUDGE_GPU 0.90
for bb in $BACKBONES; do
  for ds in $DATASETS; do judge --judge gptoss120b --task final_blind --backbone $bb --dataset $ds --splits "$(eval_splits $ds)"; done
  for ds in gsm8k math500; do
    judge --judge gptoss120b-high --task final_blind --backbone $bb --dataset $ds --splits validation,test --max_tokens 8192
  done
done > "$LOG/gptoss120b.log" 2>&1
stop_server

serve Qwen3-8B $PORT $JUDGE_GPU 0.85
for bb in $BACKBONES; do for ds in $DATASETS; do
  judge --judge qwen3judge --task final_blind --backbone $bb --dataset $ds --splits validation,test --max_tokens 3072
done; done > "$LOG/qwen3judge.log" 2>&1
stop_server

serve Qwen3-4B-Instruct-2507 $PORT $JUDGE_GPU 0.85
for bb in $BACKBONES; do for ds in hotpotqa musique babi; do
  judge --judge qwen4b --task correct --backbone $bb --dataset $ds --splits test
done; done > "$LOG/qwen4b.log" 2>&1
stop_server
stamp "judges done"
