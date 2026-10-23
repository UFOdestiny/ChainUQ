#!/usr/bin/env bash
# Alternative labelers: a second correctness labeler (Qwen3-4B, label robustness), a weaker step verifier
# (Qwen3-1.7B), and each backbone verifying its own steps (verifier swap). Run after scripts/run_main.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"
PORT=$((JUDGE_PORT + 1))
steps() { for ds in $DATASETS; do "$PY" -m chainuq.pipeline.judge --judge "$1" --task steps --backbone "$2" --dataset $ds \
            --splits validation,test --port $PORT; done; }
serve Qwen3-4B-Instruct-2507 $PORT $JUDGE_GPU
for bb in $BACKBONES; do for ds in hotpotqa musique babi; do
  "$PY" -m chainuq.pipeline.judge --judge qwen4b --task correct --backbone $bb --dataset $ds --splits test --port $PORT
done; done > "$CHAINUQ_LOGS/verifier_qwen4b.log" 2>&1
stop_server
serve Qwen3-1.7B $PORT $JUDGE_GPU
for bb in $BACKBONES; do steps qwen1b $bb; done > "$CHAINUQ_LOGS/verifier_qwen1b.log" 2>&1
stop_server
for pair in llama:Llama-3.1-8B-Instruct qwen8b:Qwen3-8B ministral:Ministral-8B-Instruct-2410; do
  bb=${pair%%:*}
  serve "${pair#*:}" $PORT $JUDGE_GPU
  steps self-$bb $bb > "$CHAINUQ_LOGS/verifier_self_$bb.log" 2>&1
  stop_server
done
for bb in $BACKBONES; do cpu pipeline.evaluate --backbone $bb --split test > "$CHAINUQ_LOGS/evaluate_${bb}_test.log" 2>&1 & done
wait
stamp "verifiers done"
