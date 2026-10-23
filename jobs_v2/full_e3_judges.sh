#!/usr/bin/env bash
# E3: alternative step judges on validation + test (run after all backbones finished full_backbone.sh).
# Stops nothing by itself: start this when GPU $JUDGE_GPU is free (main judge server shut down).
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
# Llama's MATH validation cell was generated from the old (contaminated) split: rebuild it first,
# before any alternative judge reads it. Qwen3-8B and Ministral were generated on the new split.
jobs_v2/repair_math_val.sh llama > "$CUQ_ARTIFACTS/logs/repair_math_val_llama.log" 2>&1
serve_and_wait() {  # <model-dir> <port>
  jobs_v2/serve_model.sh "$1" "$2" "$JUDGE_GPU" 0.85 &
  SERVER_PID=$!
  until curl -sf "localhost:$2/v1/models" > /dev/null; do sleep 10; done
}
steps_for() {  # <judge-key> <port> <backbone>
  for ds in hotpotqa musique strategyqa babi gsm8k math500; do
    "$CUQ_PY" pipeline/judge.py --judge "$1" --task steps --backbone "$3" --dataset $ds --splits validation,test --port "$2"
  done
}
# Second response-label judge (Qwen3-4B-Instruct-2507) on the test splits of the judge-labelled datasets.
serve_and_wait Qwen3-4B-Instruct-2507 8103
for bb in llama qwen8b ministral; do
  for ds in hotpotqa musique babi; do
    "$CUQ_PY" pipeline/judge.py --judge qwen4b --task correct --backbone $bb --dataset $ds --splits test --port 8103
  done
done
kill $SERVER_PID; wait $SERVER_PID 2>/dev/null
# Weak judge (Qwen3-1.7B) on every backbone.
serve_and_wait Qwen3-1.7B 8101
for bb in llama qwen8b ministral; do steps_for qwen1b 8101 $bb; done
kill $SERVER_PID; wait $SERVER_PID 2>/dev/null
# Self-judges: each backbone verifies its own steps.
for pair in "llama:Llama-3.1-8B-Instruct" "qwen8b:Qwen3-8B" "ministral:Ministral-8B-Instruct-2410"; do
  bb=${pair%%:*}; model=${pair#*:}
  serve_and_wait "$model" 8102
  steps_for "self-$bb" 8102 $bb
  kill $SERVER_PID; wait $SERVER_PID 2>/dev/null
done
