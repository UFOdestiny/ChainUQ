#!/usr/bin/env bash
# E3 alternative judges, run early on GPU 3 next to the gpt-oss server (each server at 0.4 memory).
# Llama's MATH validation is excluded here (it is rebuilt by repair_math_val.sh inside full_e3_judges.sh,
# which then re-runs these judges on the new cell; everything else is skipped as already done).
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
serve_and_wait() { jobs_v2/serve_model.sh "$1" "$2" "$JUDGE_GPU" 0.4 & SERVER_PID=$!
                   until curl -sf "localhost:$2/v1/models" > /dev/null; do sleep 10; done; }
stop_server() { kill $SERVER_PID; wait $SERVER_PID 2>/dev/null; sleep 10; }
steps_for() {  # judge port backbone
  for ds in hotpotqa musique strategyqa babi gsm8k math500; do
    sp=validation,test; [ "$3" = llama ] && [ $ds = math500 ] && sp=test
    "$CUQ_PY" pipeline/judge.py --judge "$1" --task steps --backbone "$3" --dataset $ds --splits $sp --port "$2"
  done
}
serve_and_wait Qwen3-4B-Instruct-2507 8103
for bb in llama qwen8b ministral; do for ds in hotpotqa musique babi; do
  "$CUQ_PY" pipeline/judge.py --judge qwen4b --task correct --backbone $bb --dataset $ds --splits test --port 8103
done; done
stop_server
serve_and_wait Qwen3-1.7B 8101
for bb in llama qwen8b ministral; do steps_for qwen1b 8101 $bb; done
stop_server
for pair in "llama:Llama-3.1-8B-Instruct" "qwen8b:Qwen3-8B" "ministral:Ministral-8B-Instruct-2410"; do
  bb=${pair%%:*}; model=${pair#*:}
  serve_and_wait "$model" 8102
  steps_for "self-$bb" 8102 $bb
  stop_server
done
echo "[$(date '+%F %T')] e3 early judges done"
