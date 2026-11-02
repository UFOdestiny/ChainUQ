#!/usr/bin/env bash
# Long reasoning: Qwen3-8B in thinking mode (backbone key qwen8b-think) on a reduced pipeline (10,000 HotpotQA train
# questions, 2,000-question pools, no test2, 5 head seeds). The main judge reads the visible response; the
# trace-visible judge (gptoss-trace) also reads the thinking block. Run after scripts/run_main.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"
BB=qwen8b-think
LOG="$CHAINUQ_LOGS/$BB"; mkdir -p "$LOG"
gen() { run $GEN_GPU pipeline.generate --backbone $BB --max_model_len 8192 --gpu_memory_utilization 0.85 "$@"; }
judge() { "$PY" -m chainuq.pipeline.judge --backbone $BB --port $JUDGE_PORT "$@"; }
splits() { [ "$1" = hotpotqa ] && echo train,validation,test || echo validation,test,train; }

{
  gen --dataset hotpotqa --splits train --max_samples 10000
  gen --dataset hotpotqa --splits validation,test
  for ds in $OOD; do gen --dataset $ds --splits validation,test; done
  for ds in musique babi strategyqa; do gen --dataset $ds --splits train --max_samples 2000; done
  for ds in gsm8k math500; do gen --dataset $ds --splits train --max_samples 2000 --data_root "$CHAINUQ_DATASETS/pools/n2000"; done
} > "$LOG/generate.log" 2>&1

serve gpt-oss-20b $JUDGE_PORT $JUDGE_GPU 0.85
{
  judge --judge gptoss --task correct --dataset hotpotqa --splits train,validation,test
  for ds in musique babi; do judge --judge gptoss --task correct --dataset $ds --splits validation,test; done
  for ds in $DATASETS; do for task in steps final_blind; do judge --judge gptoss --task $task --dataset $ds --splits "$(splits $ds)"; done; done
  for ds in $DATASETS; do judge --judge gptoss-trace --task final_blind --dataset $ds --splits validation,test; done
} > "$LOG/judge.log" 2>&1
stop_server

for ds in $DATASETS; do run $GEN_GPU pipeline.features --backbone $BB --dataset $ds --splits "$(splits $ds)"; done > "$LOG/features.log" 2>&1
{
  run $GEN_GPU pipeline.train --backbone $BB --methods saplma,uhead --seeds $SEEDS5
  run $GEN_GPU pipeline.train --backbone $BB --methods chainuq-pool2k --seeds $SEEDS5
} > "$LOG/train.log" 2>&1
stamp "thinking backbone done"
