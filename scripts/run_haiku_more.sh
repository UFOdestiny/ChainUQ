#!/usr/bin/env bash
# Continuation of run_haiku.sh stream A, split in two concurrent streams (verdicts on the pools; step labels).
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source <(grep -E '^(CHAINUQ_[A-Z_]+|HF_HUB_OFFLINE|TOKENIZERS_PARALLELISM)=' .env); set +a
LOG="$CHAINUQ_LOGS/haiku"
J() { "$CHAINUQ_PYTHON" -m chainuq.pipeline.judge_api --judge haiku --backbone llama "$@"; }
WAITPID=$1
(
  while kill -0 $WAITPID 2>/dev/null; do sleep 5; done
  for ds in babi gsm8k math500 strategyqa; do J --task final_blind --dataset $ds --splits train --concurrency 128; done
  echo HAIKU_POOLVERDICT_DONE
) >> "$LOG/train_targets.log" 2>&1 &
(
  J --task steps --dataset hotpotqa --splits train --max_samples 10500 --concurrency 160
  for ds in musique babi gsm8k math500 strategyqa; do J --task steps --dataset $ds --splits train --concurrency 160; done
  echo HAIKU_STEPS_DONE
) > "$LOG/train_steps.log" 2>&1 &
wait
