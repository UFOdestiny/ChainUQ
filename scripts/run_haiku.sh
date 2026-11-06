#!/usr/bin/env bash
# Claude Haiku (API) as a black-box judge for Llama: imitation targets on 10k source responses and the
# 2k unlabeled pools (stream A), and the chain verdict on validation/test (stream B). Both streams run concurrently.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source <(grep -E '^(CHAINUQ_[A-Z_]+|HF_HUB_OFFLINE|TOKENIZERS_PARALLELISM)=' .env); set +a
LOG="$CHAINUQ_LOGS/haiku"; mkdir -p "$LOG"
J() { "$CHAINUQ_PYTHON" -m chainuq.pipeline.judge_api --judge haiku --backbone llama "$@"; }
(
  for task in final_blind steps; do
    J --task $task --dataset hotpotqa --splits train --max_samples 10500 --concurrency 128
    for ds in musique babi gsm8k math500 strategyqa; do J --task $task --dataset $ds --splits train --concurrency 128; done
  done
  echo HAIKU_TRAIN_DONE
) > "$LOG/train_targets.log" 2>&1 &
(
  for ds in hotpotqa musique babi gsm8k math500 strategyqa; do J --task final_blind --dataset $ds --splits validation,test --concurrency 48; done
  echo HAIKU_EVAL_DONE
) > "$LOG/eval_verdicts.log" 2>&1 &
wait
