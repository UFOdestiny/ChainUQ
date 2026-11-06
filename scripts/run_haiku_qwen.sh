#!/usr/bin/env bash
# Claude Haiku imitation targets for Qwen3-8B (same reduced protocol as Llama): verdicts and step labels on the
# first 10.5k HotpotQA training responses and the 2k unlabeled pools, in two concurrent streams.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source <(grep -E '^(CHAINUQ_[A-Z_]+|HF_HUB_OFFLINE|TOKENIZERS_PARALLELISM)=' .env); set +a
LOG="$CHAINUQ_LOGS/haiku"
J() { "$CHAINUQ_PYTHON" -m chainuq.pipeline.judge_api --judge haiku --backbone qwen8b "$@"; }
for task in final_blind steps; do
  (
    J --task $task --dataset hotpotqa --splits train --max_samples 10500 --concurrency 160
    for ds in musique babi gsm8k math500 strategyqa; do J --task $task --dataset $ds --splits train --concurrency 160; done
    echo HAIKU_QWEN_${task}_DONE
  ) > "$LOG/qwen_$task.log" 2>&1 &
done
wait
