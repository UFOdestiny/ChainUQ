#!/usr/bin/env bash
# Claude Haiku chain verdicts on validation/test for the other two backbones (judge at inference only).
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source <(grep -E '^(CHAINUQ_[A-Z_]+|HF_HUB_OFFLINE|TOKENIZERS_PARALLELISM)=' .env); set +a
LOG="$CHAINUQ_LOGS/haiku"
for bb in qwen8b ministral; do
  (
    for ds in math500 gsm8k strategyqa musique babi hotpotqa; do
      "$CHAINUQ_PYTHON" -m chainuq.pipeline.judge_api --judge haiku --task final_blind --backbone $bb --dataset $ds --splits validation,test --concurrency 120
    done
    echo HAIKU_SWAP_${bb}_DONE
  ) > "$LOG/swap_$bb.log" 2>&1 &
done
wait
