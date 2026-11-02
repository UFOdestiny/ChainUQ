#!/usr/bin/env bash
# One backbone, in four stages:
#   gen    generation (greedy + 10 samples), CoT-UQ keywords, features                          [GPU $GEN_GPU]
#   judge  main-judge annotations: correctness labels, answer-blind step labels and chain verdicts
#   pools  unlabeled train pools of the OOD domains (2,000 and 6,000 questions): generation,
#          answer-blind judge targets, features; the 6,000-question pools go to $CHAINUQ_POOL_ARTIFACTS
#   heads  training-free baselines, supervised probes, and the ChainUQ head (10 seeds)          [GPU $GEN_GPU]
# judge and pools need the judge server on $JUDGE_PORT (scripts/run_main.sh starts it).
# Usage: scripts/run_backbone.sh <llama|qwen8b|ministral> <gen|judge|pools|heads>
set -euo pipefail
source "$(dirname "$0")/common.sh"
BB=$1; STAGE=$2
LOG="$CHAINUQ_LOGS/$BB"; mkdir -p "$LOG"
judge() { "$PY" -m chainuq.pipeline.judge --judge gptoss --backbone $BB --port $JUDGE_PORT "$@"; }
# GSM8K/MATH pools are separate question sets (pipeline.prepare_data --pools); the others subsample train.
pool_data() { case $1 in gsm8k|math500) echo "$CHAINUQ_DATASETS/pools/n$2" ;; *) echo "" ;; esac; }
build_pool() {  # <size> <datasets>
  local n=$1 ds
  for ds in $2; do
    run $GEN_GPU pipeline.generate --backbone $BB --dataset $ds --splits train --max_samples $n \
        --gpu_memory_utilization 0.4 --data_root "$(pool_data $ds $n)"
    for task in steps final_blind; do judge --task $task --dataset $ds --splits train; done
    run $GEN_GPU pipeline.features --backbone $BB --dataset $ds --splits train
  done
}

case $STAGE in
gen)
  stamp "$BB gen"
  for ds in $DATASETS; do
    run $GEN_GPU pipeline.generate --backbone $BB --dataset $ds --splits "$(splits_of $ds)"
    # Semantic Entropy / SelfCheckGPT: 10 temperature samples for the first 1,000 questions of each evaluated split.
    run $GEN_GPU pipeline.generate --backbone $BB --dataset $ds --splits "$(eval_splits $ds)" --max_samples 1000 \
        --n_samples 10 --temperature 1.0
  done > "$LOG/generate.log" 2>&1
  for ds in $DATASETS; do run $GEN_GPU pipeline.cotuq --backbone $BB --datasets $ds --splits "$(eval_splits $ds)"; done > "$LOG/cotuq.log" 2>&1
  for ds in $DATASETS; do run $GEN_GPU pipeline.features --backbone $BB --dataset $ds --splits "$(splits_of $ds)"; done > "$LOG/features.log" 2>&1
  ;;
judge)
  stamp "$BB judge"
  {
    # Reference-based correctness labels where answer matching does not apply.
    for ds in hotpotqa musique babi; do judge --task correct --dataset $ds --splits "$(splits_of $ds)"; done
    # Answer-blind step labels and chain verdicts: PCS targets on HotpotQA train, inference evidence elsewhere.
    for ds in $DATASETS; do for task in steps final_blind; do judge --task $task --dataset $ds --splits "$(splits_of $ds)"; done; done
  } > "$LOG/judge.log" 2>&1
  ;;
pools)
  stamp "$BB pools"
  build_pool 2000 "$OOD" > "$LOG/pools.log" 2>&1  # StrategyQA: all 1,443 train questions
  ( export CHAINUQ_ARTIFACTS="$CHAINUQ_POOL_ARTIFACTS"; build_pool 6000 "musique babi gsm8k math500" ) >> "$LOG/pools.log" 2>&1
  ;;
heads)
  stamp "$BB heads"
  run $GEN_GPU pipeline.scores --backbone $BB --splits validation,test,test2 > "$LOG/scores.log" 2>&1
  run $GEN_GPU pipeline.train --backbone $BB --methods uhead,saplma,factoscope,lookback,meanpool-mlp,attnpool-mlp \
      > "$LOG/train_baselines.log" 2>&1
  run $GEN_GPU pipeline.train --backbone $BB --methods chainuq > "$LOG/train_chainuq.log" 2>&1
  ;;
*) echo "unknown stage $STAGE" >&2; exit 1 ;;
esac
stamp "$BB $STAGE done"
