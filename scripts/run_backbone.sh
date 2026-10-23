#!/usr/bin/env bash
# Main run for one backbone, in three stages:
#   gen    generation (greedy + 10 samples), CoT-UQ keywords, features, NLI step verifier   [GPU $GEN_GPU]
#   judge  main-judge annotations; needs the judge server on $JUDGE_PORT (scripts/run_main.sh starts it)
#   heads  training-free baselines, supervised probes, and the ChainUQ head (10 seeds)       [GPU $GEN_GPU]
# Usage: scripts/run_backbone.sh <llama|qwen8b|ministral> <gen|judge|heads>
set -euo pipefail
source "$(dirname "$0")/common.sh"
BB=$1; STAGE=$2
LOG="$CHAINUQ_LOGS/$BB"; mkdir -p "$LOG"
splits_of() { [ "$1" = hotpotqa ] && echo train,validation,test,test2 || { [ "$1" = strategyqa ] && echo validation,test || echo validation,test,test2; }; }
eval_splits() { [ "$1" = strategyqa ] && echo validation,test || echo validation,test,test2; }

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
  for ds in $DATASETS; do run $GEN_GPU pipeline.nli_steps --backbone $BB --datasets $ds --splits "$(eval_splits $ds)"; done > "$LOG/nli_steps.log" 2>&1
  ;;
judge)
  stamp "$BB judge"
  judge() { "$PY" -m chainuq.pipeline.judge --judge gptoss --backbone $BB --port $JUDGE_PORT "$@"; }
  {
    for ds in hotpotqa musique babi; do judge --task correct --dataset $ds --splits "$(splits_of $ds)"; done
    # Answer-blind step labels and chain verdicts; on HotpotQA train they are the PCS training targets.
    for ds in $DATASETS; do
      for task in steps final_blind; do judge --task $task --dataset $ds --splits "$(splits_of $ds)"; done
      judge --task steps_noconcl --dataset $ds --splits "$(eval_splits $ds)"
    done
  } > "$LOG/judge.log" 2>&1
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
