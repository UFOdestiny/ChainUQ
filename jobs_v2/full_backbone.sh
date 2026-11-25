#!/usr/bin/env bash
# Full run for ONE backbone, split into stages so that two backbones can overlap:
#   gen    (GPU $GEN_GPU)   generation, CoT-UQ keywords, features, NLI step verifier
#   judge  (server :8100)   main-judge annotations (gpt-oss-20b served on $JUDGE_GPU)
#   heads  (GPU $TRAIN_GPU) Stage-1 heads x 10 seeds, ablations, non-trained scores
# Usage: jobs_v2/full_backbone.sh <llama|qwen8b|ministral> <gen|judge|heads>
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
BB=$1; STAGE=$2
TRAIN_GPU=${TRAIN_GPU:-$JUDGE_GPU}
LOG="$CUQ_ARTIFACTS/logs/full_${BB}"
mkdir -p "$LOG"
on() { local g=$1; shift; env CUDA_VISIBLE_DEVICES=$g "$CUQ_PY" "$@"; }
stamp() { echo "[$(date '+%F %T')] $BB $*" | tee -a "$LOG/timeline.txt"; }
ALL=hotpotqa,musique,strategyqa,babi        # E5 format variants: the four QA benchmarks
ALL6=$ALL,gsm8k,math500
# Stage lock: if the same backbone+stage was started early by hand, wait for it, then re-run (everything skips).
lock_or_wait() {
  local L="$LOG/$1.lock"
  if [ -f "$L" ] && kill -0 "$(cat "$L")" 2>/dev/null; then
    stamp "$1 already running (pid $(cat "$L")); waiting"
    while kill -0 "$(cat "$L")" 2>/dev/null; do sleep 60; done
  fi
  echo $$ > "$L"
}

case $STAGE in
gen)
  # Lock: if this backbone's gen stage was started early by hand, wait for it, then re-run (everything skips).
  LOCK="$LOG/gen.lock"
  if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK")" 2>/dev/null; then
    stamp "gen already running (pid $(cat "$LOCK")); waiting"
    while kill -0 "$(cat "$LOCK")" 2>/dev/null; do sleep 60; done
  fi
  echo $$ > "$LOCK"
  stamp "gen start"
  on $GEN_GPU pipeline/gen.py --backbone $BB --dataset hotpotqa --splits train,validation,test > "$LOG/gen_hotpot.log" 2>&1
  on $GEN_GPU pipeline/gen.py --backbone $BB --dataset musique,strategyqa,babi,gsm8k,math500 --splits validation,test > "$LOG/gen_ood.log" 2>&1
  for fmt in numbered bullet free; do   # E5, test only
    on $GEN_GPU pipeline/gen.py --backbone $BB --dataset $ALL --splits test --fmt $fmt > "$LOG/gen_$fmt.log" 2>&1
  done
  # E8: N=10 temperature samples on the first 1000 questions of validation and test.
  on $GEN_GPU pipeline/gen.py --backbone $BB --dataset $ALL6 --splits validation,test --max_samples 1000 \
      --n_samples 10 --temperature 1.0 > "$LOG/gen_samples.log" 2>&1
  on $GEN_GPU pipeline/cotuq.py --backbone $BB > "$LOG/cotuq.log" 2>&1
  stamp "gen done; features start"
  GEN_GPU=$GEN_GPU jobs_v2/run_features.sh $BB step > "$LOG/feat_step.log" 2>&1
  for fmt in numbered bullet free; do
    for ds in hotpotqa musique strategyqa babi; do
      on $GEN_GPU pipeline/features.py --backbone $BB --dataset $ds --splits test --fmt $fmt
    done
  done > "$LOG/feat_formats.log" 2>&1
  on $GEN_GPU pipeline/nli_steps.py --backbone $BB > "$LOG/nli_steps.log" 2>&1
  stamp "features done"
  rm -f "$LOCK"
  ;;
judge)
  lock_or_wait judge
  stamp "judge start"
  jobs_v2/run_judges.sh $BB 8100 gptoss > "$LOG/judge_main.log" 2>&1
  for fmt in numbered bullet free; do
    for ds in hotpotqa musique babi; do
      "$CUQ_PY" pipeline/judge.py --judge gptoss --task correct --backbone $BB --dataset $ds --splits test --fmt $fmt
    done
    for ds in hotpotqa musique strategyqa babi; do
      "$CUQ_PY" pipeline/judge.py --judge gptoss --task steps --backbone $BB --dataset $ds --splits test --fmt $fmt
    done
  done > "$LOG/judge_formats.log" 2>&1
  stamp "judge done"
  rm -f "$LOG/judge.lock"
  ;;
heads)
  lock_or_wait heads
  # Re-extract any feature cell that has a generation file but no DONE.json (e.g. a transient write error).
  for g in $(find "$CUQ_ARTIFACTS/gen/$BB" -name "*.jsonl" ! -name "*.samples.jsonl"); do
    rel=${g#"$CUQ_ARTIFACTS/gen/$BB/"}; ds=${rel%%/*}; rest=${rel#*/}; fmt=${rest%%/*}; split=$(basename "$g" .jsonl)
    [ "$split" = train ] && [ "$ds" != hotpotqa ] && continue
    if [ ! -f "$CUQ_ARTIFACTS/feat/$BB/$ds/$fmt/$split/DONE.json" ]; then
      stamp "re-extracting missing features $ds/$fmt/$split"
      on $TRAIN_GPU pipeline/features.py --backbone $BB --dataset $ds --splits $split --fmt $fmt >> "$LOG/feat_repair.log" 2>&1
    fi
  done
  # Baselines first (non-trained scores, then trained baseline heads), then ChainUQ and its ablations.
  stamp "baselines start"
  on $TRAIN_GPU pipeline/scores.py --backbone $BB --methods mcp,ppl,mte,ccp,judge_final,judge_step,rm,se,selfcheck,cotuq \
      > "$LOG/scores.log" 2>&1
  on $TRAIN_GPU pipeline/train.py --backbone $BB --methods uhead,saplma,factoscope,lookback,meanpool-mlp,attnpool-mlp \
      --seeds all > "$LOG/train_baselines.log" 2>&1
  stamp "baselines done"
  # Baseline-only tables right away (CPU); full_evaluate.sh later overwrites them with the complete tables.
  ( "$CUQ_PY" pipeline/evaluate.py --backbone $BB > "$LOG/evaluate_baselines.log" 2>&1
    stamp "baseline tables written" ) &
  on $TRAIN_GPU pipeline/train.py --backbone $BB --methods chainuq --seeds all > "$LOG/train_chainuq.log" 2>&1
  stamp "chainuq done"
  on $TRAIN_GPU pipeline/train.py --backbone $BB --methods chainuq-lb,chainuq-cs,chainuq-cf,chainuq-fr --seeds all \
      --eval_fmts step > "$LOG/train_ablation.log" 2>&1
  # ChainUQ v2 head (layer + token branches, branch dropout), designed on validation data only.
  on $TRAIN_GPU pipeline/train.py --backbone $BB --methods chainuq2bd --seeds all > "$LOG/train_chainuq2bd.log" 2>&1
  stamp "chainuq v2 head done"
  wait
  stamp "heads done"
  rm -f "$LOG/heads.lock"
  ;;
*) echo "unknown stage $STAGE"; exit 1 ;;
esac
