#!/usr/bin/env bash
# Stage-2 analyses behind every table and number of the paper (CPU, except the efficiency benchmark).
# Run after scripts/run_main.sh, run_judges.sh, run_ablations.sh and run_thinking.sh.
set -euo pipefail
source "$(dirname "$0")/common.sh"
LOG="$CHAINUQ_LOGS/analysis"; mkdir -p "$LOG"
INFO="$CHAINUQ_ANALYSIS/information"; mkdir -p "$INFO"
variants() { cpu analysis.variants --workers $WORKERS "$@"; }
information() { cpu analysis.information --workers $WORKERS "$@"; }

# Evidence of ChainUQ (EB) and ChainUQ-A (EA); chain features without (C*) and with (P*) the imitation outputs.
EB="B-u_reason-u_token-u_judge+ptrue+head:saplma+vq"; EA="A-u_reason-u_token+head:saplma+vq"
CB="B-u_reason-u_token-u_judge-vhat+ptrue@rlogs"; CA="A-u_reason-u_token-vhat@rlogs"
PB="B-u_reason-u_token-u_judge+ptrue+vq@rlogs"; PA="A-u_reason-u_token+vq@rlogs"
FUS="S+verdict+ptrue@lrsel"  # probe + judge fusion
join() { local IFS=,; echo "$*"; }
probe_rows() { echo "$FUS#$1 $CB#$1 S@iso#$1 $CA#$1"; }
head_rows() { echo "$(probe_rows $1) $PB#$1 $PA#$1 $EB@rlogs#$1 $EA@rlogs#$1"; }
final_rows() { local h; for h in "$@"; do echo "$EA@rlogs#$h $EB@rlogs#$h"; done; }

stamp "analysis: evidence cache"
cpu analysis.cache --workers 12 > "$LOG/cache.log" 2>&1
cpu analysis.cache --backbones qwen8b-think --splits validation,test --workers 12 >> "$LOG/cache.log" 2>&1

stamp "analysis: variant runs"
{
  variants --tag attribution --seeds $SEEDS10 --variants "$(join $(probe_rows saplma) $(probe_rows uhead) \
      $(head_rows chainuq-wo-pcs) $(head_rows chainuq) \
      $PB#uhead-pcs $PA#uhead-pcs $EB@rlogs#uhead-pcs $EA@rlogs#uhead-pcs \
      $PB#uhead-pcs-pool6k $PA#uhead-pcs-pool6k $EB@rlogs#uhead-pcs-pool6k $EA@rlogs#uhead-pcs-pool6k \
      $EB@rlog#chainuq $EA@rlog#chainuq $EB@lrsel#chainuq $EA@lrsel#chainuq $EB@gbm#chainuq $EA@gbm#chainuq \
      $EB-ptrue@rlogs#chainuq $EB-vhat-vq@rlogs#chainuq $EA-vhat-vq@rlogs#chainuq $EB-span@rlogs#chainuq \
      $EA-span@rlogs#chainuq $EB-u_gap@rlogs#chainuq $EA-u_gap@rlogs#chainuq \
      B+ptrue+head:saplma+vq@rlogs#chainuq A+head:saplma+vq@rlogs#chainuq)"
  variants --tag attribution --seeds $SEEDS3 --variants "$(join $(head_rows chainuq-source) $(head_rows chainuq-pool2k))"
  variants --tag target_pool_controls --seeds $SEEDS3 --variants "$(join $(final_rows chainuq-pool-steps chainuq-pool-verdict))"
  variants --tag leave_one_domain_out --seeds $SEEDS3 \
      --variants "$(join $(final_rows chainuq-lodo-musique chainuq-lodo-babi chainuq-lodo-gsm8k chainuq-lodo-math500 chainuq-lodo-strategyqa))"
  variants --tag probe_distillation --seeds $SEEDS3 --variants "$(join $(probe_rows saplma-pool6k-pseudo))"
  variants --tag calibration_grid --backbones llama,qwen8b --seeds $SEEDS3 --variants "$(join $(final_rows \
      chainuq-wo-pcs chainuq-source chainuq chainuq-wo-pcs-lf05 chainuq-wo-pcs-lf25 chainuq-source-lf05 \
      chainuq-source-lf25 chainuq-lf05 chainuq-lf25))"
  for f in 01 05 25; do
    variants --tag calibration_grid_extended --backbones llama,qwen8b --seeds $SEEDS3 --variants "$(join $(for h in chainuq-wo-pcs chainuq-source chainuq; do
        echo "${EA/head:saplma/head:saplma-lf$f}@rlogs#$h-lf$f ${EB/head:saplma/head:saplma-lf$f}@rlogs#$h-lf$f"; done))"
  done
  # Judge swap at inference: the chain verdict and P(true) of another judge.
  JUDGED="$FUS#saplma,$FUS#uhead,$EB@rlogs#chainuq"
  CHAINUQ_ANALYSIS_JUDGE=gptoss120b variants --tag judge_strength --seeds $SEEDS10 --variants "$JUDGED"
  CHAINUQ_ANALYSIS_JUDGE=gptoss120b-high CHAINUQ_ANALYSIS_SPLITS=test variants --tag judge_reasoning_effort \
      --datasets gsm8k,math500 --seeds $SEEDS10 --variants "$JUDGED"
  CHAINUQ_ANALYSIS_JUDGE=qwen3judge CHAINUQ_ANALYSIS_SPLITS=test variants --tag cross_family_judge --seeds $SEEDS3 --variants "$JUDGED"
  # Long reasoning (thinking-mode Qwen3-8B), with the main judge and with the trace-visible judge.
  CHAINUQ_ANALYSIS_SPLITS=test variants --tag thinking --backbones qwen8b-think --seeds $SEEDS5 \
      --variants "$(join $(final_rows chainuq-pool2k) $FUS#saplma $FUS#uhead S@iso#saplma S@iso#uhead)"
  CHAINUQ_ANALYSIS_JUDGE=gptoss-trace CHAINUQ_ANALYSIS_SPLITS=test variants --tag judge_trace --backbones qwen8b-think \
      --seeds $SEEDS5 --variants "$EB@rlogs#chainuq-pool2k,$FUS#saplma,$FUS#uhead"
} > "$LOG/variants.log" 2>&1

stamp "analysis: information decomposition and routing"
{
  information --head chainuq
  information --head chainuq --model gbm --out "$INFO/decomposition_chainuq_gbm.json"
  for h in chainuq-wo-pcs chainuq-source chainuq-pool2k chainuq chainuq-pool-steps chainuq-pool-verdict; do
    information --head $h --seeds $SEEDS3 --out "$INFO/decomposition_${h}_s3.json"
  done
  CHAINUQ_ANALYSIS_JUDGE=gptoss120b information --head chainuq --out "$INFO/decomposition_chainuq_gptoss120b.json"
  CHAINUQ_ANALYSIS_JUDGE=qwen3judge information --head chainuq --out "$INFO/decomposition_chainuq_qwen3judge.json"
  information --head chainuq-pool2k --backbones qwen8b-think --seeds $SEEDS5 --out "$INFO/decomposition_thinking.json"
  CHAINUQ_ANALYSIS_JUDGE=gptoss-trace information --head chainuq-pool2k --backbones qwen8b-think --seeds $SEEDS5 \
      --out "$INFO/decomposition_thinking_trace.json"
  cpu analysis.routing --head chainuq --workers $WORKERS
} > "$LOG/information.log" 2>&1

stamp "analysis: response-level analyses"
{
  cpu analysis.predictions --workers $WORKERS --variants "$EB@rlogs#chainuq,$EA@rlogs#chainuq,$FUS#saplma,$FUS#uhead"
  cpu analysis.fusion
  cpu analysis.responses --workers 18
  cpu analysis.subtasks
  cpu analysis.unseen --workers $WORKERS
} > "$LOG/responses.log" 2>&1

stamp "analysis: per-cell calibrators and significance"
for bb in $BACKBONES; do
  for ds in $DATASETS; do
    ( cpu analysis.calibrators --backbone $bb --dataset $ds; cpu analysis.significance --backbone $bb --dataset $ds --seeds 10 ) \
        > "$LOG/cell_${bb}_$ds.log" 2>&1 &
  done
  wait
done

stamp "analysis: costs and efficiency"
cpu analysis.costs > "$LOG/costs.log" 2>&1
run $GEN_GPU analysis.benchmark --backbone llama --dataset hotpotqa > "$LOG/benchmark.log" 2>&1
stamp "analysis done"
