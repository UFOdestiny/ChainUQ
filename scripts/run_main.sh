#!/usr/bin/env bash
# Main run for all backbones: generation, main-judge annotations and unlabeled pools, then baselines and heads,
# then the official evaluation on test and test2.
set -euo pipefail
source "$(dirname "$0")/common.sh"
serve gpt-oss-20b $JUDGE_PORT $JUDGE_GPU 0.85
for bb in $BACKBONES; do
  scripts/run_backbone.sh $bb gen &   # GPU $GEN_GPU, overlaps with judging of the previous backbone
  GEN_PID=$!
  [ -n "${PREV:-}" ] && scripts/run_backbone.sh $PREV judge
  wait $GEN_PID
  PREV=$bb
done
scripts/run_backbone.sh $PREV judge
for bb in $BACKBONES; do scripts/run_backbone.sh $bb pools; done  # pool generation needs $GEN_GPU as well
stop_server
for bb in $BACKBONES; do scripts/run_backbone.sh $bb heads; done
for bb in $BACKBONES; do
  for split in test test2; do cpu pipeline.evaluate --backbone $bb --split $split > "$CHAINUQ_LOGS/evaluate_${bb}_$split.log" 2>&1 & done
done
wait
stamp "main run done"
