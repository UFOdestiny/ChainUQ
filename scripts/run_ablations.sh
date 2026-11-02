#!/usr/bin/env bash
# Head ablations and controls (chainuq.heads.HEADS). Run after scripts/run_main.sh.
#   imitation: no PCS (10 seeds); PCS on the labeled source only; 2,000-question pools; one PCS target on the pools
#   probes:    UHead with PCS (10 seeds) and with the 6,000-question pools; SAPLMA distilled from the pool verdicts
#   domains:   leave-one-domain-out pools
#   labels:    1% / 5% / 25% of the HotpotQA correctness labels (Llama and Qwen3-8B)
set -euo pipefail
source "$(dirname "$0")/common.sh"
LOG="$CHAINUQ_LOGS/ablations"; mkdir -p "$LOG"
for bb in $BACKBONES; do
  {
    run $GEN_GPU pipeline.train --backbone $bb --methods chainuq-wo-pcs,uhead-pcs --seeds $SEEDS10
    run $GEN_GPU pipeline.train --backbone $bb --seeds $SEEDS3 \
        --methods chainuq-source,chainuq-pool2k,chainuq-pool-steps,chainuq-pool-verdict,uhead-pcs-pool6k,saplma-pool6k-pseudo
    run $GEN_GPU pipeline.train_lodo --backbone $bb --seeds $SEEDS3
  } > "$LOG/train_$bb.log" 2>&1
done
for bb in llama qwen8b; do
  for f in 01 05 25; do
    run $GEN_GPU pipeline.train --backbone $bb --seeds $SEEDS3 --label_frac 0.$f --tag=-lf$f \
        --methods chainuq,chainuq-source,chainuq-wo-pcs,saplma
  done > "$LOG/label_budget_$bb.log" 2>&1
done
stamp "ablations done"
