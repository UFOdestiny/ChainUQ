#!/usr/bin/env bash
# Head ablations, label efficiency, per-cell analyses, and significance. Run after scripts/run_verifiers.sh.
#   Llama: every ablation with 10 seeds; Qwen3-8B / Ministral: PCS ablations and head swaps with 5 seeds.
set -euo pipefail
source "$(dirname "$0")/common.sh"
FIVE=2026,2027,2028,2029,2030
LOG="$CHAINUQ_LOGS/ablations"; mkdir -p "$LOG"
run $GEN_GPU pipeline.train --backbone llama > "$LOG/train_llama.log" 2>&1 \
    --methods chainuq-wo-step,chainuq-wo-verdict,chainuq-wo-pcs,chainuq-wo-pcs-layer,chainuq-wo-pcs-token,chainuq-wo-pcs-bdrop
for bb in qwen8b ministral; do
  run $GEN_GPU pipeline.train --backbone $bb --methods chainuq-wo-step,chainuq-wo-verdict,chainuq-wo-pcs --seeds $FIVE \
      > "$LOG/train_$bb.log" 2>&1
done
for bb in $BACKBONES; do run $GEN_GPU pipeline.train --backbone $bb --methods uhead-pcs --seeds $FIVE >> "$LOG/train_$bb.log" 2>&1; done
# Label efficiency (Llama, 3 seeds): correctness labels on 10% / 25% of HotpotQA train.
for frac in 10 25; do
  run $GEN_GPU pipeline.train --backbone llama --methods chainuq,chainuq-wo-pcs --seeds 2026,2027,2028 \
      --label_frac 0.$frac --tag=-lf$frac > "$LOG/label_frac_$frac.log" 2>&1
done
for bb in $BACKBONES; do
  for ds in $DATASETS; do
    (
      if [ $bb = llama ]; then
        cpu experiments.analysis --backbone $bb --dataset $ds
        cpu experiments.analysis --backbone $bb --dataset $ds --sections heads --head_seeds 3 \
            --heads chainuq-lf10,chainuq-lf25,chainuq-wo-pcs-lf10,chainuq-wo-pcs-lf25
      else
        cpu experiments.analysis --backbone $bb --dataset $ds --head_seeds 5 \
            --heads uhead,saplma,uhead-pcs,chainuq-wo-step,chainuq-wo-verdict,chainuq-wo-pcs
      fi
      cpu experiments.significance --backbone $bb --dataset $ds
    ) > "$LOG/analysis_${bb}_$ds.log" 2>&1 &
  done
  wait
done
stamp "ablations done"
