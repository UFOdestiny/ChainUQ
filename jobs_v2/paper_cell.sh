#!/usr/bin/env bash
# One paper_tables cell for the final head: main section plus every head ablation whose 10 seeds are
# trained and predicted, then (sequentially, same file) the ablation heads that finish training later.
# Llama runs the full ablation set (main-text table, 10 seeds); the other backbones run the head swaps and
# PCS ablations only, with 5 seeds (appendix table).
# Usage: jobs_v2/paper_cell.sh <backbone> <dataset>
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
BB=$1; DS=$2
L=/projects/bduw/kytrio/chainuq_overflow/logs/final_pcs
A=/u/kytrio/nvme_bduw/chainuq_artifacts
if [ "$BB" = llama ]; then
  HEADS="uhead saplma chainuq2bd-core chainuq2bd-core-tok chainuq2bd-lay chainuq2-core chainuq2pcs-steps chainuq2pcs-verdict"; NS=0
else
  HEADS="uhead saplma chainuq2bd-core chainuq2pcs-steps chainuq2pcs-verdict"; NS=5
fi
ready() { [ "$(ls $A/preds/$1/$BB 2>/dev/null | grep -c '^seed')" -ge 10 ] && \
          [ -f "$A/preds/$1/$BB/seed2035/$DS/step/test.jsonl" ]; }
run() {  # sections, heads (comma list)
  OMP_NUM_THREADS=1 nice -n 5 "$CUQ_PY" pipeline/paper_tables.py --backbone $BB --dataset $DS --sections $1 --heads $2 \
    --head_seeds $NS >> $L/pt_${BB}_${DS}.log 2>&1
  echo "[$(date +%T)] $BB $DS $1 [$2] rc=$?" >> $L/done.txt
}
now=(); later=()
for h in $HEADS; do if ready $h; then now+=($h); else later+=($h); fi; done
run main,heads "$(IFS=,; echo "${now[*]}")"
for h in "${later[@]}"; do
  until ready $h; do sleep 60; done
  run heads $h
done
