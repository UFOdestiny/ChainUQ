#!/usr/bin/env bash
# PCS heads on GPUs 2/3 once the train-split judge labels are done, then validation comparison vs the
# current final head. Usage: jobs_v2/pcs_run.sh <gpu> <judge server pid> "<label line to wait for>" <backbone>...
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
GPU=$1; SERVER=$2; WAIT_FOR=$3; shift 3
L=/projects/bduw/kytrio/chainuq_overflow/logs/pcs
R=/projects/bduw/kytrio/chainuq_overflow/results
stamp() { echo "[$(date +%T)] $*" >> $L/done.txt; }
until grep -q "$WAIT_FOR" $L/done.txt; do sleep 30; done
kill "$SERVER" 2>/dev/null; while kill -0 "$SERVER" 2>/dev/null; do sleep 5; done; sleep 20
for bb in "$@"; do
  ( CUDA_VISIBLE_DEVICES=$GPU "$CUQ_PY" pipeline/train.py --backbone $bb --methods chainuq2pcs --seeds all > $L/train_$bb.log 2>&1
    stamp "train pcs $bb rc=$?"
    OMP_NUM_THREADS=2 nice -n 5 "$CUQ_PY" pipeline/dev_stage2.py --backbone $bb --heads chainuq2bd-core,chainuq2pcs \
      --seeds 2026,2027,2028,2029,2030 --datasets hotpotqa,musique,strategyqa,babi,gsm8k,math500 \
      --cands '[v2c+],[v2c-A+]' --out $R/dev_${bb}_pcs.json > $L/dev_$bb.log 2>&1
    stamp "dev pcs $bb rc=$?" ) &
done
wait
