#!/usr/bin/env bash
# Teacher-forced feature extraction for one backbone on $GEN_GPU.
# Usage: jobs_v2/run_features.sh <backbone> [fmt]
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
BB=$1; FMT=${2:-step}
CUDA_VISIBLE_DEVICES=$GEN_GPU $CUQ_PY pipeline/features.py --backbone $BB --dataset hotpotqa --splits train,validation,test --fmt $FMT 2>&1 | grep '^{'
for ds in musique strategyqa babi gsm8k math500; do
  CUDA_VISIBLE_DEVICES=$GEN_GPU $CUQ_PY pipeline/features.py --backbone $BB --dataset $ds --splits validation,test --fmt $FMT 2>&1 | grep '^{'
done
