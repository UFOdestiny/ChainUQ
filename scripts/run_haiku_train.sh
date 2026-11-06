#!/usr/bin/env bash
# Probe training on Llama with 10k labeled source responses + the 2k unlabeled pools; the imitation targets come
# from JUDGE (gptoss or haiku). Usage: run_haiku_train.sh <gpu> <judge> <seeds> <methods>
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source <(grep -E '^(CHAINUQ_[A-Z_]+|HF_HUB_OFFLINE|TOKENIZERS_PARALLELISM|PYTORCH_CUDA_ALLOC_CONF)=' .env); set +a
GPU=$1 JUDGE=$2 SEEDS=$3 METHODS=${4:-chainuq-pool2k}
CUDA_VISIBLE_DEVICES=$GPU "$CHAINUQ_PYTHON" -m chainuq.pipeline.train --backbone ${BB:-llama} --methods $METHODS \
  --judge $JUDGE --max_train 10000 --tag=-n10k-$JUDGE --seeds $SEEDS
