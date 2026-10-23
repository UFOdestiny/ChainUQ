#!/usr/bin/env bash
# Costs, efficiency benchmark, and every table, figure, and in-text number of the paper (into $CHAINUQ_PAPER_DIR).
set -euo pipefail
source "$(dirname "$0")/common.sh"
cpu experiments.costs > /dev/null
run $GEN_GPU experiments.benchmark --backbone llama --dataset hotpotqa > "$CHAINUQ_LOGS/benchmark.log" 2>&1
cpu experiments.paper
