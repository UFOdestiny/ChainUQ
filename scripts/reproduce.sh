#!/usr/bin/env bash
# Every experiment of the paper, end to end (several GPU-days on two 80 GB GPUs).
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$PY" -m chainuq.pipeline.prepare_data --pools 2000,6000
scripts/run_main.sh
scripts/run_verifiers.sh
scripts/run_ablations.sh
scripts/run_adaptation.sh
scripts/run_paper.sh
