#!/usr/bin/env bash
# Every experiment of the paper, end to end (several GPU-days on two 80 GB GPUs; gpt-oss-120b needs one GPU alone).
set -euo pipefail
source "$(dirname "$0")/common.sh"
"$PY" -m chainuq.pipeline.prepare_data --pools 2000,6000
scripts/run_main.sh
scripts/run_judges.sh
scripts/run_ablations.sh
scripts/run_thinking.sh
scripts/run_analysis.sh
scripts/run_paper.sh
