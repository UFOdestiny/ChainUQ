#!/usr/bin/env bash
# Main run (GPU 2+3) -> E3 alternative judges (GPU 3) -> tables + costs (CPU).
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
echo "[$(date '+%F %T')] start" >> "$CUQ_ARTIFACTS/logs/run_everything.txt"
jobs_v2/full_all.sh
echo "[$(date '+%F %T')] main run done" >> "$CUQ_ARTIFACTS/logs/run_everything.txt"
jobs_v2/full_e3_judges.sh > "$CUQ_ARTIFACTS/logs/e3_judges.log" 2>&1
echo "[$(date '+%F %T')] E3 judges done" >> "$CUQ_ARTIFACTS/logs/run_everything.txt"
jobs_v2/full_evaluate.sh > "$CUQ_ARTIFACTS/logs/evaluate.log" 2>&1
echo "[$(date '+%F %T')] evaluation done" >> "$CUQ_ARTIFACTS/logs/run_everything.txt"
