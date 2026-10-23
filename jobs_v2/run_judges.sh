#!/usr/bin/env bash
# Main-judge annotations for one backbone (server must be running on $PORT).
# Usage: jobs_v2/run_judges.sh <backbone> [port] [judge]
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
BB=$1; PORT=${2:-8100}; JUDGE=${3:-gptoss}
run() { $CUQ_PY pipeline/judge.py --judge "$JUDGE" --backbone "$BB" --port "$PORT" "$@" 2>&1 | grep '^{'; }
# Response-level labels: every split of the judge-labelled datasets.
run --task correct --dataset hotpotqa --splits train,validation,test
for ds in musique babi; do run --task correct --dataset $ds --splits validation,test; done
# Answer-blind step verdicts, the steps-only control, and Judge-Final: validation + test only.
for ds in hotpotqa musique strategyqa babi gsm8k math500; do
  run --task steps --dataset $ds --splits validation,test
  run --task steps_noconcl --dataset $ds --splits validation,test
  run --task final_blind --dataset $ds --splits validation,test
done
