#!/usr/bin/env bash
# Rebuild one backbone's MATH validation cell after the validation split moved from MATH train
# (contaminated) to MATH test minus MATH-500. Reuses the trained checkpoints (predict-only).
# Uses $GEN_GPU for generation/features; starts the gpt-oss judge on $JUDGE_GPU:8100 if none is running.
# Usage: jobs_v2/repair_math_val.sh <backbone>
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
BB=$1
LOG="$CUQ_ARTIFACTS/logs/full_${BB}"
on() { env CUDA_VISIBLE_DEVICES=$GEN_GPU "$CUQ_PY" "$@"; }
stamp() { echo "[$(date '+%F %T')] $BB $*" | tee -a "$LOG/timeline.txt"; }
stamp "math-val repair start"
# /work/nvme project quota is exhausted (Lustre grants): keep the rebuilt feature cell on /projects.
OVERFLOW=${CUQ_OVERFLOW:-/projects/bduw/kytrio/chainuq_overflow}
rm -rf "$CUQ_ARTIFACTS/feat/$BB/math500/step/validation" "$OVERFLOW/feat/$BB/math500/step/validation"
mkdir -p "$OVERFLOW/feat/$BB/math500/step/validation"
ln -s "$OVERFLOW/feat/$BB/math500/step/validation" "$CUQ_ARTIFACTS/feat/$BB/math500/step/validation"
on pipeline/gen.py --backbone $BB --dataset math500 --splits validation --overwrite
on pipeline/gen.py --backbone $BB --dataset math500 --splits validation --max_samples 1000 --n_samples 10 --temperature 1.0 --overwrite
on pipeline/cotuq.py --backbone $BB --datasets math500 --splits validation
STARTED=0
if ! curl -sf localhost:8100/v1/models > /dev/null; then
  jobs_v2/serve_model.sh gpt-oss-20b 8100 "$JUDGE_GPU" 0.85 &
  SERVER_PID=$!; STARTED=1
  until curl -sf localhost:8100/v1/models > /dev/null; do sleep 10; done
fi
for t in steps steps_noconcl final_blind; do
  "$CUQ_PY" pipeline/judge.py --judge gptoss --task $t --backbone $BB --dataset math500 --splits validation --overwrite
done
[ "$STARTED" = 1 ] && { kill $SERVER_PID; wait $SERVER_PID 2>/dev/null; }
on pipeline/features.py --backbone $BB --dataset math500 --splits validation
on pipeline/nli_steps.py --backbone $BB --datasets math500 --splits validation
on pipeline/train.py --backbone $BB --predict_only --seeds all --eval_datasets math500 --eval_fmts step \
   --methods uhead,saplma,factoscope,lookback,meanpool-mlp,attnpool-mlp,chainuq,chainuq-lb,chainuq-cs,chainuq-cf,chainuq-fr,chainuq2bd
on pipeline/scores.py --backbone $BB --datasets math500 --overwrite --methods mcp,ppl,mte,ccp,judge_final,judge_step,rm,se,selfcheck,cotuq
stamp "math-val repair done"
