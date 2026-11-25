#!/usr/bin/env bash
# Fresh held-out evaluation set (test2): generate, judge, extract features, predict with the already
# trained models and score baselines, for every backbone. No training, no method changes.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
DS=hotpotqa,musique,babi,gsm8k,math500
LOG=/projects/bduw/kytrio/chainuq_overflow/logs/test2; mkdir -p $LOG
stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/timeline.txt"; }
on2() { env CUDA_VISIBLE_DEVICES=$GEN_GPU "$CUQ_PY" "$@"; }
curl -sf localhost:8100/v1/models > /dev/null || { jobs_v2/serve_model.sh gpt-oss-20b 8100 "$JUDGE_GPU" 0.85 & }
until curl -sf localhost:8100/v1/models > /dev/null; do sleep 15; done
judge_bb() {
  local bb=$1
  for ds in hotpotqa musique babi; do "$CUQ_PY" pipeline/judge.py --judge gptoss --task correct --backbone $bb --dataset $ds --splits test2; done
  for ds in ${DS//,/ }; do for t in steps steps_noconcl final_blind; do
    "$CUQ_PY" pipeline/judge.py --judge gptoss --task $t --backbone $bb --dataset $ds --splits test2
  done; done
}
JPIDS=()
for bb in llama qwen8b ministral; do
  stamp "$bb test2 gen start"
  on2 pipeline/gen.py --backbone $bb --dataset $DS --splits test2 > $LOG/gen_$bb.log 2>&1
  on2 pipeline/gen.py --backbone $bb --dataset $DS --splits test2 --max_samples 1000 --n_samples 10 --temperature 1.0 > $LOG/gen_samples_$bb.log 2>&1
  on2 pipeline/cotuq.py --backbone $bb --datasets $DS --splits test2 > $LOG/cotuq_$bb.log 2>&1
  ( judge_bb $bb > $LOG/judge_$bb.log 2>&1; stamp "$bb test2 judge done" ) &
  JPIDS+=($!)
  for ds in ${DS//,/ }; do on2 pipeline/features.py --backbone $bb --dataset $ds --splits test2; done > $LOG/feat_$bb.log 2>&1
  stamp "$bb test2 features done"
done
for p in "${JPIDS[@]}"; do wait $p; done
for bb in llama qwen8b ministral; do
  on2 pipeline/train.py --backbone $bb --seeds all --eval_datasets $DS --eval_fmts step \
     --methods chainuq,chainuq-lb,chainuq-cs,chainuq-cf,chainuq-fr,chainuq2bd,uhead,saplma,factoscope,lookback,meanpool-mlp,attnpool-mlp > $LOG/predict_$bb.log 2>&1
  on2 pipeline/scores.py --backbone $bb --datasets $DS --splits test2 \
     --methods mcp,ppl,mte,ccp,judge_final,judge_step,rm,se,selfcheck,cotuq > $LOG/scores_$bb.log 2>&1
  stamp "$bb test2 predictions + scores done"
done
stamp "test2 all done"
