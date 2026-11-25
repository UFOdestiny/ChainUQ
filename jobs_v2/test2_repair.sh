#!/usr/bin/env bash
# After test2_run.sh: redo test2 feature cells whose shard writes failed (disk quota), then their
# predictions and baseline scores. features.py resumes and drops unreadable shards; train.py only
# writes cells with complete features; scores.py ran on the partial cells, so those are overwritten.
set -uo pipefail
source "$(dirname "$0")/env.sh"
cd "$CUQ_CODE"
DS=hotpotqa,musique,babi,gsm8k,math500
LOG=/projects/bduw/kytrio/chainuq_overflow/logs/test2
stamp() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG/timeline.txt"; }
on2() { env CUDA_VISIBLE_DEVICES=$GEN_GPU "$CUQ_PY" "$@"; }
until grep -q "test2 all done" $LOG/timeline.txt; do sleep 60; done
for bb in llama qwen8b ministral; do
  redo=()
  for ds in ${DS//,/ }; do
    [ -f "$("$CUQ_PY" -c "from cuq.registry import feat_dir; print(feat_dir('$bb','$ds','test2'))")/DONE.json" ] || redo+=($ds)
  done
  [ ${#redo[@]} -eq 0 ] && continue
  R=$(IFS=,; echo "${redo[*]}")
  stamp "$bb test2 repair: $R"
  for ds in ${redo[@]}; do on2 pipeline/features.py --backbone $bb --dataset $ds --splits test2; done > $LOG/feat_repair_$bb.log 2>&1
  on2 pipeline/train.py --backbone $bb --seeds all --eval_datasets $R --eval_fmts step \
     --methods chainuq,chainuq-lb,chainuq-cs,chainuq-cf,chainuq-fr,chainuq2bd,uhead,saplma,factoscope,lookback,meanpool-mlp,attnpool-mlp > $LOG/predict_repair_$bb.log 2>&1
  on2 pipeline/scores.py --backbone $bb --datasets $R --splits test2 --overwrite \
     --methods mcp,ppl,mte,ccp,judge_final,judge_step,rm,se,selfcheck,cotuq > $LOG/scores_repair_$bb.log 2>&1
done
stamp "test2 repair done"
