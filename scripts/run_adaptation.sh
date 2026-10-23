#!/usr/bin/env bash
# Label-free target adaptation: unlabeled target pools (2,000 train questions per OOD dataset; all 1,443 for
# StrategyQA), their answer-blind judge labels and features, the adapted heads (10 seeds), controls (5 seeds),
# verifier strength (weak: 5 seeds, mid: 3 seeds), and pool size (Llama, 5 seeds).
# Run after scripts/run_main.sh; needs `python -m chainuq.pipeline.prepare_data --pools 2000,6000`.
set -euo pipefail
source "$(dirname "$0")/common.sh"
LOG="$CHAINUQ_LOGS/adaptation"; mkdir -p "$LOG"
POOL6K="${CHAINUQ_POOL6K_ARTIFACTS:-$CHAINUQ_ARTIFACTS/pool_n6000}"  # 6,000-question pools: separate artifact tree
pool_root() { case $1 in gsm8k|math500) echo "$CHAINUQ_DATASETS/pools/n$2" ;; *) echo "" ;; esac; }
build_pool() {  # <backbone> <n> : generate, judge (needs the server on $PORT), extract features for every OOD set
  local bb=$1 n=$2 ds
  for ds in $3; do
    run $GEN_GPU pipeline.generate --backbone $bb --dataset $ds --splits train --max_samples $n \
        --gpu_memory_utilization 0.4 --data_root "$(pool_root $ds $n)"
    for task in steps final_blind; do
      "$PY" -m chainuq.pipeline.judge --judge ${JUDGE:-gptoss} --task $task --backbone $bb --dataset $ds --splits train --port $PORT
    done
    run $GEN_GPU pipeline.features --backbone $bb --dataset $ds --splits train
  done
}
adapt() {  # <backbone> <run> [adapt options]: adapt, then score on the given splits
  local bb=$1 runname=$2; shift 2
  run $GEN_GPU experiments.adapt --backbone $bb --run $runname "$@"
}
score() {  # <backbone> <run> <tag> <splits> [seeds]
  for ds in $OOD; do
    cpu experiments.adapt_eval --backbone $1 --dataset $ds --adapted $2 --tag "$3" --splits $4 --seeds ${5:-2026,2027,2028,2029,2030} &
  done
  wait
}

PORT=$JUDGE_PORT
serve gpt-oss-20b $PORT $JUDGE_GPU 0.45
for bb in $BACKBONES; do build_pool $bb 2000 "$OOD"; done > "$LOG/pools.log" 2>&1
stop_server
TEN=2026,2027,2028,2029,2030,2031,2032,2033,2034,2035
for bb in $BACKBONES; do
  adapt $bb chainuq-ta --seeds $TEN && score $bb chainuq-ta "" test,test2 $TEN
  adapt $bb chainuq-ta-src --lam 0 && score $bb chainuq-ta-src _src test
  adapt $bb chainuq-ta-pseudo --target_mode pseudo && score $bb chainuq-ta-pseudo _pseudo test
done > "$LOG/adapt.log" 2>&1
for bb in llama qwen8b; do adapt $bb chainuq-ta-steps --target_mode steps && score $bb chainuq-ta-steps _steps test; done >> "$LOG/adapt.log" 2>&1
adapt llama chainuq-ta-both --target_mode both >> "$LOG/adapt.log" 2>&1 && score llama chainuq-ta-both _both test,test2 >> "$LOG/adapt.log" 2>&1

# Verifier strength: pool labels from a mid-strength (Qwen3-4B, 3 seeds) and a weak (Qwen3-1.7B, 5 seeds) verifier.
for spec in qwen4b:Qwen3-4B-Instruct-2507:mid:2026,2027,2028 qwen1b:Qwen3-1.7B:weak:2026,2027,2028,2029,2030; do
  IFS=: read -r J MODEL NAME SEEDS <<< "$spec"
  PORT=$((JUDGE_PORT + 2)); serve $MODEL $PORT $JUDGE_GPU 0.5
  for bb in $BACKBONES; do for ds in $OOD; do for task in steps final_blind; do
    "$PY" -m chainuq.pipeline.judge --judge $J --task $task --backbone $bb --dataset $ds --splits train --port $PORT
  done; done; done > "$LOG/pool_labels_$NAME.log" 2>&1
  stop_server
  for bb in $BACKBONES; do
    adapt $bb chainuq-ta-$NAME --pool_judge $J --seeds $SEEDS && score $bb chainuq-ta-$NAME _$NAME test $SEEDS
    adapt $bb chainuq-ta-$NAME-pseudo --pool_judge $J --target_mode pseudo --seeds $SEEDS \
        && score $bb chainuq-ta-$NAME-pseudo _${NAME}_pseudo test $SEEDS
  done >> "$LOG/adapt.log" 2>&1
done

# Pool size (Llama): fixed random subsets of the 2,000-question pools, and 6,000-question pools in their own tree.
for n in 250 500 1000; do adapt llama chainuq-ta-n$n --pool_n $n && score llama chainuq-ta-n$n _n$n test; done >> "$LOG/adapt.log" 2>&1
PORT=$JUDGE_PORT; serve gpt-oss-20b $PORT $JUDGE_GPU 0.45
( export CHAINUQ_ARTIFACTS="$POOL6K"; build_pool llama 6000 "musique babi gsm8k math500" ) > "$LOG/pool6k.log" 2>&1
stop_server
adapt llama chainuq-ta-n6000 --pool_root "$POOL6K" --datasets musique,babi,gsm8k,math500 \
    >> "$LOG/adapt.log" 2>&1
for ds in musique babi gsm8k math500; do cpu experiments.adapt_eval --backbone llama --dataset $ds --adapted chainuq-ta-n6000 --tag _n6000 & done
wait
stamp "adaptation done"
