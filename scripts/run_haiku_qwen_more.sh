#!/usr/bin/env bash
# Extra Qwen3 seeds on four GPUs (gpt-oss imitation 2029, no imitation 2029, Haiku imitation 2030-2031), then re-evaluate.
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source <(grep -E '^(CHAINUQ_[A-Z_]+|HF_HUB_OFFLINE)=' .env); set +a
LOG="$CHAINUQ_LOGS/haiku"
BB=qwen8b scripts/run_haiku_train.sh 0 gptoss 2029 chainuq-pool2k > "$LOG/qwen_more_0.log" 2>&1 &
BB=qwen8b scripts/run_haiku_train.sh 1 gptoss 2029,2030 chainuq-wo-pcs > "$LOG/qwen_more_1.log" 2>&1 &
BB=qwen8b scripts/run_haiku_train.sh 2 haiku 2030 > "$LOG/qwen_more_2.log" 2>&1 &
BB=qwen8b scripts/run_haiku_train.sh 3 haiku 2031 > "$LOG/qwen_more_3.log" 2>&1 &
wait
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
EA="A-u_reason-u_token+head:saplma+vq@rlogs"
"$CHAINUQ_PYTHON" -m chainuq.analysis.variants --backbones qwen8b --workers 12 --seeds 2026,2027,2028,2029,2030,2031 --tag haiku_jf_qwen6 \
  --variants "$EA#chainuq-pool2k-n10k-haiku,$EA#chainuq-pool2k-n10k-gptoss,$EA#chainuq-wo-pcs-n10k-gptoss" > "$LOG/qwen_eval6.log" 2>&1
echo QWEN_MORE_DONE >> "$LOG/qwen_eval6.log"
