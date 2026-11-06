#!/usr/bin/env bash
# CPU evaluation of the Haiku experiment (Llama). Usage: run_haiku_eval.sh <part>
#   jf-control  judge-free JIP: gpt-oss imitation and no imitation under the reduced protocol, and the full probe
#   jf          judge-free JIP imitating Haiku (reduced protocol)
#   swap        JIP + Judge and SAPLMA/UHead + Judge with the chain verdict of Haiku (and of gpt-oss-20b) at inference
set -uo pipefail
cd "$(dirname "$0")/.."
set -a; source <(grep -E '^(CHAINUQ_[A-Z_]+|HF_HUB_OFFLINE)=' .env); set +a
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1  # one BLAS thread per worker process
EA="A-u_reason-u_token+head:saplma+vq@rlogs"
EB="B-u_reason-u_token-u_judge+ptrue+head:saplma+vq@rlogs"
FUSION="S+verdict+ptrue@lrsel#saplma,S+verdict+ptrue@lrsel#uhead"
V() { "$CHAINUQ_PYTHON" -m chainuq.analysis.variants --backbones llama --workers 12 --seeds 2026,2027,2028,2029,2030,2031,2032,2033,2034,2035 "$@"; }
case "$1" in
  jf-control) V --tag haiku_jf_control --variants "$EA#chainuq-pool2k-n10k-gptoss,$EA#chainuq-wo-pcs-n10k-gptoss,$EA#chainuq" ;;
  jf) V --tag haiku_jf --variants "$EA#chainuq-pool2k-n10k-haiku" ;;
  swap)
    for J in haiku gptoss; do
      CHAINUQ_ANALYSIS_JUDGE=$J CHAINUQ_ANALYSIS_SPLITS=test V --tag haiku_swap_$J \
        --variants "$EB#chainuq,$EB#chainuq-pool2k-n10k-gptoss,$FUSION"
    done ;;
  swap-haiku-head)
    for J in haiku gptoss; do
      CHAINUQ_ANALYSIS_JUDGE=$J CHAINUQ_ANALYSIS_SPLITS=test V --tag haiku_swap_${J}_hh --variants "$EB#chainuq-pool2k-n10k-haiku"
    done ;;
esac
