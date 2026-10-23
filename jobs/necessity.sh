#!/usr/bin/env bash

#SBATCH --job-name=khop-nec

#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120gb
#SBATCH --time=04:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-/jobs}"
source "${SCRIPT_DIR}/common.sh"

MAX_SAMPLES="${MAX_SAMPLES:-500}"
SPLIT="${SPLIT:-validation}"
NEC_BATCH_SIZE="${NEC_BATCH_SIZE:-16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
NEC_MAX_MODEL_LEN="${NEC_MAX_MODEL_LEN:-8192}"
NEC_BACKEND="${NEC_BACKEND:-${BACKEND:-vllm}}"
USE_JUDGE="${USE_JUDGE:-1}"

NECESSITY_DATASETS="${NECESSITY_DATASETS:-hotpot_qa MuSiQue}"

OUTPUT_BASE="${OUTPUT_BASE:-${RESULTS_ROOT}/necessity}"

setup_environment
detect_gpus

mkdir -p "${LOGS_ROOT}"
exec > >(tee "${LOGS_ROOT}/necessity.log") 2>&1

print_header "Necessity Experiments"
echo "Base LLM:    ${MODEL_NAME}"
echo "Judge LLM:   ${JUDGE_MODEL_NAME}"
echo "Datasets:    ${NECESSITY_DATASETS}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Split:       ${SPLIT}"
echo "Use judge:   ${USE_JUDGE}"
echo "Backend:     ${NEC_BACKEND}"
echo "Max ctx len: ${NEC_MAX_MODEL_LEN}"
echo "Output:      ${OUTPUT_BASE}"
echo "Log:         level=${LOG_LEVEL} banner=${LOG_BANNER_WIDTH}"
echo ""

log_gpu_snapshot "necessity-start"

if [[ ! -d "${MODELS_ROOT}/${MODEL_NAME}" ]]; then
    echo "[ERROR] Base model not found: ${MODELS_ROOT}/${MODEL_NAME}"
    exit 1
fi

if [[ "${USE_JUDGE}" == "1" && ! -d "${MODELS_ROOT}/${JUDGE_MODEL_NAME}" ]]; then
    echo "[ERROR] Judge model not found: ${MODELS_ROOT}/${JUDGE_MODEL_NAME}"
    exit 1
fi

FAIL_COUNT=0
PIPELINE_START=$(date +%s)

IFS=' ' read -ra DS_ARRAY <<< "${NECESSITY_DATASETS}"
for DS in "${DS_ARRAY[@]}"; do
    [[ -n "${DS}" ]] || continue

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Dataset: ${DS} | Model: ${MODEL_NAME}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    OUTPUT_DIR="${OUTPUT_BASE}/${DS}/${MODEL_NAME}"
    mkdir -p "${OUTPUT_DIR}"


    nec_args=(
        --dataset "${DS}"
        --model_name "${MODEL_NAME}"
        --max_samples "${MAX_SAMPLES}"
        --split "${SPLIT}"
        --batch_size "${NEC_BATCH_SIZE}"
        --max_new_tokens "${MAX_NEW_TOKENS}"
        --backend "${NEC_BACKEND}"
        --max_model_len "${NEC_MAX_MODEL_LEN}"
        --output_dir "${OUTPUT_DIR}"
    )

    if [[ "${USE_JUDGE}" == "1" ]]; then
        nec_args+=(--use_judge --judge_model_name "${JUDGE_MODEL_NAME}")
    fi

    echo "[CMD] ${PYTHON_BIN} ${PROJECT_DIR}/scripts/necessity_eval.py ${nec_args[*]}"
    echo ""

    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/necessity_eval.py" "${nec_args[@]}" \
        || { echo "[WARN] Necessity experiment failed for ${DS}"; FAIL_COUNT=$((FAIL_COUNT + 1)); continue; }


    if [[ -f "${OUTPUT_DIR}/necessity_summary.json" ]]; then
        echo ""
        echo "──── Summary for ${DS} ────"
        ${PYTHON_BIN} -c "
import json, sys
with open('${OUTPUT_DIR}/necessity_summary.json') as f:
    s = json.load(f)
print(f'  Exp1 — Direct acc:     {s[\"exp1_accuracy\"][\"direct_accuracy\"]:.4f}')
print(f'  Exp1 — Structured acc: {s[\"exp1_accuracy\"][\"structured_accuracy\"]:.4f}')
print(f'  Exp1 — Delta:          {s[\"exp1_accuracy\"][\"delta\"]:+.4f}')
e2 = s.get('exp2_error_localization', {})
if not e2.get('skipped'):
    print(f'  Exp2 — Localization:   {e2.get(\"localization_rate\", 0):.1%} of {e2.get(\"n_wrong\", 0)} errors')
e3 = s.get('exp3_selective_answering', {})
claim_80 = e3.get('claim_level_accuracy', {}).get('0.8', 'N/A')
sample_80 = e3.get('sample_level_accuracy', {}).get('0.8', 'N/A')
print(f'  Exp3 — Acc@80% cov:    claim={claim_80}, sample={sample_80}')
" 2>/dev/null || true
        echo ""
    fi
done

log_gpu_snapshot "necessity-end"

PIPELINE_END=$(date +%s)
ELAPSED=$((PIPELINE_END - PIPELINE_START))
MINS=$((ELAPSED / 60))
SECS=$((ELAPSED % 60))

echo ""
print_header "Necessity Complete (${MINS}m ${SECS}s)"
echo "Results: ${OUTPUT_BASE}"

if [[ ${FAIL_COUNT} -gt 0 ]]; then
    echo "[WARN] ${FAIL_COUNT} dataset(s) had failures."
    exit 1
fi
