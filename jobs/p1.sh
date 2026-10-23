#!/usr/bin/env bash

#SBATCH --job-name=khop-p1

#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200gb
#SBATCH --time=3-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

if [[ -n "${_P1_SOURCED:-}" ]]; then
    return 0 2>/dev/null || true
fi
_P1_SOURCED=1

_run_with_data_log() {
    if [[ -n "${_DATA_LOG:-}" && "${DATA_LOG_TEE:-1}" == "1" ]]; then
        "$@" 2>&1 | tee -a "${_DATA_LOG}"
        return "${PIPESTATUS[0]}"
    else
        "$@"
    fi
}

run_generation() {
    local dataset="$1"
    local cache_dir="$2"
    local splits="${3:-train,validation,test}"
    local max_samples="${4:-${GEN_MAX_TRAIN},${GEN_MAX_VAL},${GEN_MAX_TEST}}"
    local batch_size="${5:-${GEN_BATCH_SIZE}}"

    local model_path="${MODELS_ROOT}/${MODEL_NAME}"

    local orig_gpu_util="${VLLM_GPU_MEMORY_UTILIZATION}"
    local next_gpu_util
    next_gpu_util="$(get_gpu_memory_util "${dataset}")"
    export VLLM_GPU_MEMORY_UTILIZATION="${next_gpu_util}"


    if cache_splits_ready "${cache_dir}" "${splits}"; then
        echo "  [GEN] ${dataset}: all splits ready in cache, skipping."
        export VLLM_GPU_MEMORY_UTILIZATION="${orig_gpu_util}"
        return 0
    fi

    print_step "Phase 1 — Generate: ${MODEL_NAME} / ${dataset} (gpu_mem_util=${VLLM_GPU_MEMORY_UTILIZATION})"

    local gen_args=(
        --dataset "${dataset}"
        --model_path "${model_path}"
        --backend "${BACKEND}"
        --split "${splits}"
        --max_samples "${max_samples}"
        --cache_dir "${cache_dir}"
        --batch_size "${batch_size}"
        --max_new_tokens "${GEN_MAX_NEW_TOKENS}"
    )

    _run_with_data_log "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/generate.py" "${gen_args[@]}"
    local rc=$?
    export VLLM_GPU_MEMORY_UTILIZATION="${orig_gpu_util}"
    return ${rc}
}

run_judge() {
    local cache_dir="$1"
    local splits="${2:-train,validation,test}"
    local context="${3:-judge}"

    if ! should_run_judge_for_splits "${cache_dir}" "${splits}" "${context}"; then
        return 0
    fi

    print_step "Phase 1.5b — Judge: ${context}"

    local judge_args=(
        --cache_dir "${cache_dir}"
        --split "${splits}"
        --judge_model "${MODELS_ROOT}/${JUDGE_MODEL_NAME}"
        --judge_backend "${BACKEND}"
        --max_new_tokens "${CLAIM_LABELER_MAX_NEW_TOKENS}"
        --batch_size "${JUDGE_BATCH_SIZE}"
    )

    if [[ "${FORCE_JUDGE:-0}" == "1" ]]; then
        judge_args+=(--force)
    fi

    _run_with_data_log "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/judge.py" "${judge_args[@]}"
}

run_cleanup_pending_claims() {
    local cache_dir="$1"
    local splits="${2:-train,validation,test}"

    if ! should_run_cleanup_for_splits "${cache_dir}" "${splits}" "phase1"; then
        echo "  [CLEANUP] phase1: all requested splits already cleaned, skipping."
        return 0
    fi

    print_step "Phase 1.5c — Cleanup: Remove samples with pending claims"

    local cleanup_args=(
        --cache_dir "${cache_dir}"
        --split "${splits}"
    )

    _run_with_data_log "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/cleanup_pending_claims.py" "${cleanup_args[@]}"
}

run_phase1() {
    local dataset="${1:-${DATASET_NAME}}"
    local cache_dir="${2:-${CACHE_DIR}}"
    local splits="${3:-train,validation,test}"
    local max_samples="${4:-${GEN_MAX_TRAIN},${GEN_MAX_VAL},${GEN_MAX_TEST}}"
    local batch_size="${5:-${GEN_BATCH_SIZE}}"

    local phase1_start
    phase1_start=$(date +%s)

    print_header "Phase 1 — ${MODEL_NAME} / ${dataset}"

    mkdir -p "${cache_dir}"


    if [[ -n "${LOGS_ROOT:-}" ]]; then
        if [[ "${dataset}" == "${DATASET_NAME}" ]]; then
            _DATA_LOG="$(phase1_data_log_path "${dataset}" "ID")"
        else
            _DATA_LOG="$(phase1_data_log_path "${dataset}" "OOD")"
        fi
        mkdir -p "$(dirname "${_DATA_LOG}")"
        echo "  [DATA LOG] ${_DATA_LOG}"
    fi


    run_generation "${dataset}" "${cache_dir}" "${splits}" "${max_samples}" "${batch_size}"


    run_judge "${cache_dir}" "${splits}" "${dataset}"


    run_cleanup_pending_claims "${cache_dir}" "${splits}"


    print_cache_quality_summary "${cache_dir}" "${splits}" "${dataset}"

    unset _DATA_LOG

    local phase1_end
    phase1_end=$(date +%s)
    echo ""
    echo "Phase 1 (${dataset}) complete — $(format_duration $((phase1_end - phase1_start)))"
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail

    SCRIPT_DIR="${SCRIPT_DIR:-/jobs}"
    CACHE_SUBDIR="${CACHE_SUBDIR:-ALL}"

    source "${SCRIPT_DIR}/common.sh"



    _DEFAULT_SPLITS="${P1_SPLIT:-train,validation,test}"
    _DEFAULT_MAX="${P1_MAX:-0}"
    _DATASET_SPECS=()

    _p1_usage() {
        cat <<'EOF'
Usage: bash jobs/p1.sh [OPTIONS]

Options:
  --dataset SPEC      Dataset to process (repeatable).
                      SPEC format:  NAME[:SPLITS[:MAX]]
                        NAME   — dataset identifier
                        SPLITS — comma-separated splits, e.g. train,validation,test
                        MAX    — max samples per split, 0 = all  (default: 0).
                               Can be a single number (applied to all splits)
                               or a comma-separated list matching SPLITS order,
                               e.g. "0,1000" for val=all, test=1000.
                      Omitted SPLITS / MAX inherit the --split / --max defaults.
  --split  SPLITS     Default splits for all datasets  (default: train,validation,test)
  --max    N          Default max samples per split     (default: 0, i.e. all)
  --cache_subdir DIR  Set CACHE_SUBDIR for cache paths
  --help              Show this help

Examples:

  bash jobs/p1.sh --dataset hotpot_qa


  bash jobs/p1.sh \
    --dataset "hotpot_qa:validation,test:1000" \
    --dataset "MuSiQue:train,validation,test"


  bash jobs/p1.sh --split validation,test --max 500 \
    --dataset hotpot_qa --dataset MuSiQue


  bash jobs/p1.sh --cache_subdir exp1 \
    --dataset "hotpot_qa:train,validation,test:2000"


  bash jobs/p1.sh --dataset "hotpot_qa:validation,test:0,100"
EOF
    }



    _p1_normalize_spec() {
        local spec="$1"
        local colon_count
        colon_count=$(awk -F: '{print NF-1}' <<< "${spec}")
        case "${colon_count}" in
            0) printf '%s' "${spec}:__DEFAULT__:__DEFAULT__" ;;
            1) printf '%s' "${spec}:__DEFAULT__" ;;
            *)
               local name splits max
               IFS=':' read -r name splits max _ <<< "${spec}"
               printf '%s' "${name}:${splits}:${max}" ;;
        esac
    }


    while [[ $# -gt 0 ]]; do
        case "$1" in
            --dataset)
                _DATASET_SPECS+=("$(_p1_normalize_spec "$2")"); shift 2 ;;
            --split)
                _DEFAULT_SPLITS="$2"; shift 2 ;;
            --max)
                _DEFAULT_MAX="$2"; shift 2 ;;
            --cache_subdir)
                export CACHE_SUBDIR="$2"; shift 2 ;;
            --help|-h)
                _p1_usage; exit 0 ;;
            *)
                echo "[ERROR] Unknown argument: $1" >&2
                _p1_usage >&2
                exit 1 ;;
        esac
    done










    _DEFAULT_DATASET_SPECS=(

        "${DATASET_NAME}:train,validation,test:0"


        "hotpot_qa:train,validation,test:0"
        "MuSiQue:train,validation,test:0"
        "2WikiMultihopQA:train,validation,test:0"


        "babi:train,validation,test:0"
        "IIRC:train,validation,test:0"
        "StrategyQA:train,validation,test:0"
    )

    if [[ ${#_DATASET_SPECS[@]} -eq 0 ]]; then
        _DATASET_SPECS=("${_DEFAULT_DATASET_SPECS[@]}")
    fi


    setup_environment
    detect_gpus


    mkdir -p "$(dirname "$(phase_run_log_path "phase1")")"
    exec > >(tee "$(phase_run_log_path "phase1")") 2>&1


    print_header "Phase 1 — Generate + Judge (${MODEL_NAME})"
    echo "Model:        ${MODEL_NAME}"
    echo "CACHE_SUBDIR: ${CACHE_SUBDIR:-<none>}"
    echo "Backend:      ${BACKEND}"
    echo ""
    printf "  %-32s  %-32s  %s\n" "Dataset" "Splits" "Max/split"
    printf "  %-32s  %-32s  %s\n" "-------" "------" "---------"
    for _spec in "${_DATASET_SPECS[@]}"; do
        _ds="${_spec%%:*}"
        _rest="${_spec#*:}"
        _sp="${_rest%%:*}";  [[ "${_sp}" == "__DEFAULT__" ]] && _sp="${_DEFAULT_SPLITS}"
        _mx="${_rest##*:}";  [[ "${_mx}" == "__DEFAULT__" ]] && _mx="${_DEFAULT_MAX}"
        printf "  %-32s  %-32s  %s\n" "${_ds}" "${_sp}" "${_mx}"
    done
    echo ""

    log_gpu_snapshot "p1-start"


    if [[ ! -d "${MODELS_ROOT}/${MODEL_NAME}" ]]; then
        echo "[ERROR] Model not found: ${MODELS_ROOT}/${MODEL_NAME}"
        exit 1
    fi


    for _spec in "${_DATASET_SPECS[@]}"; do
        _ds="${_spec%%:*}"
        _rest="${_spec#*:}"
        _sp="${_rest%%:*}";  [[ "${_sp}" == "__DEFAULT__" ]] && _sp="${_DEFAULT_SPLITS}"
        _mx="${_rest##*:}";  [[ "${_mx}" == "__DEFAULT__" ]] && _mx="${_DEFAULT_MAX}"





        if [[ "${_mx}" == *,* ]]; then
            _max_csv="${_mx}"
        else
            _n_splits=$(awk -F',' '{print NF}' <<< "${_sp}")
            _max_csv=""
            for (( _i = 0; _i < _n_splits; _i++ )); do
                [[ -n "${_max_csv}" ]] && _max_csv+=","
                _max_csv+="${_mx}"
            done
        fi

        _ds_cache="$(cache_dir_for_dataset "${_ds}")"
        mkdir -p "${_ds_cache}"

        run_phase1 "${_ds}" "${_ds_cache}" "${_sp}" "${_max_csv}"
    done

    log_gpu_snapshot "p1-end"
    echo "Phase 1 complete — all datasets processed."
fi
