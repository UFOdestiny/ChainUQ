#!/usr/bin/env bash

#SBATCH --job-name=khop-p2

#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120gb
#SBATCH --time=1-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

if [[ -n "${_P2_SOURCED:-}" ]]; then
    return 0 2>/dev/null || true
fi
_P2_SOURCED=1

is_head_complete() {
    local results_root="$1"
    local head_type="$2"
    local weights="${results_root}/${head_type}/final_model/head_weights.pth"
    [[ -f "${weights}" ]]
}

phase2_head_types() {
    local raw="${PHASE2_HEAD_TYPES:-${HEAD_TYPES:-}}"
    local -a head_types
    if [[ -n "${raw}" ]]; then
        IFS=' ' read -ra head_types <<< "${raw}"
    else
        head_types=("${ALL_HEAD_TYPES[@]}")
    fi
    printf '%s\n' "${head_types[@]}"
}

cleanup_retrain_artifacts() {
    local results_root="${1:-${RESULTS_ROOT}}"
    local logs_root="${2:-${LOGS_ROOT}}"
    local -a head_types
    local -a ood_datasets
    mapfile -t head_types < <(phase2_head_types)
    IFS=' ' read -ra ood_datasets <<< "${OOD_DATASETS:-}"

    print_step "RETRAIN Cleanup — ${DATASET_NAME}"
    echo "Results: ${results_root}"
    echo "Logs:    ${logs_root}"
    echo "Heads:   ${head_types[*]}"
    echo "OOD:     ${ood_datasets[*]:-none}"
    echo ""

    for ht in "${head_types[@]}"; do
        [[ -n "${ht}" ]] || continue
        remove_path_if_exists "$(phase2_head_dir "${results_root}" "${ht}")"
        remove_path_if_exists "$(phase3_id_head_dir "${results_root}" "${ht}")"
        remove_path_if_exists "$(phase2_train_log_path "${ht}")"
        remove_path_if_exists "$(phase3_eval_log_path "ID" "" "test" "${ht}")"

        for ood_ds in "${ood_datasets[@]}"; do
            [[ -n "${ood_ds}" ]] || continue
            remove_path_if_exists "$(phase3_ood_head_dir "${results_root}" "${ood_ds}" "validation" "${ht}")"
            remove_path_if_exists "$(phase3_ood_head_dir "${results_root}" "${ood_ds}" "test" "${ht}")"
            remove_path_if_exists "$(phase3_eval_log_path "OOD" "${ood_ds}" "validation" "${ht}")"
            remove_path_if_exists "$(phase3_eval_log_path "OOD" "${ood_ds}" "test" "${ht}")"
            remove_path_if_exists "$(phase3_manifest_path "${results_root}" "OOD" "${ood_ds}")"
        done
    done

    remove_path_if_exists "$(phase2_train_root "${results_root}")"
    remove_path_if_exists "$(phase3_results_root "${results_root}")"
    remove_path_if_exists "$(phase4_root "${results_root}")"
    remove_path_if_exists "$(results_summary_root "${results_root}")"
    remove_path_if_exists "$(phase_logs_root "phase1")"
    remove_path_if_exists "$(phase_logs_root "phase2")"
    remove_path_if_exists "$(phase_logs_root "phase3")"
    remove_path_if_exists "$(phase_logs_root "phase4")"
    remove_path_if_exists "$(logs_summary_root)"
    remove_path_if_exists "$(logs_figure_root)"
    remove_path_if_exists "$(phase_run_log_path "phase1")"
    remove_path_if_exists "$(phase_run_log_path "phase2")"
    remove_path_if_exists "$(phase_run_log_path "phase3")"
    remove_path_if_exists "$(phase_run_log_path "phase4")"
}

train_head() {
    local head_type="$1"
    local cache_dir="$2"
    local output_base="$3"
    local gpu_idx="${4:-0}"
    local output_dir="${output_base}/${head_type}"
    local batch_size="${TRAIN_BATCH_SIZE}"
    local min_batch_size="${TRAIN_MIN_BATCH_SIZE:-256}"

    while true; do
        echo "  [TRAIN] Starting ${head_type} on GPU ${gpu_idx} (batch_size=${batch_size})..."

        local -a resume_args=()
        if [[ "${RETRAIN:-0}" != "1" ]] && compgen -G "${output_dir}/checkpoint-*" > /dev/null; then
            echo "  [TRAIN] Found existing checkpoints for ${head_type}; resuming from latest."
            resume_args=(--resume_from_checkpoint auto)
        fi

        local train_log=""
        if [[ -n "${LOGS_ROOT:-}" ]]; then
            train_log="$(phase2_train_log_path "${head_type}")"
            mkdir -p "$(dirname "${train_log}")"
        fi

        if [[ -n "${train_log}" ]]; then
            CUDA_VISIBLE_DEVICES="${gpu_idx}" "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/train.py" \
                --head_type "${head_type}" \
                --cache_dir "${cache_dir}" \
                --output_dir "${output_dir}" \
                --num_epochs "${TRAIN_EPOCHS}" \
                --batch_size "${batch_size}" \
                --learning_rate "${TRAIN_LEARNING_RATE}" \
                --loss_type "${LOSS_TYPE}" \
                --loss_pos_weight "${LOSS_POS_WEIGHT}" \
                --focal_gamma "${FOCAL_GAMMA}" \
                --sample_pos_weight "${SAMPLE_POS_WEIGHT:-${LOSS_POS_WEIGHT}}" \
                "${resume_args[@]}" \
                --report_to "${REPORT_TO}" \
                2>&1 | tee -a "${train_log}"
            local rc=${PIPESTATUS[0]}
        else
            CUDA_VISIBLE_DEVICES="${gpu_idx}" "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/train.py" \
                --head_type "${head_type}" \
                --cache_dir "${cache_dir}" \
                --output_dir "${output_dir}" \
                --num_epochs "${TRAIN_EPOCHS}" \
                --batch_size "${batch_size}" \
                --learning_rate "${TRAIN_LEARNING_RATE}" \
                --loss_type "${LOSS_TYPE}" \
                --loss_pos_weight "${LOSS_POS_WEIGHT}" \
                --focal_gamma "${FOCAL_GAMMA}" \
                --sample_pos_weight "${SAMPLE_POS_WEIGHT:-${LOSS_POS_WEIGHT}}" \
                "${resume_args[@]}" \
                --report_to "${REPORT_TO}"
            local rc=$?
        fi
        if [[ ${rc} -eq 0 ]]; then
            echo "  [TRAIN] ${head_type} complete ✓"
            return 0
        fi

        if [[ ${rc} -eq 137 && ${batch_size} -gt ${min_batch_size} ]]; then
            local next_batch_size=$(( batch_size / 2 ))
            if [[ ${next_batch_size} -lt ${min_batch_size} ]]; then
                next_batch_size="${min_batch_size}"
            fi
            if [[ ${next_batch_size} -lt ${batch_size} ]]; then
                echo "  [TRAIN] ${head_type} was killed (exit ${rc}); retrying with smaller batch_size=${next_batch_size}..."
                batch_size="${next_batch_size}"
                continue
            fi
        fi

        echo "  [TRAIN] ${head_type} FAILED (exit ${rc}) ✗"
        return ${rc}
    done
}

run_phase2() {
    local cache_dir="${1:-${CACHE_DIR}}"
    local results_base="${2:-$(phase2_train_root "${RESULTS_ROOT}")}"
    local -a head_types
    mapfile -t head_types < <(phase2_head_types)

    local phase2_start
    phase2_start=$(date +%s)

    print_header "Phase 2 — Train Supervised Heads"
    echo "Model:   ${MODEL_NAME}"
    echo "Dataset: ${DATASET_NAME}"
    echo "Cache:   ${cache_dir}"
    echo "Output:  ${results_base}"
    echo "Heads:   ${head_types[*]}"
    echo "Epochs:  ${TRAIN_EPOCHS}"
    echo "Batch:   ${TRAIN_BATCH_SIZE}"
    echo "Budget:  train=${CACHE_TRAIN_MEM_BUDGET_GB} GB / val=${CACHE_VAL_MEM_BUDGET_GB} GB"
    echo ""

    mkdir -p "${results_base}" "$(phase_logs_root "phase2")/train"


    local -a heads_to_train=()
    for ht in "${head_types[@]}"; do
        if [[ "${RETRAIN:-0}" == "1" ]]; then
            heads_to_train+=("${ht}")
        elif is_head_complete "${results_base}" "${ht}"; then
            echo "  [SKIP] ${ht} already trained."
        else
            heads_to_train+=("${ht}")
        fi
    done

    if [[ ${#heads_to_train[@]} -eq 0 ]]; then
        echo "All heads already trained — nothing to do."
        return 0
    fi

    echo "Heads to train: ${heads_to_train[*]} (${#heads_to_train[@]} total)"
    echo ""

    local train_gpu="${GPU_IDS[0]:-0}"
    echo "Sequential mode: training ${#heads_to_train[@]} head(s) one at a time on GPU ${train_gpu}"
    echo ""
    for ht in "${heads_to_train[@]}"; do
        train_head "${ht}" "${cache_dir}" "${results_base}" "${train_gpu}"
    done

    local phase2_end
    phase2_end=$(date +%s)
    echo ""
    echo "Phase 2 complete — $(format_duration $((phase2_end - phase2_start)))"

    return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail

    SCRIPT_DIR="${SCRIPT_DIR:-/jobs}"

    source "${SCRIPT_DIR}/common.sh"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --retrain)
                RETRAIN=1
                shift
                ;;
            *)
                echo "[ERROR] Unknown argument: $1"
                exit 2
                ;;
        esac
    done


    if [[ -n "${HEAD_TYPES:-}" ]]; then
        PHASE2_HEAD_TYPES="${HEAD_TYPES}"
    fi

    setup_environment
    detect_gpus


    mkdir -p "${LOGS_ROOT}"
    mkdir -p "$(dirname "$(phase_run_log_path "phase2")")"
    exec > >(tee "$(phase_run_log_path "phase2")") 2>&1

    print_header "Phase 2 — Train (${MODEL_NAME} / ${DATASET_NAME})"
    echo "Cache:   ${CACHE_DIR}"
    echo "Budget:  train=${CACHE_TRAIN_MEM_BUDGET_GB} GB / val=${CACHE_VAL_MEM_BUDGET_GB} GB"
    echo ""

    log_gpu_snapshot "p2-start"

    if [[ "${RETRAIN:-0}" == "1" ]]; then
        cleanup_retrain_artifacts "${RESULTS_ROOT}" "${LOGS_ROOT}"
    fi

    run_phase2 "${CACHE_DIR}" "$(phase2_train_root "${RESULTS_ROOT}")"
    RC=$?

    log_gpu_snapshot "p2-end"
    exit ${RC}
fi
