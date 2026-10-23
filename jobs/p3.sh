#!/usr/bin/env bash

#SBATCH --job-name=khop-p3

#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120gb
#SBATCH --time=1-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

if [[ -n "${_P3_SOURCED:-}" ]]; then
    return 0 2>/dev/null || true
fi
_P3_SOURCED=1

is_eval_complete() {
    local eval_dir="$1"

    report_file_valid "${eval_dir}/evaluation_report.json" || report_file_valid "${eval_dir}/combined_evaluation.json"
}

phase3_head_types() {
    local raw="${PHASE3_HEAD_TYPES:-${HEAD_TYPES:-}}"
    local -a head_types
    if [[ -n "${raw}" ]]; then
        IFS=' ' read -ra head_types <<< "${raw}"
    else
        head_types=("${ALL_HEAD_TYPES[@]}")
    fi
    printf '%s\n' "${head_types[@]}"
}

phase3_ood_datasets() {
    local raw
    if [[ "${PHASE3_OOD_DATASETS+x}" == "x" ]]; then
        raw="${PHASE3_OOD_DATASETS}"
    else
        raw="${OOD_DATASETS:-}"
    fi
    local -a datasets
    IFS=' ' read -ra datasets <<< "${raw}"
    for ds in "${datasets[@]}"; do
        [[ -n "${ds}" ]] || continue
        printf '%s\n' "${ds}"
    done
}

print_eval_summary() {
    local results_base="$1"
    local scope="${2:-ID}"
    local dataset="${3:-}"
    local result_log_path="${4:-}"
    local summary_json_path="${5:-}"
    local calibration_method="${6:-}"
    local -a summary_args=(
        --results_dir "${results_base}"
        --scope "${scope}"
        --best_by "pr_auc"
        --compare_mode "best_non_uq_supervised"
        --log_mode "rewrite"
    )
    if [[ -n "${result_log_path}" ]]; then
        summary_args+=(--results_log "${result_log_path}")
    fi
    if [[ -n "${summary_json_path}" ]]; then
        summary_args+=(--summary_json_path "${summary_json_path}")
    fi
    if [[ -n "${calibration_method}" ]]; then
        summary_args+=(--calibration_method "${calibration_method}")
    fi
    if [[ -n "${dataset}" ]]; then
        summary_args+=(--dataset "${dataset}")
    fi
    if [[ -n "${SUMMARY_HEAD_UNIVERSE:-}" ]]; then
        summary_args+=(--head_universe "${SUMMARY_HEAD_UNIVERSE}")
    fi
    "${PYTHON_BIN}" "${PROJECT_DIR}/utils/result.py" "${summary_args[@]}"
}

eval_head() {
    local head_type="$1"
    local head_path="$2"
    local cache_dir="$3"
    local output_dir="$4"
    local split="${5:-test}"
    local gpu_idx="${6:-0}"
    local log_path="${7:-}"
    local tune_split="${8:-${THRESHOLD_TUNE_SPLIT:-validation}}"
    local threshold_source_report="${9:-}"
    local fit_thresholds_on_eval_split="${10:-0}"
    local require_tune="${REQUIRE_TUNE_SPLIT:-1}"
    local force_eval="${FORCE_EVAL:-0}"
    local enable_temp_scaling="${ENABLE_TEMP_SCALING:-1}"
    local enable_difficulty_thresholds="${ENABLE_DIFFICULTY_THRESHOLDS:-1}"
    local difficulty_threshold_min_samples="${DIFFICULTY_THRESHOLD_MIN_SAMPLES:-100}"
    local -a require_tune_arg=()
    local -a threshold_source_arg=()
    local -a aux_threshold_args=()
    local -a fit_eval_split_arg=()
    if [[ "${require_tune}" == "1" ]]; then
        require_tune_arg=(--require_tune_split)
    fi
    if [[ -n "${threshold_source_report}" ]]; then
        if [[ ! -f "${threshold_source_report}" ]]; then
            echo "  [ERROR] ${head_type}: threshold source report missing: ${threshold_source_report}"
            return 1
        fi
        threshold_source_arg=(--threshold_source_report "${threshold_source_report}")
    fi
    if [[ "${enable_temp_scaling}" == "1" ]]; then
        aux_threshold_args+=(--enable_temp_scaling)
    fi
    if [[ "${enable_difficulty_thresholds}" == "1" ]]; then
        aux_threshold_args+=(
            --enable_difficulty_thresholds
            --difficulty_threshold_min_samples "${difficulty_threshold_min_samples}"
        )
    fi
    if [[ "${fit_thresholds_on_eval_split}" == "1" ]]; then
        fit_eval_split_arg=(--fit_thresholds_on_eval_split)
    fi

    if [[ ! -f "${head_path}/head_weights.pth" ]]; then
        echo "  [SKIP] ${head_type}: no trained model at ${head_path}"
        return 0
    fi

    if [[ "${force_eval}" != "1" ]] && is_eval_complete "${output_dir}"; then
        echo "  [SKIP] ${head_type}: already evaluated on ${split}."
        return 0
    fi

    mkdir -p "$(dirname "${log_path}")"
    echo "  [EVAL] ${head_type} on ${split} (GPU ${gpu_idx})..."

    CUDA_VISIBLE_DEVICES="${gpu_idx}" "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/evaluate.py" \
        --head_path "${head_path}" \
        --cache_dir "${cache_dir}" \
        --split "${split}" \
        --threshold "${EVAL_THRESHOLD}" \
        --threshold_tune_split "${tune_split}" \
        "${threshold_source_arg[@]}" \
        "${aux_threshold_args[@]}" \
        "${fit_eval_split_arg[@]}" \
        --no_enable_posthoc \
        --output_dir "${output_dir}" \
        --batch_size "${TRAIN_BATCH_SIZE}" \
        "${require_tune_arg[@]}" \
        2>&1 | tee -a "${log_path}"

    local rc=${PIPESTATUS[0]}
    if [[ ${rc} -eq 0 ]]; then
        echo "  [EVAL] ${head_type} (${split}) complete ✓"
    else
        echo "  [EVAL] ${head_type} (${split}) FAILED ✗"
    fi
    return "${rc}"
}

eval_baselines() {
    local cache_dir="$1"
    local output_dir="$2"
    local split="${3:-test}"
    local log_path="${4:-}"
    local force_eval="${FORCE_EVAL:-0}"

    if [[ "${force_eval}" != "1" ]] && is_eval_complete "${output_dir}"; then
        echo "  [SKIP] Baselines: already evaluated on ${split}."
        return 0
    fi

    mkdir -p "$(dirname "${log_path}")"
    print_step "Baselines — ${split}"

    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/evaluate.py" \
        --cache_dir "${cache_dir}" \
        --split "${split}" \
        --threshold "${EVAL_THRESHOLD}" \
        --eval_baselines \
        --output_dir "${output_dir}" \
        2>&1 | tee -a "${log_path}"

    local rc=${PIPESTATUS[0]}
    return "${rc}"
}

eval_id_heads() {
    local cache_dir="$1"
    local results_base="$2"
    local eval_base="${3:-$(phase3_id_results_root "${results_base}")}"
    local split="${4:-test}"
    local -a head_types
    mapfile -t head_types < <(phase3_head_types)

    print_step "ID Head Evaluation — ${split}"

    local eval_gpu="${GPU_IDS[0]:-0}"
    echo "  Sequential mode: evaluating ${#head_types[@]} head(s) one at a time on GPU ${eval_gpu}"

    for ht in "${head_types[@]}"; do
        local head_path="$(phase2_head_dir "${results_base}" "${ht}")/final_model"
        local out_dir="${eval_base}/${ht}"
        mkdir -p "${out_dir}"
        eval_head "${ht}" "${head_path}" "${cache_dir}" "${out_dir}" "${split}" "${eval_gpu}" "$(phase3_eval_log_path "ID" "" "${split}" "${ht}")"
    done

    return 0
}

eval_ood_dataset() {
    local ood_ds="$1"
    local ood_cache="$2"
    local id_results="$3"
    local ood_eval_base="$4"
    local old_data_log="${_DATA_LOG-}"
    local -a head_types
    mapfile -t head_types < <(phase3_head_types)
    local heads_csv="${head_types[*]}"
    local manifest_path
    manifest_path="$(phase3_manifest_path "${id_results}" "OOD" "${ood_ds}")"

    if [[ "${FORCE_EVAL:-0}" != "1" ]] && completion_manifest_valid "${manifest_path}" "phase3" "OOD" "${ood_ds}" "" "${heads_csv}"; then
        echo "  [SKIP] Phase 3 OOD ${ood_ds} already complete."
        return 0
    fi

    print_step "OOD Evaluation — ${ood_ds}"

    local val_split="${OOD_THRESHOLD_TUNE_SPLIT:-validation}"
    local eval_split="test"
    if [[ "${val_split}" == "${eval_split}" ]]; then
        echo "  [ERROR] OOD tune split (${val_split}) must differ from eval split (${eval_split}) to avoid leakage."
        return 1
    fi
    local required_splits="${val_split},${eval_split}"
    local required_max_samples="${GEN_MAX_OOD_VAL},${GEN_MAX_OOD_TEST}"


    if ! cache_splits_ready "${ood_cache}" "${required_splits}"; then
        echo "  [OOD] ${ood_ds}/${required_splits}: cache not ready. Run p1.sh for OOD first."
        echo "  [OOD] Attempting generation now..."
        if [[ -n "${LOGS_ROOT:-}" ]]; then
            _DATA_LOG="$(phase1_data_log_path "${ood_ds}" "OOD")"
            mkdir -p "$(dirname "${_DATA_LOG}")"
            echo "  [DATA LOG] ${_DATA_LOG}"
        fi
        if type run_generation &>/dev/null; then
            run_generation "${ood_ds}" "${ood_cache}" "${required_splits}" "${required_max_samples}" "${GEN_OOD_BATCH_SIZE}"
            if type run_judge &>/dev/null; then
                run_judge "${ood_cache}" "${required_splits}" "ood-${ood_ds}"
            fi
            if type run_cleanup_pending_claims &>/dev/null; then
                run_cleanup_pending_claims "${ood_cache}" "${required_splits}"
            fi
        else
            echo "  [ERROR] p1.sh not sourced — cannot generate OOD data."
            if [[ -n "${old_data_log}" ]]; then
                _DATA_LOG="${old_data_log}"
            else
                unset _DATA_LOG
            fi
            return 1
        fi
        if [[ -n "${old_data_log}" ]]; then
            _DATA_LOG="${old_data_log}"
        else
            unset _DATA_LOG
        fi
    fi


    for ht in "${head_types[@]}"; do
        local head_path="$(phase2_head_dir "${id_results}" "${ht}")/final_model"
        local val_out_dir
        val_out_dir="$(phase3_ood_head_dir "${id_results}" "${ood_ds}" "${val_split}" "${ht}")"
        local test_out_dir
        test_out_dir="$(phase3_ood_head_dir "${id_results}" "${ood_ds}" "${eval_split}" "${ht}")"
        local seed_report="$(phase3_id_head_dir "${id_results}" "${ht}")/evaluation_report.json"
        local val_report="${val_out_dir}/evaluation_report.json"
        local val_threshold_source=""
        local val_fit_thresholds="${OOD_FIT_THRESHOLDS_ON_VAL:-1}"
        if [[ "${OOD_USE_ID_THRESHOLD_SEED:-0}" == "1" ]]; then
            val_threshold_source="${seed_report}"
        fi
        mkdir -p "${val_out_dir}" "${test_out_dir}"
        eval_head "${ht}" "${head_path}" "${ood_cache}" "${val_out_dir}" "${val_split}" "${GPU_IDS[0]:-0}" "$(phase3_eval_log_path "OOD" "${ood_ds}" "${val_split}" "${ht}")" "${val_split}" "${val_threshold_source}" "${val_fit_thresholds}"
        eval_head "${ht}" "${head_path}" "${ood_cache}" "${test_out_dir}" "${eval_split}" "${GPU_IDS[0]:-0}" "$(phase3_eval_log_path "OOD" "${ood_ds}" "${eval_split}" "${ht}")" "${val_split}" "${val_report}"
    done


    eval_baselines "${ood_cache}" "$(phase3_ood_baselines_dir "${id_results}" "${ood_ds}")" "${eval_split}" "$(phase3_baselines_log_path "OOD" "${ood_ds}")"
    local phase3_root
    phase3_root="$(phase3_results_root "${id_results}")"
    local result_log
    local figure_dir
    local summary_json
    result_log="$(phase3_result_log_path "OOD" "${ood_ds}")"
    figure_dir="$(phase3_figure_dir "OOD" "${ood_ds}")"
    summary_json="$(phase3_summary_json_path "${id_results}" "OOD" "${ood_ds}")"
    print_eval_summary "${phase3_root}" "OOD" "${ood_ds}" "${result_log}" "${summary_json}"
    run_visualization "${phase3_root}" "${figure_dir}" "OOD" "${ood_ds}"
    local -a manifest_files=(
        "${result_log}"
        "${summary_json}"
        "${figure_dir}/$(artifact_safe_name "${ood_ds}").png"
        "${figure_dir}/$(artifact_safe_name "${ood_ds}")_ranking_heatmap.png"
        "$(phase3_ood_baselines_dir "${id_results}" "${ood_ds}")/combined_evaluation.json"
    )
    for ht in "${head_types[@]}"; do
        manifest_files+=(
            "$(phase3_ood_head_dir "${id_results}" "${ood_ds}" "${val_split}" "${ht}")/evaluation_report.json"
            "$(phase3_ood_head_dir "${id_results}" "${ood_ds}" "${eval_split}" "${ht}")/evaluation_report.json"
        )
    done
    write_completion_manifest "${manifest_path}" "phase3" "OOD" "${ood_ds}" "" "${heads_csv}" "${manifest_files[@]}"

    return 0
}

run_phase3() {
    local cache_dir="${1:-${CACHE_DIR}}"
    local results_base="${2:-${RESULTS_ROOT}}"

    local -a ood_datasets
    mapfile -t ood_datasets < <(phase3_ood_datasets)

    local phase3_start
    phase3_start=$(date +%s)

    print_header "Phase 3 — Evaluate (${MODEL_NAME} / ${DATASET_NAME})"
    echo "Model:       ${MODEL_NAME}"
    echo "Dataset:     ${DATASET_NAME}"
    echo "Cache:       ${cache_dir}"
    echo "Results:     ${results_base}"
    echo "OOD:         ${ood_datasets[*]:-none}"
    echo "Batch:       ${TRAIN_BATCH_SIZE}"
    echo "Budget:      eval=${CACHE_EVAL_MEM_BUDGET_GB} GB / val=${CACHE_VAL_MEM_BUDGET_GB} GB"
    echo "Force eval:  ${FORCE_EVAL:-0}"
    echo ""

    local phase3_root
    phase3_root="$(phase3_results_root "${results_base}")"

    local -a head_types
    mapfile -t head_types < <(phase3_head_types)
    local heads_csv="${head_types[*]}"
    local id_manifest
    id_manifest="$(phase3_manifest_path "${results_base}" "ID")"
    if [[ "${FORCE_EVAL:-0}" != "1" ]] && completion_manifest_valid "${id_manifest}" "phase3" "ID" "" "" "${heads_csv}"; then
        echo "  [SKIP] Phase 3 ID already complete."
    else
        eval_id_heads "${cache_dir}" "${results_base}" "$(phase3_id_results_root "${results_base}")" "test"
        eval_baselines "${cache_dir}" "$(phase3_id_baselines_dir "${results_base}")" "test" "$(phase3_baselines_log_path "ID")"
        local id_result_log
        local id_figure_dir
        local id_summary_json
        id_result_log="$(phase3_result_log_path "ID")"
        id_figure_dir="$(phase3_figure_dir "ID")"
        id_summary_json="$(phase3_summary_json_path "${results_base}" "ID")"
        print_eval_summary "${phase3_root}" "ID" "" "${id_result_log}" "${id_summary_json}"
        run_visualization "${phase3_root}" "${id_figure_dir}" "ID"
        local -a id_manifest_files=(
            "${id_result_log}"
            "${id_summary_json}"
            "${id_figure_dir}/$(artifact_safe_name "${DATASET_NAME}").png"
            "${id_figure_dir}/$(artifact_safe_name "${DATASET_NAME}")_ranking_heatmap.png"
            "${id_figure_dir}/global_pr_auc_ranking_heatmap.png"
            "$(phase3_id_baselines_dir "${results_base}")/combined_evaluation.json"
        )
        local ht
        for ht in "${head_types[@]}"; do
            id_manifest_files+=("$(phase3_id_head_dir "${results_base}" "${ht}")/evaluation_report.json")
        done
        write_completion_manifest "${id_manifest}" "phase3" "ID" "" "" "${heads_csv}" "${id_manifest_files[@]}"
    fi


    for ood_ds in "${ood_datasets[@]}"; do
        [[ -n "${ood_ds}" ]] || continue

        local ood_cache
        ood_cache="$(cache_dir_for_dataset "${ood_ds}")"

        local ood_eval
        ood_eval="$(phase3_ood_split_root "${results_base}" "${ood_ds}" "test")"
        mkdir -p "${ood_eval}"

        eval_ood_dataset "${ood_ds}" "${ood_cache}" "${results_base}" "${ood_eval}"
    done

    local phase3_end
    phase3_end=$(date +%s)
    echo ""
    echo "Phase 3 complete — $(format_duration $((phase3_end - phase3_start)))"

    return 0
}

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail

    SCRIPT_DIR="${SCRIPT_DIR:-/jobs}"

    source "${SCRIPT_DIR}/common.sh"


    if [[ -n "${HEAD_TYPES:-}" ]]; then
        PHASE3_HEAD_TYPES="${HEAD_TYPES}"
    fi



    source "${SCRIPT_DIR}/p1.sh" 2>/dev/null || true

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

    setup_environment
    detect_gpus


    mkdir -p "${LOGS_ROOT}"
    mkdir -p "$(dirname "$(phase_run_log_path "phase3")")"
    exec > >(tee "$(phase_run_log_path "phase3")") 2>&1

    print_header "Phase 3 — Evaluate (${MODEL_NAME} / ${DATASET_NAME})"
    echo "Cache:   ${CACHE_DIR}"
    echo "Budget:  eval=${CACHE_EVAL_MEM_BUDGET_GB} GB / val=${CACHE_VAL_MEM_BUDGET_GB} GB"
    echo ""
    log_gpu_snapshot "p3-start"

    run_phase3 "${CACHE_DIR}" "${RESULTS_ROOT}"
    RC=$?

    log_gpu_snapshot "p3-end"
    exit ${RC}
fi
