#!/usr/bin/env bash


set -euo pipefail

PIPELINE_START="${PIPELINE_START:-$(date +%s)}"
SCRIPT_DIR="${SCRIPT_DIR:-/jobs}"

if ! declare -p ALL_HEAD_TYPES >/dev/null 2>&1; then
    ALL_HEAD_TYPES=(
        "uq_v1"
        "uq_v2"
        "uq_v3"
        "uq_abl_v1"
        "uq_abl_v2"
        "uq_abl_v3"
        "uq_abl_v4"

        "luh_head"
        "luh_light"
        "saplma"
        "lookback_lens"
        "factoscope"



    )
fi

CACHE_SUBDIR="${CACHE_SUBDIR:-100k}"
GEN_MAX_TRAIN="${GEN_MAX_TRAIN:-100000}"
GEN_MAX_VAL="${GEN_MAX_VAL:-100000}"
GEN_MAX_TEST="${GEN_MAX_TEST:-100000}"
GEN_MAX_OOD_VAL="${GEN_MAX_OOD_VAL:-100000}"
GEN_MAX_OOD_TEST="${GEN_MAX_OOD_TEST:-100000}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
ENABLE_POSTHOC="${ENABLE_POSTHOC:-1}"
POSTHOC_METHOD="${POSTHOC_METHOD:-reasoning_logistic_blend}"
POSTHOC_METHODS="${POSTHOC_METHODS:-reasoning_logistic_isotonic reasoning_logistic_blend platt_base temperature_scaling isotonic_regression binwise_hybrid}"
POSTHOC_BASELINE_METHOD="${POSTHOC_BASELINE_METHOD:-reasoning_logistic_blend}"
POSTHOC_TUNE_SPLIT="${POSTHOC_TUNE_SPLIT:-validation}"
PARALLEL_N="${PARALLEL_N:-4}"
PARALLEL_N_EVAL="${PARALLEL_N_EVAL:-6}"

REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-khop-10k}"

PIPELINE_LOG_BASENAME="${PIPELINE_LOG_BASENAME:-multi.log}"

source "${SCRIPT_DIR}/common.sh"
source "${SCRIPT_DIR}/p1.sh"
source "${SCRIPT_DIR}/p2.sh"
source "${SCRIPT_DIR}/p3.sh"
source "${SCRIPT_DIR}/p4.sh"

CACHE_DIR="$(cache_dir_for_dataset "${DATASET_NAME}")"

validate_parallel_n() {
    local name="$1"
    local value="$2"
    if ! [[ "${value}" =~ ^[0-9]+$ ]] || [[ "${value}" -lt 1 ]]; then
        echo "[ERROR] ${name} must be a positive integer, got: ${value}"
        exit 2
    fi
}

wait_one_worker() {
    local stage="$1"
    local done_pid=""
    local rc=0

    if wait -n -p done_pid; then
        rc=0
    else
        rc=$?
    fi
    local label="${WAIT_LABELS[${done_pid:-}]:-pid=${done_pid:-unknown}}"
    if [[ "${rc}" -eq 0 ]]; then
        echo "  [PARALLEL][${stage}] done: ${label}"
    else
        echo "  [PARALLEL][${stage}] FAILED(${rc}): ${label}"
    fi

    if [[ -n "${done_pid:-}" ]]; then
        local -a kept=()
        local pid
        for pid in "${WAIT_PIDS[@]}"; do
            [[ "${pid}" == "${done_pid}" ]] && continue
            kept+=("${pid}")
        done
        WAIT_PIDS=("${kept[@]}")
        unset "WAIT_LABELS[${done_pid}]"
    fi
    return "${rc}"
}

wait_all_workers() {
    local stage="$1"
    local fail=0
    while [[ ${#WAIT_PIDS[@]} -gt 0 ]]; do
        wait_one_worker "${stage}" || fail=1
    done
    return "${fail}"
}

run_phase2_parallel() {
    local cache_dir="${1:-${CACHE_DIR}}"
    local results_base="${2:-$(phase2_train_root "${RESULTS_ROOT}")}"
    local -a head_types
    mapfile -t head_types < <(phase2_head_types)

    local phase2_start
    phase2_start=$(date +%s)

    print_header "Phase 2 — Train Supervised Heads (parallel=${PARALLEL_N})"
    echo "Model:   ${MODEL_NAME}"
    echo "Dataset: ${DATASET_NAME}"
    echo "Cache:   ${cache_dir}"
    echo "Output:  ${results_base}"
    echo "Heads:   ${head_types[*]}"
    echo "Epochs:  ${TRAIN_EPOCHS}"
    echo "Batch:   ${TRAIN_BATCH_SIZE}"
    echo "Budget:  train=${CACHE_TRAIN_MEM_BUDGET_GB} GB / val=${CACHE_VAL_MEM_BUDGET_GB} GB"
    echo "GPUs:    ${GPU_IDS[*]:-0}"
    echo ""

    mkdir -p "${results_base}" "$(phase_logs_root "phase2")/train"

    local -a heads_to_train=()
    local ht
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
    echo "Parallel workers: ${PARALLEL_N}"
    echo ""

    local total=${#heads_to_train[@]}
    local gpu_count=${#GPU_IDS[@]}
    local launch_idx=0
    local gpu_idx
    local any_fail=0
    WAIT_PIDS=()
    declare -gA WAIT_LABELS=()

    for ht in "${heads_to_train[@]}"; do
        while [[ ${#WAIT_PIDS[@]} -ge ${PARALLEL_N} ]]; do
            wait_one_worker "P2-TRAIN" || any_fail=1
        done
        if [[ "${gpu_count}" -gt 0 ]]; then
            gpu_idx="${GPU_IDS[$((launch_idx % gpu_count))]}"
        else
            gpu_idx="0"
        fi
        launch_idx=$((launch_idx + 1))
        (
            echo "    [WORKER][P2] head=${ht} gpu=${gpu_idx}"
            train_head "${ht}" "${cache_dir}" "${results_base}" "${gpu_idx}"
        ) &
        WAIT_PIDS+=("$!")
        WAIT_LABELS["$!"]="head=${ht} gpu=${gpu_idx}"
        echo "  [PARALLEL][P2-TRAIN] launch ${launch_idx}/${total}: head=${ht} gpu=${gpu_idx} active=${#WAIT_PIDS[@]}"
    done

    wait_all_workers "P2-TRAIN" || any_fail=1
    [[ "${any_fail}" -eq 0 ]] || return 1

    local phase2_end
    phase2_end=$(date +%s)
    echo ""
    echo "Phase 2 complete — $(format_duration $((phase2_end - phase2_start)))"
    return 0
}

eval_id_heads_parallel() {
    local cache_dir="$1"
    local results_base="$2"
    local eval_base="${3:-$(phase3_id_results_root "${results_base}")}"
    local split="${4:-test}"
    local -a head_types
    mapfile -t head_types < <(phase3_head_types)

    print_step "ID Head Evaluation — ${split} (parallel=${PARALLEL_N_EVAL})"

    local total=${#head_types[@]}
    local gpu_count=${#GPU_IDS[@]}
    local launch_idx=0
    local gpu_idx ht head_path out_dir prediction_cache_dir
    local any_fail=0
    WAIT_PIDS=()
    declare -gA WAIT_LABELS=()

    for ht in "${head_types[@]}"; do
        while [[ ${#WAIT_PIDS[@]} -ge ${PARALLEL_N_EVAL} ]]; do
            wait_one_worker "P3-ID" || any_fail=1
        done
        head_path="$(phase2_head_dir "${results_base}" "${ht}")/final_model"
        out_dir="${eval_base}/${ht}"
        mkdir -p "${out_dir}"
        if [[ "${gpu_count}" -gt 0 ]]; then
            gpu_idx="${GPU_IDS[$((launch_idx % gpu_count))]}"
        else
            gpu_idx="0"
        fi
        launch_idx=$((launch_idx + 1))
        (
            echo "    [WORKER][P3-ID] head=${ht} split=${split} gpu=${gpu_idx}"
            eval_head "${ht}" "${head_path}" "${cache_dir}" "${out_dir}" "${split}" "${gpu_idx}" \
                "$(phase3_eval_log_path "ID" "" "${split}" "${ht}")"
        ) &
        WAIT_PIDS+=("$!")
        WAIT_LABELS["$!"]="head=${ht} split=${split} gpu=${gpu_idx}"
        echo "  [PARALLEL][P3-ID] launch ${launch_idx}/${total}: head=${ht} gpu=${gpu_idx} active=${#WAIT_PIDS[@]}"
    done
    wait_all_workers "P3-ID" || any_fail=1
    [[ "${any_fail}" -eq 0 ]] || return 1
    return 0
}

eval_ood_dataset_parallel() {
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

    print_step "OOD Evaluation — ${ood_ds} (parallel=${PARALLEL_N_EVAL})"

    local val_split="${OOD_THRESHOLD_TUNE_SPLIT:-validation}"
    local eval_split="test"
    local seed_tune_split="${THRESHOLD_TUNE_SPLIT:-validation}"
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
            [[ -n "${old_data_log}" ]] && _DATA_LOG="${old_data_log}" || unset _DATA_LOG
            return 1
        fi
        [[ -n "${old_data_log}" ]] && _DATA_LOG="${old_data_log}" || unset _DATA_LOG
    fi

    local total=${#head_types[@]}
    local gpu_count=${#GPU_IDS[@]}
    local launch_idx=0
    local gpu_idx ht head_path val_out_dir test_out_dir seed_report val_report val_threshold_source val_fit_thresholds
    local any_fail=0
    WAIT_PIDS=()
    declare -gA WAIT_LABELS=()

    for ht in "${head_types[@]}"; do
        while [[ ${#WAIT_PIDS[@]} -ge ${PARALLEL_N_EVAL} ]]; do
            wait_one_worker "P3-OOD:${ood_ds}" || any_fail=1
        done
        head_path="$(phase2_head_dir "${id_results}" "${ht}")/final_model"
        val_out_dir="$(phase3_ood_head_dir "${id_results}" "${ood_ds}" "${val_split}" "${ht}")"
        test_out_dir="$(phase3_ood_head_dir "${id_results}" "${ood_ds}" "${eval_split}" "${ht}")"
        seed_report="$(phase3_id_head_dir "${id_results}" "${ht}")/evaluation_report.json"
        val_report="${val_out_dir}/evaluation_report.json"
        val_threshold_source=""
        val_fit_thresholds="${OOD_FIT_THRESHOLDS_ON_VAL:-1}"
        if [[ "${OOD_USE_ID_THRESHOLD_SEED:-0}" == "1" ]]; then
            val_threshold_source="${seed_report}"
        fi
        mkdir -p "${val_out_dir}" "${test_out_dir}"
        if [[ "${gpu_count}" -gt 0 ]]; then
            gpu_idx="${GPU_IDS[$((launch_idx % gpu_count))]}"
        else
            gpu_idx="0"
        fi
        launch_idx=$((launch_idx + 1))
        (
            echo "    [WORKER][P3-OOD] dataset=${ood_ds} head=${ht} val=${val_split} test=${eval_split} gpu=${gpu_idx}"
            eval_head "${ht}" "${head_path}" "${ood_cache}" "${val_out_dir}" "${val_split}" "${gpu_idx}" \
                "$(phase3_eval_log_path "OOD" "${ood_ds}" "${val_split}" "${ht}")" "${val_split}" "${val_threshold_source}" "${val_fit_thresholds}" &&
                eval_head "${ht}" "${head_path}" "${ood_cache}" "${test_out_dir}" "${eval_split}" "${gpu_idx}" \
                    "$(phase3_eval_log_path "OOD" "${ood_ds}" "${eval_split}" "${ht}")" "${val_split}" "${val_report}"
        ) &
        WAIT_PIDS+=("$!")
        WAIT_LABELS["$!"]="dataset=${ood_ds} head=${ht} gpu=${gpu_idx}"
        echo "  [PARALLEL][P3-OOD:${ood_ds}] launch ${launch_idx}/${total}: head=${ht} gpu=${gpu_idx} active=${#WAIT_PIDS[@]}"
    done
    wait_all_workers "P3-OOD:${ood_ds}" || any_fail=1
    [[ "${any_fail}" -eq 0 ]] || return 1

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

run_phase3_parallel() {
    local cache_dir="${1:-${CACHE_DIR}}"
    local results_base="${2:-${RESULTS_ROOT}}"
    local -a ood_datasets
    mapfile -t ood_datasets < <(phase3_ood_datasets)

    local phase3_start
    phase3_start=$(date +%s)

    print_header "Phase 3 — Evaluate (${MODEL_NAME} / ${DATASET_NAME}, parallel=${PARALLEL_N_EVAL})"
    echo "Model:       ${MODEL_NAME}"
    echo "Dataset:     ${DATASET_NAME}"
    echo "Cache:       ${cache_dir}"
    echo "Results:     ${results_base}"
    echo "OOD:         ${ood_datasets[*]:-none}"
    echo "Batch:       ${TRAIN_BATCH_SIZE}"
    echo "Budget:      eval=${CACHE_EVAL_MEM_BUDGET_GB} GB / val=${CACHE_VAL_MEM_BUDGET_GB} GB"
    echo "Force eval:  ${FORCE_EVAL:-0}"
    echo "GPUs:        ${GPU_IDS[*]:-0}"
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
        eval_id_heads_parallel "${cache_dir}" "${results_base}" "$(phase3_id_results_root "${results_base}")" "test"
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

    local ood_ds ood_cache ood_eval
    for ood_ds in "${ood_datasets[@]}"; do
        [[ -n "${ood_ds}" ]] || continue
        ood_cache="$(cache_dir_for_dataset "${ood_ds}")"
        ood_eval="$(phase3_ood_split_root "${results_base}" "${ood_ds}" "test")"
        mkdir -p "${ood_eval}"
        eval_ood_dataset_parallel "${ood_ds}" "${ood_cache}" "${results_base}" "${ood_eval}"
    done

    local phase3_end
    phase3_end=$(date +%s)
    echo ""
    echo "Phase 3 complete — $(format_duration $((phase3_end - phase3_start)))"
    return 0
}

run_phase4_parallel() {
    local cache_dir="${1:-${CACHE_DIR}}"
    local results_base="${2:-${RESULTS_ROOT}}"

    if [[ "${ENABLE_POSTHOC:-1}" != "1" ]]; then
        print_step "Phase 4 disabled (ENABLE_POSTHOC=${ENABLE_POSTHOC:-0}); skipping."
        return 0
    fi

    local -a head_types
    mapfile -t head_types < <(phase4_head_types)
    local -a posthoc_methods
    mapfile -t posthoc_methods < <(phase4_posthoc_methods)
    local -a ood_datasets
    mapfile -t ood_datasets < <(phase4_ood_datasets)
    local baseline_method
    baseline_method="$(posthoc_baseline_method)"

    local phase4_start
    phase4_start=$(date +%s)

    print_header "Phase 4 — Post-hoc Calibration (${MODEL_NAME} / ${DATASET_NAME}, parallel=${PARALLEL_N_EVAL})"
    echo "Model:        ${MODEL_NAME}"
    echo "Dataset:      ${DATASET_NAME}"
    echo "Cache:        ${cache_dir}"
    echo "Results:      ${results_base}"
    echo "OOD:          ${ood_datasets[*]:-none}"
    echo "Heads:        ${head_types[*]}"
    echo "Posthoc:      methods=${posthoc_methods[*]} baseline=${baseline_method} tune=${POSTHOC_TUNE_SPLIT:-validation} mode=${POSTHOC_FEATURE_MODE:-auto} min_full=${POSTHOC_MIN_SAMPLES_FOR_FULL:-1500}"
    echo "Force rerun:  ${FORCE_POSTHOC:-0}"
    echo "GPUs:         ${GPU_IDS[*]:-0}"
    echo ""

    local total=${#head_types[@]}
    local gpu_count=${#GPU_IDS[@]}
    local launch_idx=0
    local gpu_idx ht head_path out_dir
    local any_fail=0
    local method method_results compare_results_dir
    local ood_ds ood_cache val_split eval_split seed_tune_split required_splits required_max_samples
    local val_out_dir test_out_dir seed_report val_report old_data_log
    old_data_log="${_DATA_LOG-}"
    local heads_csv="${head_types[*]}"

    for method in "${posthoc_methods[@]}"; do
        method_results="$(phase4_method_results_root "${results_base}" "${method}")"
        compare_results_dir=""
        if [[ "${method}" != "${baseline_method}" ]]; then
            compare_results_dir="$(phase4_method_results_root "${results_base}" "${baseline_method}")"
        fi

        print_step "Phase 4 method — ${method} (parallel=${PARALLEL_N_EVAL})"
        local id_manifest
        id_manifest="$(phase4_manifest_path "${results_base}" "${method}" "ID")"
        if [[ "${FORCE_POSTHOC:-0}" != "1" ]] && completion_manifest_valid "${id_manifest}" "phase4" "ID" "" "${method}" "${heads_csv}"; then
            echo "  [SKIP] Phase 4 ID [${method}] already complete."
        else
        launch_idx=0
        any_fail=0
        WAIT_PIDS=()
        declare -gA WAIT_LABELS=()
        for ht in "${head_types[@]}"; do
            while [[ ${#WAIT_PIDS[@]} -ge ${PARALLEL_N_EVAL} ]]; do
                wait_one_worker "P4-ID:${method}" || any_fail=1
            done
            head_path="$(phase2_head_dir "${results_base}" "${ht}")/final_model"
            out_dir="$(phase4_id_head_dir "${results_base}" "${method}" "${ht}")"
            prediction_cache_dir="$(phase4_prediction_cache_dir "${results_base}" "ID" "${ht}")"
            if [[ "${gpu_count}" -gt 0 ]]; then
                gpu_idx="${GPU_IDS[$((launch_idx % gpu_count))]}"
            else
                gpu_idx="0"
            fi
            launch_idx=$((launch_idx + 1))
            (
                echo "    [WORKER][P4-ID] method=${method} head=${ht} split=test gpu=${gpu_idx}"
                calibrate_head "${ht}" "${head_path}" "${cache_dir}" "${out_dir}" "test" "${gpu_idx}" \
                    "$(phase4_eval_log_path "${method}" "ID" "" "test" "${ht}")" "${THRESHOLD_TUNE_SPLIT:-validation}" "" "${FIT_THRESHOLDS_ON_EVAL_SPLIT:-0}" "${method}" "${prediction_cache_dir}"
            ) &
            WAIT_PIDS+=("$!")
            WAIT_LABELS["$!"]="method=${method} head=${ht} split=test gpu=${gpu_idx}"
            echo "  [PARALLEL][P4-ID:${method}] launch ${launch_idx}/${total}: head=${ht} gpu=${gpu_idx} active=${#WAIT_PIDS[@]}"
        done
        wait_all_workers "P4-ID:${method}" || any_fail=1
        [[ "${any_fail}" -eq 0 ]] || return 1
        local id_result_log
        local id_figure_dir
        local id_summary_json
        id_result_log="$(phase4_result_log_path "${method}" "ID")"
        id_figure_dir="$(phase4_figure_dir "${method}" "ID")"
        id_summary_json="$(phase4_summary_json_path "${results_base}" "${method}" "ID")"
        print_eval_summary "${method_results}" "ID" "" "${id_result_log}" "${id_summary_json}" "${method}"
        run_visualization "${method_results}" "${id_figure_dir}" "ID" "" "${compare_results_dir}" "${baseline_method}" "${method}" "${method}" "${method}" "${baseline_method}"
        local -a id_manifest_files=(
            "${id_result_log}"
            "${id_summary_json}"
            "${id_figure_dir}/$(join_artifact_name "${method}" "${DATASET_NAME}").png"
            "${id_figure_dir}/$(join_artifact_name "${method}" "${DATASET_NAME}")_ranking_heatmap.png"
            "${id_figure_dir}/$(join_artifact_name "${method}" "global_pr_auc_ranking_heatmap").png"
            "${id_figure_dir}/$(join_artifact_name "${method}" "posthoc_improvement_summary").png"
            "${id_figure_dir}/$(join_artifact_name "${method}" "posthoc_improvement_summary").csv"
            "${id_figure_dir}/global_posthoc_method_improvement_heatmap.png"
            "${id_figure_dir}/global_posthoc_method_improvement_heatmap.csv"
        )
        for ht in "${head_types[@]}"; do
            local head_out_dir
            head_out_dir="$(phase4_id_head_dir "${results_base}" "${method}" "${ht}")"
            local -a head_files=()
            mapfile -t head_files < <(phase4_head_artifact_files "${head_out_dir}" "test")
            id_manifest_files+=("${head_files[@]}")
        done
        write_completion_manifest "${id_manifest}" "phase4" "ID" "" "${method}" "${heads_csv}" "${id_manifest_files[@]}"
        fi

        for ood_ds in "${ood_datasets[@]}"; do
            [[ -n "${ood_ds}" ]] || continue
            print_step "Phase 4 OOD — ${ood_ds} [${method}] (parallel=${PARALLEL_N_EVAL})"
            local ood_manifest
            ood_manifest="$(phase4_manifest_path "${results_base}" "${method}" "OOD" "${ood_ds}")"
            if [[ "${FORCE_POSTHOC:-0}" != "1" ]] && completion_manifest_valid "${ood_manifest}" "phase4" "OOD" "${ood_ds}" "${method}" "${heads_csv}"; then
                echo "  [SKIP] Phase 4 OOD ${ood_ds} [${method}] already complete."
                continue
            fi

            ood_cache="$(cache_dir_for_dataset "${ood_ds}")"
            val_split="${OOD_THRESHOLD_TUNE_SPLIT:-validation}"
            eval_split="test"
            seed_tune_split="${THRESHOLD_TUNE_SPLIT:-validation}"
            if [[ "${val_split}" == "${eval_split}" ]]; then
                echo "  [ERROR] OOD tune split (${val_split}) must differ from eval split (${eval_split}) to avoid leakage."
                return 1
            fi
            required_splits="${val_split},${eval_split}"
            required_max_samples="${GEN_MAX_OOD_VAL},${GEN_MAX_OOD_TEST}"

            if ! cache_splits_ready "${ood_cache}" "${required_splits}"; then
                echo "  [P4/OOD] ${ood_ds}: cache not ready for ${required_splits}, attempting generation."
                if [[ -n "${LOGS_ROOT:-}" ]]; then
                    _DATA_LOG="$(phase1_data_log_path "${ood_ds}" "OOD")"
                    mkdir -p "$(dirname "${_DATA_LOG}")"
                fi
                if type run_generation &>/dev/null; then
                    run_generation "${ood_ds}" "${ood_cache}" "${required_splits}" "${required_max_samples}" "${GEN_OOD_BATCH_SIZE}"
                    if type run_judge &>/dev/null; then
                        run_judge "${ood_cache}" "${required_splits}" "p4-ood-${ood_ds}"
                    fi
                    if type run_cleanup_pending_claims &>/dev/null; then
                        run_cleanup_pending_claims "${ood_cache}" "${required_splits}"
                    fi
                else
                    echo "  [ERROR] p1 helpers unavailable, cannot build OOD cache for ${ood_ds}."
                    [[ -n "${old_data_log}" ]] && _DATA_LOG="${old_data_log}" || unset _DATA_LOG
                    return 1
                fi
                [[ -n "${old_data_log}" ]] && _DATA_LOG="${old_data_log}" || unset _DATA_LOG
            fi

            launch_idx=0
            any_fail=0
            WAIT_PIDS=()
            declare -gA WAIT_LABELS=()
            for ht in "${head_types[@]}"; do
                while [[ ${#WAIT_PIDS[@]} -ge ${PARALLEL_N_EVAL} ]]; do
                    wait_one_worker "P4-OOD:${ood_ds}:${method}" || any_fail=1
                done
                head_path="$(phase2_head_dir "${results_base}" "${ht}")/final_model"
                val_out_dir="$(phase4_ood_head_dir "${results_base}" "${method}" "${ood_ds}" "${val_split}" "${ht}")"
                test_out_dir="$(phase4_ood_head_dir "${results_base}" "${method}" "${ood_ds}" "${eval_split}" "${ht}")"
                seed_report="$(phase4_id_head_dir "${results_base}" "${method}" "${ht}")/evaluation_report.json"
                val_report="${val_out_dir}/evaluation_report.json"
                prediction_cache_dir="$(phase4_prediction_cache_dir "${results_base}" "OOD" "${ht}" "${ood_ds}")"
                mkdir -p "${val_out_dir}" "${test_out_dir}"
                if [[ "${gpu_count}" -gt 0 ]]; then
                    gpu_idx="${GPU_IDS[$((launch_idx % gpu_count))]}"
                else
                    gpu_idx="0"
                fi
                launch_idx=$((launch_idx + 1))
                (
                    echo "    [WORKER][P4-OOD] method=${method} dataset=${ood_ds} head=${ht} val=${val_split} test=${eval_split} gpu=${gpu_idx}"
                    calibrate_head "${ht}" "${head_path}" "${ood_cache}" "${val_out_dir}" "${val_split}" "${gpu_idx}" \
                        "$(phase4_eval_log_path "${method}" "OOD" "${ood_ds}" "${val_split}" "${ht}")" "${seed_tune_split}" "${seed_report}" "${FIT_THRESHOLDS_ON_EVAL_SPLIT:-0}" "${method}" "${prediction_cache_dir}" &&
                        calibrate_head "${ht}" "${head_path}" "${ood_cache}" "${test_out_dir}" "${eval_split}" "${gpu_idx}" \
                            "$(phase4_eval_log_path "${method}" "OOD" "${ood_ds}" "${eval_split}" "${ht}")" "${val_split}" "${val_report}" "0" "${method}" "${prediction_cache_dir}"
                ) &
                WAIT_PIDS+=("$!")
                WAIT_LABELS["$!"]="method=${method} dataset=${ood_ds} head=${ht} gpu=${gpu_idx}"
                echo "  [PARALLEL][P4-OOD:${ood_ds}:${method}] launch ${launch_idx}/${total}: head=${ht} gpu=${gpu_idx} active=${#WAIT_PIDS[@]}"
            done
            wait_all_workers "P4-OOD:${ood_ds}:${method}" || any_fail=1
            [[ "${any_fail}" -eq 0 ]] || return 1

            local result_log
            local figure_dir
            local summary_json
            result_log="$(phase4_result_log_path "${method}" "OOD" "${ood_ds}")"
            figure_dir="$(phase4_figure_dir "${method}" "OOD" "${ood_ds}")"
            summary_json="$(phase4_summary_json_path "${results_base}" "${method}" "OOD" "${ood_ds}")"
            print_eval_summary "${method_results}" "OOD" "${ood_ds}" "${result_log}" "${summary_json}" "${method}"
            run_visualization "${method_results}" "${figure_dir}" "OOD" "${ood_ds}" "${compare_results_dir}" "${baseline_method}" "${method}" "${method}" "${method}" "${baseline_method}"
            local -a ood_manifest_files=(
                "${result_log}"
                "${summary_json}"
                "${figure_dir}/$(join_artifact_name "${method}" "${ood_ds}").png"
                "${figure_dir}/$(join_artifact_name "${method}" "${ood_ds}")_ranking_heatmap.png"
                "${figure_dir}/$(join_artifact_name "${method}" "posthoc_improvement_summary").png"
                "${figure_dir}/$(join_artifact_name "${method}" "posthoc_improvement_summary").csv"
                "${figure_dir}/global_posthoc_method_improvement_heatmap.png"
                "${figure_dir}/global_posthoc_method_improvement_heatmap.csv"
            )
            for ht in "${head_types[@]}"; do
                local val_head_dir
                local test_head_dir
                val_head_dir="$(phase4_ood_head_dir "${results_base}" "${method}" "${ood_ds}" "${val_split}" "${ht}")"
                test_head_dir="$(phase4_ood_head_dir "${results_base}" "${method}" "${ood_ds}" "${eval_split}" "${ht}")"
                local -a val_head_files=()
                local -a test_head_files=()
                mapfile -t val_head_files < <(phase4_head_artifact_files "${val_head_dir}" "${val_split}")
                mapfile -t test_head_files < <(phase4_head_artifact_files "${test_head_dir}" "${eval_split}")
                ood_manifest_files+=("${val_head_files[@]}" "${test_head_files[@]}")
            done
            write_completion_manifest "${ood_manifest}" "phase4" "OOD" "${ood_ds}" "${method}" "${heads_csv}" "${ood_manifest_files[@]}"
        done
    done

    local phase4_end
    phase4_end=$(date +%s)
    echo ""
    echo "Phase 4 complete — $(format_duration $((phase4_end - phase4_start)))"
    return 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --resume)
            if [[ -z "${2:-}" ]]; then
                echo "[ERROR] --resume requires a job id"
                exit 2
            fi
            RESUME_JOB_ID="$2"
            shift 2
            ;;
        --retrain)
            RETRAIN=1
            shift
            ;;
        --parallel|-N)
            if [[ -z "${2:-}" ]]; then
                echo "[ERROR] $1 requires a positive integer"
                exit 2
            fi
            PARALLEL_N="$2"
            shift 2
            ;;
        --parallel-eval)
            if [[ -z "${2:-}" ]]; then
                echo "[ERROR] $1 requires a positive integer"
                exit 2
            fi
            PARALLEL_N_EVAL="$2"
            shift 2
            ;;
        *)
            echo "[ERROR] Unknown argument: $1"
            exit 2
            ;;
    esac
done

validate_parallel_n "PARALLEL_N" "${PARALLEL_N}"
validate_parallel_n "PARALLEL_N_EVAL" "${PARALLEL_N_EVAL}"

if [[ -n "${RESUME_JOB_ID:-}" ]]; then
    echo "Resuming from job ${RESUME_JOB_ID}..."
    RESULTS_ROOT="${BASE_RESULTS_ROOT}/${RESUME_JOB_ID}"
    LOGS_ROOT="${BASE_LOGS_ROOT}/${RESUME_JOB_ID}"
fi

setup_environment
detect_gpus

IFS=' ' read -ra OOD_ARRAY <<< "${OOD_DATASETS:-MuSiQue IIRC StrategyQA 2WikiMultihopQA StepGame}"

mkdir -p "${LOGS_ROOT}"
PIPE_LOG="${LOGS_ROOT}/${PIPELINE_LOG_BASENAME}"
if [[ -n "${RESUME_JOB_ID:-}" && -f "${PIPE_LOG}" ]]; then
    _resume_n=1
    _base_log="${PIPELINE_LOG_BASENAME%.log}"
    [[ -n "${_base_log}" ]] || _base_log="multi"
    while [[ -f "${LOGS_ROOT}/${_base_log}_resume_${_resume_n}.log" ]]; do
        _resume_n=$((_resume_n + 1))
    done
    PIPE_LOG="${LOGS_ROOT}/${_base_log}_resume_${_resume_n}.log"
    echo "Resume log: ${PIPE_LOG}"
fi
exec > >(tee "${PIPE_LOG}") 2>&1

print_header "khop Multi Pipeline"
mapfile -t _head_types < <(phase2_head_types)
echo "Model:      ${MODEL_NAME}"
echo "Dataset:    ${DATASET_NAME}"
echo "OOD:        ${OOD_ARRAY[*]}"
echo "Heads:      ${_head_types[*]}"
echo "Epochs:     ${TRAIN_EPOCHS}"
echo "Batch:      ${TRAIN_BATCH_SIZE}"
echo "Parallel:   train=${PARALLEL_N} / eval=${PARALLEL_N_EVAL}"
echo "GPU IDs:    ${GPU_IDS[*]:-0}"
echo "Samples:    train=${GEN_MAX_TRAIN} / val=${GEN_MAX_VAL} / test=${GEN_MAX_TEST} / ood_val=${GEN_MAX_OOD_VAL} / ood_test=${GEN_MAX_OOD_TEST}"
echo "Cache:      ${CACHE_DIR}"
echo "Results:    ${RESULTS_ROOT}"
echo "Budget:     train=${CACHE_TRAIN_MEM_BUDGET_GB} GB / val=${CACHE_VAL_MEM_BUDGET_GB} GB / eval=${CACHE_EVAL_MEM_BUDGET_GB} GB"
echo "Threshold:  ${EVAL_THRESHOLD} (tune=${THRESHOLD_TUNE_SPLIT}, require_tune=${REQUIRE_TUNE_SPLIT}, fit_eval=${FIT_THRESHOLDS_ON_EVAL_SPLIT})"
echo "Force eval: ${FORCE_EVAL:-0}"
echo "P4:         enable=${ENABLE_POSTHOC} method=${POSTHOC_METHOD} tune=${POSTHOC_TUNE_SPLIT} force=${FORCE_POSTHOC:-0}"
echo "Retrain:    ${RETRAIN:-0}"
echo "Log:        level=${LOG_LEVEL} banner=${LOG_BANNER_WIDTH}"
echo ""
unset _head_types

log_gpu_snapshot "multi-start"

if [[ "${RETRAIN:-0}" == "1" ]]; then
    cleanup_retrain_artifacts "${RESULTS_ROOT}" "${LOGS_ROOT}"
fi

_validated_models=()
for M in "${MODEL_NAME}" "${JUDGE_MODEL_NAME}"; do
    for _done in "${_validated_models[@]+"${_validated_models[@]}"}"; do
        [[ "${_done}" == "${M}" ]] && continue 2
    done
    if [[ ! -d "${MODELS_ROOT}/${M}" ]]; then
        echo "[ERROR] Model not found: ${MODELS_ROOT}/${M}"
        exit 1
    fi
    _validated_models+=("${M}")
done
unset _validated_models

run_phase1
if [[ "${PARALLEL_N}" == "1" ]]; then
    run_phase2 "${CACHE_DIR}" "$(phase2_train_root "${RESULTS_ROOT}")"
else
    run_phase2_parallel "${CACHE_DIR}" "$(phase2_train_root "${RESULTS_ROOT}")"
fi

if [[ "${PARALLEL_N_EVAL}" == "1" ]]; then
    run_phase3 "${CACHE_DIR}" "${RESULTS_ROOT}"
    run_phase4 "${CACHE_DIR}" "${RESULTS_ROOT}"
else
    run_phase3_parallel "${CACHE_DIR}" "${RESULTS_ROOT}"
    run_phase4_parallel "${CACHE_DIR}" "${RESULTS_ROOT}"
fi

"${PYTHON_BIN}" "${PROJECT_DIR}/scripts/analyze_uq_heads.py" \
    --results_dir "$(phase3_results_root "${RESULTS_ROOT}")" \
    --output_path "$(phase_logs_root "phase3")/id/result/uq_head_analysis.md"

log_gpu_snapshot "multi-end"

PIPELINE_END=$(date +%s)
echo ""
echo "=== Multi Pipeline Complete ($(format_duration $((PIPELINE_END - PIPELINE_START)))) ==="
