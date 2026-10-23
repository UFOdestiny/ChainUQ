#!/usr/bin/env bash

#SBATCH --job-name=khop-p4

#SBATCH --qos=fsu-compsci-dept
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120gb
#SBATCH --time=1-00:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

if [[ -n "${_P4_SOURCED:-}" ]]; then
    return 0 2>/dev/null || true
fi
_P4_SOURCED=1

phase4_head_types() {
    local raw="${PHASE4_HEAD_TYPES:-${PHASE3_HEAD_TYPES:-${HEAD_TYPES:-}}}"
    local -a head_types
    if [[ -n "${raw}" ]]; then
        IFS=' ' read -ra head_types <<< "${raw}"
    elif type phase3_head_types &>/dev/null; then
        mapfile -t head_types < <(phase3_head_types)
    else
        head_types=("${ALL_HEAD_TYPES[@]}")
    fi
    printf '%s\n' "${head_types[@]}"
}

phase4_ood_datasets() {
    if [[ "${PHASE4_OOD_DATASETS+x}" == "x" ]]; then
        IFS=' ' read -ra _datasets <<< "${PHASE4_OOD_DATASETS}"
        printf '%s\n' "${_datasets[@]}"
        return
    fi
    if type phase3_ood_datasets &>/dev/null; then
        phase3_ood_datasets
        return
    fi
    IFS=' ' read -ra _datasets <<< "${OOD_DATASETS:-}"
    printf '%s\n' "${_datasets[@]}"
}

posthoc_baseline_method() {
    local baseline="${POSTHOC_BASELINE_METHOD:-reasoning_logistic}"
    baseline="$(printf '%s' "${baseline}" | xargs)"
    if [[ -z "${baseline}" ]]; then
        baseline="reasoning_logistic"
    fi
    printf '%s\n' "${baseline}"
}

phase4_posthoc_methods() {
    local raw="${POSTHOC_METHODS:-${POSTHOC_METHOD:-reasoning_logistic}}"
    local baseline
    baseline="$(posthoc_baseline_method)"
    local -a parsed=()
    local -a ordered=()
    local token
    raw="${raw//,/ }"
    IFS=' ' read -r -a parsed <<< "${raw}"
    for token in "${parsed[@]}"; do
        token="$(printf '%s' "${token}" | xargs)"
        [[ -n "${token}" ]] || continue
        if [[ " ${ordered[*]} " != *" ${token} "* ]]; then
            ordered+=("${token}")
        fi
    done
    if [[ ${#ordered[@]} -eq 0 ]]; then
        ordered=("${baseline}")
    fi
    if [[ " ${ordered[*]} " != *" ${baseline} "* ]]; then
        ordered=("${baseline}" "${ordered[@]}")
    elif [[ "${ordered[0]}" != "${baseline}" ]]; then
        local -a reordered=("${baseline}")
        for token in "${ordered[@]}"; do
            [[ "${token}" == "${baseline}" ]] && continue
            reordered+=("${token}")
        done
        ordered=("${reordered[@]}")
    fi
    printf '%s\n' "${ordered[@]}"
}

is_posthoc_complete() {
    local eval_dir="$1"
    local split="${2:-test}"
    local status_path="${eval_dir}/posthoc_status.json"
    local report_path="${eval_dir}/evaluation_report.json"
    if [[ -f "${status_path}" ]]; then
        if "${PYTHON_BIN:-python3}" - "${status_path}" "${split}" <<'PY' >/dev/null 2>&1
import json
import os
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
split = str(sys.argv[2] or "test").strip().lower()
try:
    payload = json.loads(status_path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)

if payload.get("status") != "complete":
    sys.exit(1)

reported_split = str(payload.get("split", "") or "").strip().lower()
if reported_split and reported_split != split:
    sys.exit(1)

required = [str(key) for key in (payload.get("required_artifacts", []) or []) if key]
if split == "test" and "calibrator" not in required:
    required.append("calibrator")
for key in ("evaluation_report", "diagnostics"):
    if key not in required:
        required.append(key)

artifacts = payload.get("artifacts", {}) or {}
base_dir = status_path.parent
for key in required:
    path_str = artifacts.get(key)
    if not path_str:
        sys.exit(1)
    path = Path(path_str)
    if not path.is_absolute():
        path = base_dir / path
    if not path.exists():
        sys.exit(1)

if split == "test":
    if not bool(payload.get("posthoc_enabled", False)):
        sys.exit(1)
    if not bool(payload.get("posthoc_applied", False)):
        sys.exit(1)

sys.exit(0)
PY
        then
            return 0
        fi
    fi

    [[ -f "${report_path}" ]] || return 1

    "${PYTHON_BIN:-python3}" - "${report_path}" "${split}" <<'PY' >/dev/null 2>&1
import json
import os
import sys

path = sys.argv[1]
split = str(sys.argv[2] or "test").strip().lower()
try:
    with open(path, "r", encoding="utf-8") as f:
        report = json.load(f)
except Exception:
    sys.exit(1)

posthoc = report.get("posthoc", {}) or {}
if not posthoc:
    tp = report.get("threshold_protocol", {}) or {}
    posthoc = tp.get("posthoc", {}) or {}
if split != "test":
    diagnostics_path = report.get("diagnostics_path")
    if diagnostics_path and not os.path.isabs(diagnostics_path):
        diagnostics_path = os.path.join(os.path.dirname(path), diagnostics_path)
    if diagnostics_path and not os.path.isfile(diagnostics_path):
        sys.exit(1)
    sys.exit(0)

if not bool(posthoc.get("applied", False)):
    sys.exit(1)
cal_path = posthoc.get("calibrator_path")
if not cal_path:
    sys.exit(1)
if not os.path.isabs(cal_path):
    cal_path = os.path.join(os.path.dirname(path), cal_path)
if not os.path.isfile(cal_path):
    sys.exit(1)
sys.exit(0)
PY
}

calibrate_head() {
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
    local posthoc_method="${11:-${POSTHOC_METHOD:-reasoning_logistic}}"
    local prediction_cache_dir="${12:-}"
    local force_posthoc="${FORCE_POSTHOC:-0}"
    local require_tune="${REQUIRE_TUNE_SPLIT:-1}"

    local -a require_tune_arg=()
    local -a threshold_source_arg=()
    local -a aux_threshold_args=()
    local -a fit_eval_split_arg=()
    local -a prediction_cache_arg=()

    if [[ ! -f "${head_path}/head_weights.pth" ]]; then
        echo "  [SKIP] ${head_type}: no trained model at ${head_path}"
        return 0
    fi

    if [[ "${force_posthoc}" != "1" ]] && is_posthoc_complete "${output_dir}" "${split}"; then
        echo "  [SKIP] ${head_type}: post-hoc already complete for ${split}."
        return 0
    fi

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
    if [[ "${ENABLE_TEMP_SCALING:-0}" == "1" ]]; then
        aux_threshold_args+=(--enable_temp_scaling)
    fi
    if [[ "${ENABLE_DIFFICULTY_THRESHOLDS:-0}" == "1" ]]; then
        aux_threshold_args+=(
            --enable_difficulty_thresholds
            --difficulty_threshold_min_samples "${DIFFICULTY_THRESHOLD_MIN_SAMPLES:-100}"
        )
    fi
    if [[ "${fit_thresholds_on_eval_split}" == "1" ]]; then
        fit_eval_split_arg=(--fit_thresholds_on_eval_split)
    fi
    if [[ -n "${prediction_cache_dir}" ]]; then
        prediction_cache_arg=(--prediction_cache_dir "${prediction_cache_dir}")
    fi

    mkdir -p "$(dirname "${log_path}")" "${output_dir}"
    echo "  [P4] ${head_type} post-hoc on ${split} (GPU ${gpu_idx})..."

    CUDA_VISIBLE_DEVICES="${gpu_idx}" "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/evaluate.py" \
        --head_path "${head_path}" \
        --cache_dir "${cache_dir}" \
        --split "${split}" \
        --threshold "${EVAL_THRESHOLD}" \
        --threshold_tune_split "${tune_split}" \
        "${threshold_source_arg[@]}" \
        "${aux_threshold_args[@]}" \
        "${fit_eval_split_arg[@]}" \
        --enable_posthoc \
        --posthoc_method "${posthoc_method}" \
        --posthoc_tune_split "${POSTHOC_TUNE_SPLIT:-validation}" \
        --posthoc_feature_mode "${POSTHOC_FEATURE_MODE:-auto}" \
        --posthoc_min_samples_for_full "${POSTHOC_MIN_SAMPLES_FOR_FULL:-1500}" \
        --output_dir "${output_dir}" \
        "${prediction_cache_arg[@]}" \
        --batch_size "${TRAIN_BATCH_SIZE}" \
        --force \
        "${require_tune_arg[@]}" \
        2>&1 | tee -a "${log_path}"

    local rc=${PIPESTATUS[0]}
    if [[ ${rc} -eq 0 ]]; then
        echo "  [P4] ${head_type} post-hoc (${split}) complete ✓"
    else
        echo "  [P4] ${head_type} post-hoc (${split}) FAILED ✗"
    fi
    return "${rc}"
}

run_phase4() {
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

    print_header "Phase 4 — Post-hoc Calibration (${MODEL_NAME} / ${DATASET_NAME})"
    echo "Model:        ${MODEL_NAME}"
    echo "Dataset:      ${DATASET_NAME}"
    echo "Cache:        ${cache_dir}"
    echo "Results:      ${results_base}"
    echo "OOD:          ${ood_datasets[*]:-none}"
    echo "Heads:        ${head_types[*]}"
    echo "Posthoc:      methods=${posthoc_methods[*]} baseline=${baseline_method} tune=${POSTHOC_TUNE_SPLIT:-validation} mode=${POSTHOC_FEATURE_MODE:-auto} min_full=${POSTHOC_MIN_SAMPLES_FOR_FULL:-1500}"
    echo "Force rerun:  ${FORCE_POSTHOC:-0}"
    echo ""

    local method method_results compare_results_dir
    local old_data_log="${_DATA_LOG-}"
    local heads_csv="${head_types[*]}"
    for method in "${posthoc_methods[@]}"; do
        method_results="$(phase4_method_results_root "${results_base}" "${method}")"
        compare_results_dir=""
        if [[ "${method}" != "${baseline_method}" ]]; then
            compare_results_dir="$(phase4_method_results_root "${results_base}" "${baseline_method}")"
        fi

        print_step "Phase 4 method — ${method}"
        local id_manifest
        id_manifest="$(phase4_manifest_path "${results_base}" "${method}" "ID")"
        if [[ "${FORCE_POSTHOC:-0}" != "1" ]] && completion_manifest_valid "${id_manifest}" "phase4" "ID" "" "${method}" "${heads_csv}"; then
            echo "  [SKIP] Phase 4 ID [${method}] already complete."
        else
            for ht in "${head_types[@]}"; do
                local head_path="$(phase2_head_dir "${results_base}" "${ht}")/final_model"
                local out_dir
                out_dir="$(phase4_id_head_dir "${results_base}" "${method}" "${ht}")"
                local prediction_cache_dir
                prediction_cache_dir="$(phase4_prediction_cache_dir "${results_base}" "ID" "${ht}")"
                calibrate_head "${ht}" "${head_path}" "${cache_dir}" "${out_dir}" "test" "${GPU_IDS[0]:-0}" \
                    "$(phase4_eval_log_path "${method}" "ID" "" "test" "${ht}")" "${THRESHOLD_TUNE_SPLIT:-validation}" "" "${FIT_THRESHOLDS_ON_EVAL_SPLIT:-0}" "${method}" "${prediction_cache_dir}"
            done
            local id_result_log
            local id_figure_dir
            local id_summary_json
            id_result_log="$(phase4_result_log_path "${method}" "ID")"
            id_figure_dir="$(phase4_figure_dir "${method}" "ID")"
            id_summary_json="$(phase4_summary_json_path "${results_base}" "${method}" "ID")"
            if type print_eval_summary &>/dev/null; then
                print_eval_summary "${method_results}" "ID" "" "${id_result_log}" "${id_summary_json}" "${method}"
            fi
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

        local ood_ds
        for ood_ds in "${ood_datasets[@]}"; do
            [[ -n "${ood_ds}" ]] || continue
            print_step "Phase 4 OOD — ${ood_ds} [${method}]"
            local ood_manifest
            ood_manifest="$(phase4_manifest_path "${results_base}" "${method}" "OOD" "${ood_ds}")"
            if [[ "${FORCE_POSTHOC:-0}" != "1" ]] && completion_manifest_valid "${ood_manifest}" "phase4" "OOD" "${ood_ds}" "${method}" "${heads_csv}"; then
                echo "  [SKIP] Phase 4 OOD ${ood_ds} [${method}] already complete."
                continue
            fi

            local ood_cache
            ood_cache="$(cache_dir_for_dataset "${ood_ds}")"
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

            for ht in "${head_types[@]}"; do
                local head_path="$(phase2_head_dir "${results_base}" "${ht}")/final_model"
                local val_out_dir
                val_out_dir="$(phase4_ood_head_dir "${results_base}" "${method}" "${ood_ds}" "${val_split}" "${ht}")"
                local test_out_dir
                test_out_dir="$(phase4_ood_head_dir "${results_base}" "${method}" "${ood_ds}" "${eval_split}" "${ht}")"
                local seed_report="$(phase4_id_head_dir "${results_base}" "${method}" "${ht}")/evaluation_report.json"
                local val_report="${val_out_dir}/evaluation_report.json"
                local prediction_cache_dir
                prediction_cache_dir="$(phase4_prediction_cache_dir "${results_base}" "OOD" "${ht}" "${ood_ds}")"
                mkdir -p "${val_out_dir}" "${test_out_dir}"

                calibrate_head "${ht}" "${head_path}" "${ood_cache}" "${val_out_dir}" "${val_split}" "${GPU_IDS[0]:-0}" \
                    "$(phase4_eval_log_path "${method}" "OOD" "${ood_ds}" "${val_split}" "${ht}")" "${seed_tune_split}" "${seed_report}" "${FIT_THRESHOLDS_ON_EVAL_SPLIT:-0}" "${method}" "${prediction_cache_dir}"
                calibrate_head "${ht}" "${head_path}" "${ood_cache}" "${test_out_dir}" "${eval_split}" "${GPU_IDS[0]:-0}" \
                    "$(phase4_eval_log_path "${method}" "OOD" "${ood_ds}" "${eval_split}" "${ht}")" "${val_split}" "${val_report}" "0" "${method}" "${prediction_cache_dir}"
            done

            local result_log
            local figure_dir
            local summary_json
            result_log="$(phase4_result_log_path "${method}" "OOD" "${ood_ds}")"
            figure_dir="$(phase4_figure_dir "${method}" "OOD" "${ood_ds}")"
            summary_json="$(phase4_summary_json_path "${results_base}" "${method}" "OOD" "${ood_ds}")"
            if type print_eval_summary &>/dev/null; then
                print_eval_summary "${method_results}" "OOD" "${ood_ds}" "${result_log}" "${summary_json}" "${method}"
            fi
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

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail

    SCRIPT_DIR="${SCRIPT_DIR:-/jobs}"

    source "${SCRIPT_DIR}/common.sh"

    source "${SCRIPT_DIR}/p3.sh" 2>/dev/null || true

    source "${SCRIPT_DIR}/p1.sh" 2>/dev/null || true

    if [[ -n "${HEAD_TYPES:-}" ]]; then
        PHASE4_HEAD_TYPES="${HEAD_TYPES}"
    fi

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

    mkdir -p "$(dirname "$(phase_run_log_path "phase4")")"
    exec > >(tee "$(phase_run_log_path "phase4")") 2>&1

    log_gpu_snapshot "p4-start"
    run_phase4 "${CACHE_DIR}" "${RESULTS_ROOT}"
    RC=$?
    log_gpu_snapshot "p4-end"
    exit ${RC}
fi
