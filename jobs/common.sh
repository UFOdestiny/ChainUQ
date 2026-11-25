#!/usr/bin/env bash

_SCRIPT_DIR="/jobs"
_PROJECT_ROOT="$(dirname "${_SCRIPT_DIR}")"

PROJECT_DIR="${PROJECT_DIR:-${_PROJECT_ROOT}}"
POPLLM_ROOT="${POPLLM_ROOT:-/popllm}"
MODELS_ROOT="${MODELS_ROOT:-${POPLLM_ROOT}/models}"
DATASETS_ROOT="${DATASETS_ROOT:-${POPLLM_ROOT}/datasets}"
BASE_RESULTS_ROOT="${BASE_RESULTS_ROOT:-${POPLLM_ROOT}/results}"
BASE_LOGS_ROOT="${BASE_LOGS_ROOT:-${POPLLM_ROOT}/logs}"

RESULTS_ROOT="${RESULTS_ROOT:-${BASE_RESULTS_ROOT}/${SLURM_JOB_ID:-manual_run}}"
LOGS_ROOT="${LOGS_ROOT:-${BASE_LOGS_ROOT}/${SLURM_JOB_ID:-manual_run}}"
export LOGS_ROOT
HF_CACHE="${HF_CACHE:-${MODELS_ROOT}/.hf_cache}"

VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.6}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"

GPU_MEM_UTIL_OVERRIDES="${GPU_MEM_UTIL_OVERRIDES:-MuSiQue=0.5 hotpot_qa=0.5 2WikiMultihopQA=0.5 2wikimultihopqa=0.5 IIRC=0.5 iirc=0.5 StepGame=0.5 StrategyQA=0.5 strategyqa=0.5}"

MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"
DATASET_NAME="${DATASET_NAME:-hotpot_qa}"
JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME:-Mistral-Small-3.2-24B-Instruct-2506}"
CLAIM_LABELER_MAX_NEW_TOKENS="${CLAIM_LABELER_MAX_NEW_TOKENS:-192}"
JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE:-512}"
FORCE_JUDGE="${FORCE_JUDGE:-0}"


CACHE_SUBDIR="${CACHE_SUBDIR:-}"
MEM_TRAIN_BUDGET_RATIO="${MEM_TRAIN_BUDGET_RATIO:-0.55}"
MEM_VAL_BUDGET_RATIO="${MEM_VAL_BUDGET_RATIO:-0.25}"
MEM_EVAL_BUDGET_RATIO="${MEM_EVAL_BUDGET_RATIO:-0.70}"
if [[ -n "${CACHE_SUBDIR}" ]]; then
    CACHE_DIR="${POPLLM_ROOT}/cached_features/${CACHE_SUBDIR}/${DATASET_NAME}/${MODEL_NAME}"
else
    CACHE_DIR="${POPLLM_ROOT}/cached_features/${DATASET_NAME}/${MODEL_NAME}"
fi

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

HEAD_UNIVERSE="${HEAD_UNIVERSE:-uq_v1 uq_v2 uq_v3 uq_abl_v1 uq_abl_v2 uq_abl_v3 uq_abl_v4 luh_head luh_light saplma lookback_lens factoscope random mcp perplexity token_entropy ccp}"

TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-128}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-0.00015}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-1}"
RETRAIN="${RETRAIN:-0}"
LOSS_TYPE="${LOSS_TYPE:-balanced_bce}"
LOSS_POS_WEIGHT="${LOSS_POS_WEIGHT:--1}"
FOCAL_GAMMA="${FOCAL_GAMMA:-2.0}"
SAMPLE_POS_WEIGHT="${SAMPLE_POS_WEIGHT:--1}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.02}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.05}"
WARMUP_RATIO="${WARMUP_RATIO:-0.06}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"

REPORT_TO="${REPORT_TO:-none}"
WANDB_PROJECT="${WANDB_PROJECT:-khop}"
WANDB_ENTITY="${WANDB_ENTITY:-ufodestiny-}"

GEN_BATCH_SIZE="${GEN_BATCH_SIZE:-512}"
GEN_OOD_BATCH_SIZE="${GEN_OOD_BATCH_SIZE:-256}"
BACKEND="${BACKEND:-vllm}"
GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS:-512}"
GEN_PROMPT_MAX_TOKENS="${GEN_PROMPT_MAX_TOKENS:-3072}"
JUDGE_PROMPT_MAX_TOKENS="${JUDGE_PROMPT_MAX_TOKENS:-2048}"
FEATURE_HIDDEN_STATE_LAYERS="${FEATURE_HIDDEN_STATE_LAYERS:--1,-2,-4,-8}"
FEATURE_HIDDEN_STATE_FUSION="${FEATURE_HIDDEN_STATE_FUSION:-weighted_sum}"
FEATURE_HIDDEN_STATE_WEIGHTS="${FEATURE_HIDDEN_STATE_WEIGHTS:-0.4,0.3,0.2,0.1}"
FEATURE_TOP_N_PROBS="${FEATURE_TOP_N_PROBS:-6}"
FEATURE_TOKEN_APPEND_STATS="${FEATURE_TOKEN_APPEND_STATS:-1}"
FEATURE_ATTENTION_LAYERS="${FEATURE_ATTENTION_LAYERS:--1}"
FEATURE_ATTENTION_HEADS="${FEATURE_ATTENTION_HEADS:-all}"
FEATURE_ATTN_HISTORY_SZ="${FEATURE_ATTN_HISTORY_SZ:-3}"
FEATURE_POOL_ATTENTION_LAYERS="${FEATURE_POOL_ATTENTION_LAYERS:-1}"

GEN_MAX_TRAIN="${GEN_MAX_TRAIN:-0}"
GEN_MAX_VAL="${GEN_MAX_VAL:-0}"
GEN_MAX_TEST="${GEN_MAX_TEST:-0}"
GEN_MAX_OOD_VAL="${GEN_MAX_OOD_VAL:-${GEN_MAX_OOD:-0}}"
GEN_MAX_OOD_TEST="${GEN_MAX_OOD_TEST:-${GEN_MAX_OOD:-${GEN_MAX_TEST}}}"

OOD_DATASETS="${OOD_DATASETS:-MuSiQue IIRC StrategyQA 2WikiMultihopQA StepGame babi}" # spartqa TruthfulQA MathQA ScienceQA
EVAL_THRESHOLD="${EVAL_THRESHOLD:-0.5}"
FORCE_EVAL="${FORCE_EVAL:-0}"
THRESHOLD_TUNE_SPLIT="${THRESHOLD_TUNE_SPLIT:-validation}"
OOD_THRESHOLD_TUNE_SPLIT="${OOD_THRESHOLD_TUNE_SPLIT:-validation}"
REQUIRE_TUNE_SPLIT="${REQUIRE_TUNE_SPLIT:-1}"
ENABLE_TEMP_SCALING="${ENABLE_TEMP_SCALING:-0}"
ENABLE_DIFFICULTY_THRESHOLDS="${ENABLE_DIFFICULTY_THRESHOLDS:-0}"
DIFFICULTY_THRESHOLD_MIN_SAMPLES="${DIFFICULTY_THRESHOLD_MIN_SAMPLES:-100}"
FIT_THRESHOLDS_ON_EVAL_SPLIT="${FIT_THRESHOLDS_ON_EVAL_SPLIT:-0}"
OOD_FIT_THRESHOLDS_ON_VAL="${OOD_FIT_THRESHOLDS_ON_VAL:-1}"
OOD_USE_ID_THRESHOLD_SEED="${OOD_USE_ID_THRESHOLD_SEED:-0}"
ENABLE_POSTHOC="${ENABLE_POSTHOC:-1}"
POSTHOC_METHOD="${POSTHOC_METHOD:-reasoning_logistic_blend}"
POSTHOC_METHODS="${POSTHOC_METHODS:-reasoning_logistic_isotonic reasoning_logistic_blend platt_base temperature_scaling isotonic_regression binwise_hybrid}"
POSTHOC_BASELINE_METHOD="${POSTHOC_BASELINE_METHOD:-reasoning_logistic_blend}"
POSTHOC_TUNE_SPLIT="${POSTHOC_TUNE_SPLIT:-validation}"
POSTHOC_FEATURE_MODE="${POSTHOC_FEATURE_MODE:-auto}"
POSTHOC_MIN_SAMPLES_FOR_FULL="${POSTHOC_MIN_SAMPLES_FOR_FULL:-1500}"

LOG_LEVEL="${LOG_LEVEL:-INFO}"
LOG_FORMAT="${LOG_FORMAT:-%(asctime)s [%(levelname)s] %(message)s}"
LOG_DATEFMT="${LOG_DATEFMT:-%Y-%m-%d %H:%M:%S}"
LOG_BANNER_WIDTH="${LOG_BANNER_WIDTH:-72}"

setup_environment() {
    export HF_TOKEN="${HF_TOKEN:-}"
    export VLLM_LOGGING_LEVEL=ERROR
    if [[ -z "${HF_TOKEN:-}" ]]; then
        echo "[WARN] HF_TOKEN not set. Set it in environment for gated model downloads."
    fi

    export XDG_RUNTIME_DIR="/tmp/runtime-${USER:-$(whoami)}"
    export HF_HOME="${HF_CACHE}"
    export DISABLE_TQDM=1
    export PYTHONUNBUFFERED=1

    export USE_TF=0
    export USE_FLAX=0
    export TRANSFORMERS_NO_TF=1
    export TRANSFORMERS_NO_FLAX=1
    export TRANSFORMERS_NO_JAX=1

    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND}"
    export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION}"
    export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN}"
    export VLLM_WORKER_MULTIPROC_METHOD="spawn"
    export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

    export TORCHINDUCTOR_CACHE_DIR="${POPLLM_ROOT}/.cache/torchinductor"
    export TRITON_CACHE_DIR="${POPLLM_ROOT}/.cache/triton"


    export MODEL_NAME="${MODEL_NAME}"
    export DATASET_NAME="${DATASET_NAME}"
    export JUDGE_MODEL_NAME="${JUDGE_MODEL_NAME}"
    export GEN_BATCH_SIZE="${GEN_BATCH_SIZE}"
    export GEN_OOD_BATCH_SIZE="${GEN_OOD_BATCH_SIZE}"
    export JUDGE_BATCH_SIZE="${JUDGE_BATCH_SIZE}"
    export GEN_MAX_NEW_TOKENS="${GEN_MAX_NEW_TOKENS}"
    export GEN_PROMPT_MAX_TOKENS="${GEN_PROMPT_MAX_TOKENS}"
    export JUDGE_PROMPT_MAX_TOKENS="${JUDGE_PROMPT_MAX_TOKENS}"
    export BACKEND="${BACKEND}"
    export MEM_TRAIN_BUDGET_RATIO="${MEM_TRAIN_BUDGET_RATIO}"
    export MEM_VAL_BUDGET_RATIO="${MEM_VAL_BUDGET_RATIO}"
    export MEM_EVAL_BUDGET_RATIO="${MEM_EVAL_BUDGET_RATIO}"
    export CACHE_TRAIN_MEM_BUDGET_GB="${CACHE_TRAIN_MEM_BUDGET_GB:-$(_resolve_cache_split_budget_gb "${MEM_TRAIN_BUDGET_RATIO}")}"
    export CACHE_VAL_MEM_BUDGET_GB="${CACHE_VAL_MEM_BUDGET_GB:-$(_resolve_cache_split_budget_gb "${MEM_VAL_BUDGET_RATIO}")}"
    export CACHE_EVAL_MEM_BUDGET_GB="${CACHE_EVAL_MEM_BUDGET_GB:-$(_resolve_cache_split_budget_gb "${MEM_EVAL_BUDGET_RATIO}")}"

    export TRAIN_EPOCHS="${TRAIN_EPOCHS}"
    export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
    export TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE}"
    export RETRAIN="${RETRAIN}"
    export LOSS_TYPE="${LOSS_TYPE}"
    export LOSS_POS_WEIGHT="${LOSS_POS_WEIGHT}"
    export FOCAL_GAMMA="${FOCAL_GAMMA}"
    export SAMPLE_POS_WEIGHT="${SAMPLE_POS_WEIGHT}"
    export LABEL_SMOOTHING="${LABEL_SMOOTHING}"
    export WEIGHT_DECAY="${WEIGHT_DECAY}"
    export WARMUP_RATIO="${WARMUP_RATIO}"
    export EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE}"
    export TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM}"
    export REPORT_TO="${REPORT_TO}"
    export WANDB_PROJECT="${WANDB_PROJECT}"
    export WANDB_ENTITY="${WANDB_ENTITY}"
    export SUMMARY_HEAD_UNIVERSE="${SUMMARY_HEAD_UNIVERSE:-${HEAD_UNIVERSE}}"
    export VIS_HEAD_UNIVERSE="${VIS_HEAD_UNIVERSE:-${HEAD_UNIVERSE}}"
    export LOG_LEVEL="${LOG_LEVEL}"
    export LOG_FORMAT="${LOG_FORMAT}"
    export LOG_DATEFMT="${LOG_DATEFMT}"
    export LOG_BANNER_WIDTH="${LOG_BANNER_WIDTH}"
    export EVAL_THRESHOLD="${EVAL_THRESHOLD}"
    export FORCE_EVAL="${FORCE_EVAL}"
    export THRESHOLD_TUNE_SPLIT="${THRESHOLD_TUNE_SPLIT}"
    export OOD_THRESHOLD_TUNE_SPLIT="${OOD_THRESHOLD_TUNE_SPLIT}"
    export REQUIRE_TUNE_SPLIT="${REQUIRE_TUNE_SPLIT}"
    export ENABLE_TEMP_SCALING="${ENABLE_TEMP_SCALING}"
    export ENABLE_DIFFICULTY_THRESHOLDS="${ENABLE_DIFFICULTY_THRESHOLDS}"
    export DIFFICULTY_THRESHOLD_MIN_SAMPLES="${DIFFICULTY_THRESHOLD_MIN_SAMPLES}"
    export FIT_THRESHOLDS_ON_EVAL_SPLIT="${FIT_THRESHOLDS_ON_EVAL_SPLIT}"
    export ENABLE_POSTHOC="${ENABLE_POSTHOC}"
    export POSTHOC_METHOD="${POSTHOC_METHOD}"
    export POSTHOC_METHODS="${POSTHOC_METHODS}"
    export POSTHOC_BASELINE_METHOD="${POSTHOC_BASELINE_METHOD}"
    export POSTHOC_TUNE_SPLIT="${POSTHOC_TUNE_SPLIT}"
    export POSTHOC_FEATURE_MODE="${POSTHOC_FEATURE_MODE}"
    export POSTHOC_MIN_SAMPLES_FOR_FULL="${POSTHOC_MIN_SAMPLES_FOR_FULL}"
    export FEATURE_HIDDEN_STATE_LAYERS="${FEATURE_HIDDEN_STATE_LAYERS}"
    export FEATURE_HIDDEN_STATE_FUSION="${FEATURE_HIDDEN_STATE_FUSION}"
    export FEATURE_HIDDEN_STATE_WEIGHTS="${FEATURE_HIDDEN_STATE_WEIGHTS}"
    export FEATURE_TOP_N_PROBS="${FEATURE_TOP_N_PROBS}"
    export FEATURE_TOKEN_APPEND_STATS="${FEATURE_TOKEN_APPEND_STATS}"
    export FEATURE_ATTENTION_LAYERS="${FEATURE_ATTENTION_LAYERS}"
    export FEATURE_ATTENTION_HEADS="${FEATURE_ATTENTION_HEADS}"
    export FEATURE_ATTN_HISTORY_SZ="${FEATURE_ATTN_HISTORY_SZ}"
    export FEATURE_POOL_ATTENTION_LAYERS="${FEATURE_POOL_ATTENTION_LAYERS}"

    VLLM_COMPILE_CACHE="${HOME}/.cache/vllm/torch_compile_cache"
    CACHE_STAMP="${TORCHINDUCTOR_CACHE_DIR}/.vllm_cache_initialized"
    mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" "${TRITON_CACHE_DIR}"
    if [[ ! -f "${CACHE_STAMP}" ]]; then
        echo "[cache] First run — clearing stale vLLM compile cache..."
        rm -rf "${VLLM_COMPILE_CACHE}"
        touch "${CACHE_STAMP}"
        echo "[cache] Done."
    fi

    mkdir -p "${RESULTS_ROOT}" "${LOGS_ROOT}" "${HF_CACHE}" \
             "${MODELS_ROOT}" "${DATASETS_ROOT}" "${CACHE_DIR}" \
             "$(results_summary_root "${RESULTS_ROOT}")" \
             "$(logs_summary_root)" "$(logs_figure_eval_root)" "$(logs_figure_cal_root)" \
             "$(phase_logs_root "phase1")" "$(phase_logs_root "phase2")" \
             "$(phase_logs_root "phase3")" "$(phase_logs_root "phase4")"

    module load cuda/12.8.1 2>/dev/null || true
    module load gcc 2>/dev/null || true
    module load conda 2>/dev/null || true
    conda activate "${CONDA_ENV:-llm}" 2>/dev/null || true

    if command -v python >/dev/null 2>&1; then
        export PYTHON_BIN="python"
    elif command -v python3 >/dev/null 2>&1; then
        export PYTHON_BIN="python3"
    else
        echo "[ERROR] Neither python nor python3 found in PATH."
        return 127
    fi

    cd "${PROJECT_DIR}" || return 1
}

detect_gpus() {
    if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
        IFS=',' read -ra GPU_IDS <<< "${CUDA_VISIBLE_DEVICES}"
        NUM_GPUS=${#GPU_IDS[@]}
    else
        NUM_GPUS=$( (nvidia-smi -L 2>/dev/null || true) | wc -l )
        NUM_GPUS=${NUM_GPUS//[[:space:]]/}
        if [ "${NUM_GPUS}" -gt 0 ]; then
            mapfile -t GPU_IDS < <(seq 0 $((NUM_GPUS - 1)))
        else
            GPU_IDS=(0)
        fi
    fi
    echo "Detected GPUs: [${GPU_IDS[*]}] (total: ${NUM_GPUS})"

    if [ "${NUM_GPUS}" -le 0 ]; then
        echo "[WARN] No visible GPUs; falling back to CPU-compatible single-worker mode."
    elif [ "${NUM_GPUS}" -gt 1 ]; then
        echo "Multi-GPU environment available."
    else
        echo "Single GPU environment."
    fi
}


get_gpu_memory_util() {
    local dataset="${1:-${DATASET_NAME}}"
    for entry in ${GPU_MEM_UTIL_OVERRIDES:-}; do
        local ds="${entry%%=*}"
        local val="${entry#*=}"
        if [[ "${ds}" == "${dataset}" ]]; then
            echo "${val}"
            return 0
        fi
    done
    echo "${VLLM_GPU_MEMORY_UTILIZATION}"
}

_resolve_cache_split_budget_gb() {


    local ratio="${1:-0.4}"
    local mem_mb="${SLURM_MEM_PER_NODE:-}"

    if ! [[ "${ratio}" =~ ^[0-9]*\.?[0-9]+$ ]]; then
        ratio="0.4"
    fi

    if ! [[ "${mem_mb}" =~ ^[0-9]+$ ]] || [[ "${mem_mb}" -le 0 ]]; then
        if [[ -r /proc/meminfo ]]; then
            mem_mb="$(awk '/MemTotal:/ {printf "%.0f", $2 / 1024.0}' /proc/meminfo)"
        fi
    fi

    if ! [[ "${mem_mb}" =~ ^[0-9]+$ ]] || [[ "${mem_mb}" -le 0 ]]; then

        awk -v ratio="${ratio}" 'BEGIN { printf "%.1f", 250.0 * ratio }'
        return 0
    fi

    awk -v mem_mb="${mem_mb}" -v ratio="${ratio}" 'BEGIN {
        gb = (mem_mb / 1024.0) * ratio
        if (gb < 1.0) gb = 1.0
        printf "%.1f", gb
    }'
}

print_header() {
    local title="$1"
    echo ""
    echo "###################################################################"
    echo "#  ${title}"
    echo "###################################################################"
    echo ""
}

print_step() {
    local step_name="$1"
    echo "============================================================"
    echo "${step_name}"
    echo "============================================================"
    echo ""
}

log_gpu_snapshot() {
    local tag="${1:-snapshot}"
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "[gpu][${tag}] nvidia-smi not found"
        return 0
    fi
    echo "[gpu][${tag}] $(date '+%F %T')"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
        --format=csv,noheader,nounits || true
}

format_duration() {
    local seconds=$1
    local minutes=$((seconds / 60))
    local hours=$((minutes / 60))
    minutes=$((minutes % 60))
    seconds=$((seconds % 60))
    if [ ${hours} -gt 0 ]; then
        echo "${hours}h ${minutes}m"
    elif [ ${minutes} -gt 0 ]; then
        echo "${minutes}m ${seconds}s"
    else
        echo "${seconds}s"
    fi
}

cache_dir_for_dataset() {
    local dataset="${1:-${DATASET_NAME}}"
    if [[ -n "${CACHE_SUBDIR:-}" ]]; then
        echo "${POPLLM_ROOT}/cached_features/${CACHE_SUBDIR}/${dataset}/${MODEL_NAME}"
    else
        echo "${POPLLM_ROOT}/cached_features/${dataset}/${MODEL_NAME}"
    fi
}

artifact_safe_name() {
    local name="$1"
    name="${name//\//_}"
    name="${name// /_}"
    printf '%s' "${name}"
}

join_artifact_name() {
    local out=""
    local part
    for part in "$@"; do
        [[ -n "${part}" ]] || continue
        part="$(artifact_safe_name "${part}")"
        if [[ -z "${out}" ]]; then
            out="${part}"
        else
            out="${out}_${part}"
        fi
    done
    printf '%s\n' "${out}"
}

variant_artifact_name() {
    local split="$1"
    shift
    local base
    base="$(join_artifact_name "$@")"
    if [[ "${split}" == "validation" ]]; then
        printf '%s\n' "$(join_artifact_name "validation" "${base}")"
    else
        printf '%s\n' "${base}"
    fi
}

phase_results_root() {
    local results_root="$1"
    local phase="$2"
    case "${phase}" in
        phase2) printf '%s\n' "${results_root}/train" ;;
        phase3) printf '%s\n' "${results_root}/eval" ;;
        phase4) printf '%s\n' "${results_root}/cal" ;;
        summary) printf '%s\n' "${results_root}/summary" ;;
        *) printf '%s\n' "${results_root}/${phase}" ;;
    esac
}

phase_logs_root() {
    local phase="$1"
    case "${phase}" in
        phase1) printf '%s\n' "${LOGS_ROOT}/data" ;;
        phase2) printf '%s\n' "${LOGS_ROOT}/train" ;;
        phase3) printf '%s\n' "${LOGS_ROOT}/eval" ;;
        phase4) printf '%s\n' "${LOGS_ROOT}/cal" ;;
        summary) printf '%s\n' "${LOGS_ROOT}/summary" ;;
        *) printf '%s\n' "${LOGS_ROOT}/${phase}" ;;
    esac
}

phase_run_log_path() {
    local phase="$1"
    printf '%s\n' "${LOGS_ROOT}/${phase}.log"
}

phase1_data_log_path() {
    local dataset="$1"
    local scope="${2:-ID}"
    printf '%s\n' "$(phase_logs_root "phase1")/$(artifact_safe_name "${dataset}").log"
}

phase2_train_root() {
    local results_root="$1"
    printf '%s\n' "$(phase_results_root "${results_root}" "phase2")"
}

phase2_head_dir() {
    local results_root="$1"
    local head_type="$2"
    printf '%s\n' "$(phase2_train_root "${results_root}")/${head_type}"
}

phase2_train_log_path() {
    local head_type="$1"
    printf '%s\n' "$(phase_logs_root "phase2")/${head_type}.log"
}

logs_summary_root() {
    printf '%s\n' "${LOGS_ROOT}/summary"
}

logs_figure_root() {
    printf '%s\n' "${LOGS_ROOT}/figure"
}

logs_figure_eval_root() {
    printf '%s\n' "$(logs_figure_root)/eval"
}

logs_figure_cal_root() {
    printf '%s\n' "$(logs_figure_root)/cal"
}

results_summary_root() {
    local results_root="$1"
    printf '%s\n' "${results_root}/summary"
}

phase3_dataset_label() {
    local scope="$1"
    local dataset="${2:-}"
    if [[ "${scope}" == "ID" ]]; then
        printf '%s\n' "${DATASET_NAME}"
    else
        printf '%s\n' "${dataset}"
    fi
}

phase4_dataset_label() {
    local scope="$1"
    local dataset="${2:-}"
    if [[ "${scope}" == "ID" ]]; then
        printf '%s\n' "${DATASET_NAME}"
    else
        printf '%s\n' "${dataset}"
    fi
}

phase4_head_variant_name() {
    local method="$1"
    local head_type="$2"
    local split="${3:-test}"
    printf '%s\n' "$(variant_artifact_name "${split}" "${method}" "${head_type}")"
}

phase4_figure_prefix() {
    local method="$1"
    local scope="$2"
    local dataset="${3:-}"
    printf '%s\n' "$(join_artifact_name "${method}" "$(phase4_dataset_label "${scope}" "${dataset}")")"
}

phase3_head_variant_name() {
    local head_type="$1"
    local split="${2:-test}"
    printf '%s\n' "$(variant_artifact_name "${split}" "${head_type}")"
}

phase3_figure_prefix() {
    local scope="$1"
    local dataset="${2:-}"
    printf '%s\n' "$(artifact_safe_name "$(phase3_dataset_label "${scope}" "${dataset}")")"
}

phase3_results_root() {
    local results_root="$1"
    printf '%s\n' "$(phase_results_root "${results_root}" "phase3")"
}

phase3_id_results_root() {
    local results_root="$1"
    printf '%s\n' "$(phase3_results_root "${results_root}")/$(artifact_safe_name "${DATASET_NAME}")"
}

phase3_id_head_dir() {
    local results_root="$1"
    local head_type="$2"
    printf '%s\n' "$(phase3_id_results_root "${results_root}")/${head_type}"
}

phase3_id_baselines_dir() {
    local results_root="$1"
    printf '%s\n' "$(phase3_id_results_root "${results_root}")/baselines"
}

phase3_ood_dataset_root() {
    local results_root="$1"
    local dataset="$2"
    printf '%s\n' "$(phase3_results_root "${results_root}")/$(artifact_safe_name "${dataset}")"
}

phase3_ood_split_root() {
    local results_root="$1"
    local dataset="$2"
    local split="$3"
    printf '%s\n' "$(phase3_ood_dataset_root "${results_root}" "${dataset}")"
}

phase3_ood_head_dir() {
    local results_root="$1"
    local dataset="$2"
    local split="$3"
    local head_type="$4"
    printf '%s\n' "$(phase3_ood_split_root "${results_root}" "${dataset}" "${split}")/$(phase3_head_variant_name "${head_type}" "${split}")"
}

phase3_ood_baselines_dir() {
    local results_root="$1"
    local dataset="$2"
    printf '%s\n' "$(phase3_ood_dataset_root "${results_root}" "${dataset}")/baselines"
}

phase3_eval_log_path() {
    local scope="$1"
    local dataset="$2"
    local split="$3"
    local head_type="$4"
    local label
    label="$(phase3_dataset_label "${scope}" "${dataset}")"
    printf '%s\n' "$(phase_logs_root "phase3")/$(artifact_safe_name "${label}")/$(phase3_head_variant_name "${head_type}" "${split}").log"
}

phase3_baselines_log_path() {
    local scope="$1"
    local dataset="${2:-}"
    local label
    label="$(phase3_dataset_label "${scope}" "${dataset}")"
    printf '%s\n' "$(phase_logs_root "phase3")/$(artifact_safe_name "${label}")/baselines.log"
}

phase3_result_log_path() {
    local scope="$1"
    local dataset="${2:-}"
    local label
    label="$(phase3_dataset_label "${scope}" "${dataset}")"
    printf '%s\n' "$(logs_summary_root)/$(join_artifact_name "eval" "${label}").txt"
}

phase3_figure_dir() {
    local scope="$1"
    local dataset="${2:-}"
    printf '%s\n' "$(logs_figure_eval_root)"
}

phase3_summary_json_path() {
    local results_root="$1"
    local scope="$2"
    local dataset="${3:-}"
    local label
    label="$(phase3_dataset_label "${scope}" "${dataset}")"
    printf '%s\n' "$(results_summary_root "${results_root}")/$(join_artifact_name "eval" "${label}").json"
}

phase3_manifest_path() {
    local results_root="$1"
    local scope="$2"
    local dataset="${3:-}"
    local label
    label="$(phase3_dataset_label "${scope}" "${dataset}")"
    printf '%s\n' "$(phase3_results_root "${results_root}")/$(artifact_safe_name "${label}")/_complete.json"
}

phase4_root() {
    local results_root="$1"
    printf '%s\n' "$(phase_results_root "${results_root}" "phase4")"
}

phase4_method_results_root() {
    local results_root="$1"
    local method="$2"
    printf '%s\n' "$(phase4_root "${results_root}")"
}

phase4_id_results_root() {
    local results_root="$1"
    local method="$2"
    printf '%s\n' "$(phase4_method_results_root "${results_root}" "${method}")/$(artifact_safe_name "${DATASET_NAME}")"
}

phase4_id_head_dir() {
    local results_root="$1"
    local method="$2"
    local head_type="$3"
    printf '%s\n' "$(phase4_id_results_root "${results_root}" "${method}")/$(phase4_head_variant_name "${method}" "${head_type}" "test")"
}

phase4_ood_dataset_root() {
    local results_root="$1"
    local method="$2"
    local dataset="$3"
    printf '%s\n' "$(phase4_method_results_root "${results_root}" "${method}")/$(artifact_safe_name "${dataset}")"
}

phase4_ood_split_root() {
    local results_root="$1"
    local method="$2"
    local dataset="$3"
    local split="$4"
    printf '%s\n' "$(phase4_ood_dataset_root "${results_root}" "${method}" "${dataset}")"
}

phase4_ood_head_dir() {
    local results_root="$1"
    local method="$2"
    local dataset="$3"
    local split="$4"
    local head_type="$5"
    printf '%s\n' "$(phase4_ood_split_root "${results_root}" "${method}" "${dataset}" "${split}")/$(phase4_head_variant_name "${method}" "${head_type}" "${split}")"
}

phase4_prediction_cache_root() {
    local results_root="$1"
    printf '%s\n' "$(phase4_root "${results_root}")/cache"
}

phase4_prediction_cache_dir() {
    local results_root="$1"
    local scope="$2"
    local head_type="$3"
    local dataset="${4:-}"
    local label
    label="$(phase4_dataset_label "${scope}" "${dataset}")"
    printf '%s\n' "$(phase4_prediction_cache_root "${results_root}")/$(artifact_safe_name "${label}")/${head_type}"
}

phase4_eval_log_path() {
    local method="$1"
    local scope="$2"
    local dataset="$3"
    local split="$4"
    local head_type="$5"
    local label
    label="$(phase4_dataset_label "${scope}" "${dataset}")"
    printf '%s\n' "$(phase_logs_root "phase4")/$(artifact_safe_name "${label}")/$(phase4_head_variant_name "${method}" "${head_type}" "${split}").log"
}

phase4_result_log_path() {
    local method="$1"
    local scope="$2"
    local dataset="${3:-}"
    local label
    label="$(phase4_dataset_label "${scope}" "${dataset}")"
    printf '%s\n' "$(logs_summary_root)/$(join_artifact_name "cal" "${method}" "${label}").txt"
}

phase4_figure_dir() {
    local method="$1"
    local scope="$2"
    local dataset="${3:-}"
    printf '%s\n' "$(logs_figure_cal_root)"
}

phase4_summary_json_path() {
    local results_root="$1"
    local method="$2"
    local scope="$3"
    local dataset="${4:-}"
    local label
    label="$(phase4_dataset_label "${scope}" "${dataset}")"
    printf '%s\n' "$(results_summary_root "${results_root}")/$(join_artifact_name "cal" "${method}" "${label}").json"
}

phase4_manifest_path() {
    local results_root="$1"
    local method="$2"
    local scope="$3"
    local dataset="${4:-}"
    local label
    label="$(phase4_dataset_label "${scope}" "${dataset}")"
    printf '%s\n' "$(phase4_root "${results_root}")/$(artifact_safe_name "${label}")/$(join_artifact_name "${method}" "complete").json"
}

phase4_head_status_path() {
    local output_dir="$1"
    printf '%s\n' "${output_dir}/posthoc_status.json"
}

phase4_head_artifact_files() {
    local output_dir="$1"
    local split="${2:-test}"
    local -a files=(
        "${output_dir}/evaluation_report.json"
        "${output_dir}/diagnostics.json"
        "${output_dir}/predictions.json"
        "$(phase4_head_status_path "${output_dir}")"
    )
    if [[ "${split}" == "test" ]]; then
        files+=("${output_dir}/posthoc_calibrator.json")
    fi
    printf '%s\n' "${files[@]}"
}

remove_path_if_exists() {
    local path="$1"
    if [[ -e "${path}" ]]; then
        rm -rf "${path}"
        echo "  [CLEAN] Removed ${path}"
    fi
}

manifest_path() {
    local cache_dir="$1"
    local split="$2"
    echo "${cache_dir}/${split}/manifest.json"
}

manifest_query() {
    local cache_dir="$1"
    local split="$2"
    local query="$3"
    local manifest
    manifest="$(manifest_path "${cache_dir}" "${split}")"

    [[ -f "${manifest}" ]] || return 1

    "${PYTHON_BIN:-python3}" - "${manifest}" "${query}" <<'PY' 2>/dev/null
import json
import sys

manifest_path, query = sys.argv[1:3]

try:
    with open(manifest_path, "r", encoding="utf-8") as handle:
        value = json.load(handle)
    for key in query.split("."):
        if not isinstance(value, dict) or key not in value:
            raise KeyError(key)
        value = value[key]
except Exception:
    sys.exit(1)

if isinstance(value, bool):
    sys.stdout.write("1" if value else "0")
elif value is None:
    sys.exit(1)
else:
    sys.stdout.write(str(value))
PY
}

manifest_phase_complete() {



    local cache_dir="$1"
    local split="$2"
    local phase="$3"
    local status
    status=$(manifest_query "${cache_dir}" "${split}" "phase_status.${phase}.status") || return 1
    [[ "${status}" == "complete" ]]
}

report_file_valid() {


    local report_path="$1"
    [[ -f "${report_path}" ]] || return 1

    "${PYTHON_BIN:-python3}" - "${report_path}" <<'PY' >/dev/null 2>&1
import json
import sys

path = sys.argv[1]
try:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
except Exception:
    sys.exit(1)

if isinstance(payload, dict) and payload:
    sys.exit(0)
if isinstance(payload, list):
    sys.exit(0)
sys.exit(1)
PY
}

write_completion_manifest() {
    local manifest_path="$1"
    local phase="$2"
    local scope="$3"
    local dataset="$4"
    local method="$5"
    local heads_csv="$6"
    shift 6

    mkdir -p "$(dirname "${manifest_path}")"
    "${PYTHON_BIN:-python3}" - "${manifest_path}" "${phase}" "${scope}" "${dataset}" "${method}" "${heads_csv}" "$@" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
phase = sys.argv[2]
scope = sys.argv[3]
dataset = sys.argv[4] or None
method = sys.argv[5] or None
heads = [token for token in sys.argv[6].split() if token]
files = [str(Path(path)) for path in sys.argv[7:] if path]

payload = {
    "status": "complete",
    "phase": phase,
    "scope": scope,
    "dataset": dataset,
    "method": method,
    "heads": heads,
    "files": files,
}
manifest_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

completion_manifest_valid() {
    local manifest_path="$1"
    local phase="$2"
    local scope="$3"
    local dataset="$4"
    local method="$5"
    local heads_csv="$6"

    [[ -f "${manifest_path}" ]] || return 1

    "${PYTHON_BIN:-python3}" - "${manifest_path}" "${phase}" "${scope}" "${dataset}" "${method}" "${heads_csv}" <<'PY' >/dev/null 2>&1
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
phase = sys.argv[2]
scope = sys.argv[3]
dataset = sys.argv[4] or None
method = sys.argv[5] or None
heads = [token for token in sys.argv[6].split() if token]

try:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)

if payload.get("status") != "complete":
    sys.exit(1)
if payload.get("phase") != phase or payload.get("scope") != scope:
    sys.exit(1)
if payload.get("dataset") != dataset or payload.get("method") != method:
    sys.exit(1)
if payload.get("heads") != heads:
    sys.exit(1)

for path_str in payload.get("files", []):
    if not Path(path_str).exists():
        sys.exit(1)

sys.exit(0)
PY
}

cache_split_ready() {
    local cache_dir="$1"
    local split="$2"
    local split_dir="${cache_dir}/${split}"

    local expected_chunks
    expected_chunks=$(manifest_query "${cache_dir}" "${split}" "num_chunks") || return 1
    if [[ -z "${expected_chunks}" || "${expected_chunks}" -le 0 ]]; then
        return 1
    fi

    local existing_chunks
    existing_chunks=$(find "${split_dir}" -maxdepth 1 -type f -name 'chunk_*.pt' ! -name '*_reasoning.pt' 2>/dev/null | wc -l)
    [[ "${existing_chunks}" -ge "${expected_chunks}" ]]
}

split_reasoning_sidecar_ready() {
    local cache_dir="$1"
    local split="$2"
    local split_dir="${cache_dir}/${split}"

    local expected_chunks
    expected_chunks=$(manifest_query "${cache_dir}" "${split}" "num_chunks") || return 1
    if [[ -z "${expected_chunks}" || "${expected_chunks}" -le 0 ]]; then
        return 1
    fi

    local sidecar_chunks
    sidecar_chunks=$(find "${split_dir}" -maxdepth 1 -type f -name 'chunk_*_reasoning.pt' 2>/dev/null | wc -l)
    [[ "${sidecar_chunks}" -ge "${expected_chunks}" ]]
}

cache_splits_ready() {
    local cache_dir="$1"
    local splits_csv="$2"
    local split
    local splits=()
    IFS=',' read -ra splits <<< "${splits_csv}"

    for split in "${splits[@]}"; do
        split="${split// /}"
        [[ -n "${split}" ]] || continue
        if ! cache_split_ready "${cache_dir}" "${split}"; then
            return 1
        fi
    done
    return 0
}

manifest_total_pending() {
    local cache_dir="$1"
    local split="$2"

    local pending
    pending=$(manifest_query "${cache_dir}" "${split}" "sample_pending") || return 1
    [[ -n "${pending}" ]] || return 1
    echo "${pending}"
}

manifest_total_claim_pending() {
    local cache_dir="$1"
    local split="$2"

    local pending
    pending=$(manifest_query "${cache_dir}" "${split}" "total_claim_pending") || return 1
    [[ -n "${pending}" ]] || return 1
    echo "${pending}"
}

should_run_judge_for_splits() {
    local cache_dir="$1"
    local splits_csv="$2"
    local context="${3:-Judge}"
    local ignore_force_judge="${4:-0}"

    if [[ "${ignore_force_judge}" != "1" && "${FORCE_JUDGE:-0}" == "1" ]]; then
        echo "  [JUDGE] ${context}: FORCE_JUDGE=1, will run judge regardless."
        return 0
    fi

    local split
    local splits=()
    local considered=0
    IFS=',' read -ra splits <<< "${splits_csv}"
    for split in "${splits[@]}"; do
        split="${split// /}"
        [[ -n "${split}" ]] || continue
        considered=1


        if manifest_phase_complete "${cache_dir}" "${split}" "judge"; then
            local claim_pending
            if claim_pending=$(manifest_total_claim_pending "${cache_dir}" "${split}") && [[ "${claim_pending}" -eq 0 ]]; then
                echo "  [JUDGE] ${context}/${split}: judge phase complete, claim_pending=0."
                continue
            fi
        fi

        if ! cache_split_ready "${cache_dir}" "${split}"; then

            local n_chunks
            n_chunks=$(find "${cache_dir}/${split}" -maxdepth 1 -name 'chunk_*.pt' ! -name '*_reasoning.pt' 2>/dev/null | wc -l)
            if [[ "${n_chunks}" -eq 0 ]]; then
                echo "  [JUDGE] ${context}/${split}: no chunk files — generation incomplete, skip."
                continue
            fi
            echo "  [JUDGE] ${context}/${split}: cache not ready, will run judge."
            return 0
        fi

        local claim_pending
        if claim_pending=$(manifest_total_claim_pending "${cache_dir}" "${split}"); then
            if [[ "${claim_pending}" -gt 0 ]]; then
                echo "  [JUDGE] ${context}/${split}: claim_pending=${claim_pending}, will run judge."
                return 0
            fi
        fi

        local pending
        if ! pending=$(manifest_total_pending "${cache_dir}" "${split}"); then
            echo "  [JUDGE] ${context}/${split}: cannot read total_pending, will run judge."
            return 0
        fi

        if [[ "${pending}" -eq 0 ]]; then
            echo "  [JUDGE] ${context}/${split}: pending=0 and claim_pending=0, no judge needed."
            continue
        fi
        echo "  [JUDGE] ${context}/${split}: pending=${pending}, will run judge."
        return 0
    done

    if [[ "${considered}" -eq 0 ]]; then
        echo "  [JUDGE] ${context}: empty split list, will run judge."
        return 0
    fi

    return 1
}

should_run_cleanup_for_splits() {
    local cache_dir="$1"
    local splits_csv="$2"
    local context="${3:-Cleanup}"
    local split
    local splits=()
    local considered=0
    IFS=',' read -ra splits <<< "${splits_csv}"
    for split in "${splits[@]}"; do
        split="${split// /}"
        [[ -n "${split}" ]] || continue
        considered=1

        if ! cache_split_ready "${cache_dir}" "${split}"; then
            echo "  [CLEANUP] ${context}/${split}: cache split not ready yet; will run cleanup later."
            continue
        fi

        local claim_pending
        if ! claim_pending=$(manifest_total_claim_pending "${cache_dir}" "${split}"); then
            echo "  [CLEANUP] ${context}/${split}: cannot read total_claim_pending, will run cleanup."
            return 0
        fi

        if [[ "${claim_pending}" -gt 0 ]]; then
            echo "  [CLEANUP] ${context}/${split}: claim_pending=${claim_pending}, will run cleanup."
            return 0
        fi

        if ! split_reasoning_sidecar_ready "${cache_dir}" "${split}"; then
            echo "  [CLEANUP] ${context}/${split}: reasoning sidecars missing, will run cleanup."
            return 0
        fi

        echo "  [CLEANUP] ${context}/${split}: claim_pending=0 and sidecars ready."
    done

    if [[ "${considered}" -eq 0 ]]; then
        echo "  [CLEANUP] ${context}: empty split list, skip cleanup."
        return 1
    fi
    return 1
}

run_visualization() {
    local results_dir="${1}"
    local figure_dir="${2}"
    local scope="${3:-all}"
    local dataset="${4:-}"
    local compare_results_dir="${5:-}"
    local compare_baseline_label="${6:-}"
    local compare_target_label="${7:-}"
    local file_prefix="${8:-}"
    local calibration_method="${9:-}"
    local compare_calibration_method="${10:-}"
    local -a viz_args=(
        --results_dir "${results_dir}"
        --figure_dir "${figure_dir}"
        --scope "${scope}"
    )
    if [[ -n "${dataset}" ]]; then
        viz_args+=(--dataset "${dataset}")
    fi
    if [[ -n "${compare_results_dir}" ]]; then
        viz_args+=(--compare_results_dir "${compare_results_dir}")
    fi
    if [[ -n "${compare_baseline_label}" ]]; then
        viz_args+=(--compare_baseline_label "${compare_baseline_label}")
    fi
    if [[ -n "${compare_target_label}" ]]; then
        viz_args+=(--compare_target_label "${compare_target_label}")
    fi
    if [[ -n "${file_prefix}" ]]; then
        viz_args+=(--file_prefix "${file_prefix}")
    fi
    if [[ -n "${calibration_method}" ]]; then
        viz_args+=(--calibration_method "${calibration_method}")
    fi
    if [[ -n "${compare_calibration_method}" ]]; then
        viz_args+=(--compare_calibration_method "${compare_calibration_method}")
    fi
    if [[ -n "${VIS_HEAD_UNIVERSE:-}" ]]; then
        viz_args+=(--head_universe "${VIS_HEAD_UNIVERSE}")
    fi
    mkdir -p "${figure_dir}"
    echo ""
    if [[ -n "${dataset}" ]]; then
        echo "=== Generating Figures (${scope}:${dataset}) -> ${figure_dir} ==="
    else
        echo "=== Generating Figures (${scope}) -> ${figure_dir} ==="
    fi
    "${PYTHON_BIN}" "${PROJECT_DIR}/utils/visualize.py" "${viz_args[@]}" \
        || echo "[WARN] Visualization failed (non-fatal)."
}

publish_summary_json() {
    local src="$1"
    local dst="$2"
    [[ -f "${src}" ]] || return 1
    mkdir -p "$(dirname "${dst}")"
    cp -f "${src}" "${dst}"
}

print_split_drop_summary() {
    local cache_dir="$1"
    local split="$2"
    local context="${3:-quality}"

    local total_samples total_claims gen_dropped judge_dropped
    total_samples="$(manifest_query "${cache_dir}" "${split}" "total_samples" 2>/dev/null || echo "NA")"
    total_claims="$(manifest_query "${cache_dir}" "${split}" "total_claims" 2>/dev/null || echo "NA")"
    gen_dropped="$(manifest_query "${cache_dir}" "${split}" "dropped_samples_generation" 2>/dev/null || echo "0")"
    judge_dropped="$(manifest_query "${cache_dir}" "${split}" "judge_dropped_samples" 2>/dev/null || echo "0")"

    echo "  [QUALITY] ${context}/${split}: samples=${total_samples} claims=${total_claims} dropped(gen=${gen_dropped},judge=${judge_dropped})"
}

print_cache_quality_summary() {
    local cache_dir="$1"
    local splits_csv="$2"
    local context="${3:-quality}"
    local split
    local splits=()
    IFS=',' read -ra splits <<< "${splits_csv}"

    echo "  [QUALITY] ${context}: cache=${cache_dir}"
    for split in "${splits[@]}"; do
        split="${split// /}"
        [[ -n "${split}" ]] || continue
        if [[ ! -f "$(manifest_path "${cache_dir}" "${split}")" ]]; then
            echo "  [QUALITY] ${context}/${split}: manifest missing"
            continue
        fi
        print_split_drop_summary "${cache_dir}" "${split}" "${context}"
    done
}
