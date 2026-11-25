#!/usr/bin/env bash

#SBATCH --job-name=khop-ablation

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220gb
#SBATCH --time=6-23:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-/jobs}"
PROJECT_DIR="${PROJECT_DIR:-}"

CACHE_SUBDIR="${CACHE_SUBDIR:-100k}"
GEN_MAX_TRAIN="${GEN_MAX_TRAIN:-100000}"
GEN_MAX_VAL="${GEN_MAX_VAL:-100000}"
GEN_MAX_TEST="${GEN_MAX_TEST:-100000}"
GEN_MAX_OOD_VAL="${GEN_MAX_OOD_VAL:-100000}"
GEN_MAX_OOD_TEST="${GEN_MAX_OOD_TEST:-100000}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-30}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-1024}"
MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"

HEAD_TYPES="${HEAD_TYPES:-uq_v1 uq_v2 uq_v3 uq_abl_v1 uq_abl_v2 uq_abl_v3 uq_abl_v4}"
PARALLEL_N="${PARALLEL_N:-2}"
PARALLEL_N_EVAL="${PARALLEL_N_EVAL:-2}"
ABLATION_TAG="${ABLATION_TAG:-ablation_${SLURM_JOB_ID:-manual_run}}"

echo "Ablation tag:        ${ABLATION_TAG}"
echo "Model:               ${MODEL_NAME}"
echo "Cache subdir:        ${CACHE_SUBDIR}"
echo "Head set:            ${HEAD_TYPES}"
echo "Parallel:            train=${PARALLEL_N}, eval=${PARALLEL_N_EVAL}"
echo "Post-hoc:            disabled"
echo "Threshold aux:       disabled"
echo ""

cd "${PROJECT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

cmd=(
    "${PYTHON_BIN}" "${PROJECT_DIR}/scripts/ablation.py"
    --tag "${ABLATION_TAG}"
    --script_dir "${SCRIPT_DIR}"
    --parallel_n "${PARALLEL_N}"
    --parallel_n_eval "${PARALLEL_N_EVAL}"
    --head_types "${HEAD_TYPES}"
)

echo "Running: ${cmd[*]}"
"${cmd[@]}"
