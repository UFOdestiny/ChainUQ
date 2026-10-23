#!/usr/bin/env bash

#SBATCH --job-name=lla

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=6-23:00:00
#SBATCH --partition=hpg-b200
#SBATCH --gres=gpu:1
#SBATCH --output=%x_%j.log

#sbatch jobs/lla.sh

set -euo pipefail

SCRIPT_DIR="${SCRIPT_DIR:-/jobs}"

ALL_HEAD_TYPES=(
    "uq_v1" "uq_v2" "uq_v3" "uq_abl_v1" "uq_abl_v2"
    "uq_abl_v3" "uq_abl_v4"
    "luh_head" "luh_light" "saplma" "lookback_lens" "factoscope"
)

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
PARALLEL_N="${PARALLEL_N:-3}"
PARALLEL_N_EVAL="${PARALLEL_N_EVAL:-4}"
PIPELINE_LOG_BASENAME="${PIPELINE_LOG_BASENAME:-Llama.log}"

MODEL_NAME="${MODEL_NAME:-Llama-3.1-8B-Instruct}"

source "${SCRIPT_DIR}/pipeline.sh"
