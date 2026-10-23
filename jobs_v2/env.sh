#!/usr/bin/env bash
# Shared environment for every v2 job. Source it, do not execute it.
export CUQ_CODE=/u/kytrio/proj/ChainUQ/code
export CUQ_MODELS_ROOT=/u/kytrio/proj/ChainUQ/models
export CUQ_DATASETS_ROOT=/u/kytrio/proj/ChainUQ/datasets
export CUQ_ARTIFACTS=${CUQ_ARTIFACTS:-/u/kytrio/nvme_bduw/chainuq_artifacts}
export CUQ_PY=/u/kytrio/.conda/envs/llm/bin/python
export CUQ_VLLM=/u/kytrio/.conda/envs/llm/bin/vllm
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export VLLM_NO_USAGE_STATS=1
# GPUs reserved for this project.
export GEN_GPU=${GEN_GPU:-2}
export JUDGE_GPU=${JUDGE_GPU:-3}
mkdir -p "$CUQ_ARTIFACTS/logs"
