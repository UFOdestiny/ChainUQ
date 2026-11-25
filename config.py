"""
config.py - Central configuration for the ChainUQ release package.

All tunable parameters are here. Paths can be overridden via environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

def _get_env(key: str, default: str = "") -> str:
    return os.environ.get(key, default)


def _get_env_int(key: str, default: int) -> int:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _get_env_float(key: str, default: float) -> float:
    val = os.environ.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _get_env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _get_env_int_csv(key: str) -> Optional[List[int]]:
    val = os.environ.get(key)
    if val is None or not val.strip():
        return None
    return [int(part.strip()) for part in val.split(",") if part.strip()]

PROJECT_DIR = _get_env("PROJECT_DIR", str(Path(__file__).resolve().parent))
ARTIFACTS_ROOT = _get_env("ARTIFACTS_ROOT", str(Path(PROJECT_DIR) / "artifacts"))

MODELS_ROOT = _get_env("MODELS_ROOT", f"{ARTIFACTS_ROOT}/models")
DATASETS_ROOT = _get_env("DATASETS_ROOT", f"{ARTIFACTS_ROOT}/datasets")
RESULTS_ROOT = _get_env("RESULTS_ROOT", f"{ARTIFACTS_ROOT}/results")
LOGS_ROOT = _get_env("LOGS_ROOT", f"{ARTIFACTS_ROOT}/logs")
HF_CACHE = _get_env("HF_CACHE", f"{MODELS_ROOT}/.hf_cache")
HF_TOKEN = _get_env("HF_TOKEN", "")

GLOBAL_SEED: int = 2026

@dataclass
class PathConfig:
    models_root: str = field(default_factory=lambda: MODELS_ROOT)
    datasets_root: str = field(default_factory=lambda: DATASETS_ROOT)
    results_root: str = field(default_factory=lambda: RESULTS_ROOT)
    logs_root: str = field(default_factory=lambda: LOGS_ROOT)
    hf_cache_dir: Optional[str] = field(default_factory=lambda: HF_CACHE)

@dataclass
class ModelConfig:
    pretrained_model_name_or_path: str = field(
        default_factory=lambda: f"{MODELS_ROOT}/{_get_env('MODEL_NAME', 'Llama-3.1-8B-Instruct')}"
    )
    device_map: str = "cuda"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True
    hf_cache_dir: Optional[str] = field(default_factory=lambda: HF_CACHE)
    hf_token: Optional[str] = field(default_factory=lambda: HF_TOKEN or None)
    model_max_length: int = 4096
    padding_side: str = "left"

@dataclass
class FeatureConfig:
    """Cached per-token features for Phase 1 → Phase 2 heads.

    Defaults favor **last-layer hidden states + top-k log-probs + single-layer
    attention** (lower Phase 1 GPU/time vs. multi-layer attention) while keeping
    enough signal for claim-level UQ. Override via ``FEATURE_*`` env vars.

    ``temperature`` applies only to **logit softmax inside TokenProbExtractor**,
    not vLLM decoding temperature (see ``GenerationConfig.temperature``).
    """

    hidden_state_layers: str = field(
        default_factory=lambda: _get_env("FEATURE_HIDDEN_STATE_LAYERS", "-1,-2,-4,-8")
    )
    hidden_state_fusion: str = field(
        default_factory=lambda: _get_env("FEATURE_HIDDEN_STATE_FUSION", "weighted_sum")
    )
    # Optional comma-separated per-layer fusion weights; when empty, use uniform.
    hidden_state_weights: str = field(
        default_factory=lambda: _get_env("FEATURE_HIDDEN_STATE_WEIGHTS", "0.4,0.3,0.2,0.1")
    )
    top_n_probs: int = field(
        default_factory=lambda: _get_env_int("FEATURE_TOP_N_PROBS", 6)
    )
    token_append_stats: bool = field(
        default_factory=lambda: _get_env_bool("FEATURE_TOKEN_APPEND_STATS", True)
    )
    temperature: float = field(
        default_factory=lambda: _get_env_float("FEATURE_LOGIT_TEMPERATURE", 1.0)
    )
    # Selective attention: only listed layers use eager attention in the HF pass
    # (rest keep FlashAttention). Use "" to disable attention features entirely.
    # Default one layer (-1): ~half the attention cost of "-1,-2" with similar
    # downstream quality when combined with hidden + log-probs.
    attention_layers: str = field(
        default_factory=lambda: _get_env("FEATURE_ATTENTION_LAYERS", "-1")
    )
    attention_heads: str = field(
        default_factory=lambda: _get_env("FEATURE_ATTENTION_HEADS", "all")
    )
    attn_history_sz: int = field(
        default_factory=lambda: _get_env_int("FEATURE_ATTN_HISTORY_SZ", 3)
    )
    # When multiple attention layers are listed, pool across layers (amax) to
    # limit feature width and stabilize cross-layer signal.
    pool_attention_layers: bool = field(
        default_factory=lambda: _get_env_bool("FEATURE_POOL_ATTENTION_LAYERS", True)
    )

@dataclass
class HeadConfig:
    head_type: str = field(default_factory=lambda: _get_env("HEAD_TYPE", "chainuq"))
    num_classes: int = 1  # Binary: 1=consistent/non-hallucination, 0=inconsistent/hallucination
    head_dim: int = 512
    n_layers: int = 2
    n_heads: int = 8
    dropout: float = 0.1
    target_params_m: float = field(
        default_factory=lambda: _get_env_float("TARGET_HEAD_PARAMS_M", _get_env_float("UQ_TARGET_PARAMS_M", 2.0))
    )
    head_dim_search_min: int = field(default_factory=lambda: _get_env_int("HEAD_DIM_SEARCH_MIN", 96))
    head_dim_search_max: int = field(default_factory=lambda: _get_env_int("HEAD_DIM_SEARCH_MAX", 640))
    head_dim_search_step: int = field(default_factory=lambda: _get_env_int("HEAD_DIM_SEARCH_STEP", 8))

@dataclass
class DatasetConfig:
    granularity: str = "claim"
    dataset_name: str = field(default_factory=lambda: _get_env("DATASET_NAME", "stepgame").lower())
    dataset_path: Optional[str] = field(default_factory=lambda: None)  # None → each loader uses its own default
    max_train_samples: int = 0
    max_eval_samples: int = 0
    hf_cache_dir: Optional[str] = field(default_factory=lambda: HF_CACHE)
    # Reasoning depth (number of hops in multi-hop chain)
    n_hop_values: Optional[List[int]] = field(default_factory=lambda: _get_env_int_csv("N_HOP_VALUES"))

@dataclass
class TrainingConfig:
    num_epochs: int = field(default_factory=lambda: _get_env_int("TRAIN_EPOCHS", 40))
    learning_rate: float = field(default_factory=lambda: _get_env_float("TRAIN_LEARNING_RATE", 1.5e-4))
    warmup_steps: int = field(default_factory=lambda: _get_env_int("WARMUP_STEPS", 0))
    warmup_ratio: float = field(default_factory=lambda: _get_env_float("WARMUP_RATIO", 0.06))
    weight_decay: float = field(default_factory=lambda: _get_env_float("WEIGHT_DECAY", 0.05))
    per_device_train_batch_size: int = field(default_factory=lambda: _get_env_int("TRAIN_BATCH_SIZE", 2048))
    per_device_eval_batch_size: int = field(default_factory=lambda: _get_env_int("TRAIN_BATCH_SIZE", 2048))
    gradient_accumulation_steps: int = field(default_factory=lambda: _get_env_int("TRAIN_GRAD_ACCUM", 1))
    max_grad_norm: float = 1.0
    max_auto_pos_weight: float = field(default_factory=lambda: _get_env_float("MAX_AUTO_POS_WEIGHT", 3.0))
    loss_type: str = field(default_factory=lambda: _get_env("LOSS_TYPE", "balanced_bce"))
    loss_pos_weight: float = field(default_factory=lambda: _get_env_float("LOSS_POS_WEIGHT", -1.0))
    focal_gamma: float = field(default_factory=lambda: _get_env_float("FOCAL_GAMMA", 2.0))
    label_smoothing: float = field(default_factory=lambda: _get_env_float("LABEL_SMOOTHING", 0.02))
    sample_pos_weight: float = field(default_factory=lambda: _get_env_float("SAMPLE_POS_WEIGHT", -1.0))
    lr_scheduler_type: str = "cosine"
    fp16: bool = False
    bf16: bool = True
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "pr_auc"
    greater_is_better: Optional[bool] = None
    early_stopping_patience: int = field(default_factory=lambda: _get_env_int("EARLY_STOPPING_PATIENCE", 10))
    seed: int = GLOBAL_SEED
    report_to: str = field(default_factory=lambda: _get_env("REPORT_TO", "none"))
    wandb_project: str = field(default_factory=lambda: _get_env("WANDB_PROJECT", "chainuq"))
    wandb_entity: Optional[str] = field(default_factory=lambda: _get_env("WANDB_ENTITY", "") or None)
    wandb_run_name: Optional[str] = None
    dataloader_num_workers: int = field(default_factory=lambda: _get_env_int("DATALOADER_NUM_WORKERS", 4))
    dataloader_pin_memory: bool = True
    dataloader_prefetch_factor: int = field(default_factory=lambda: _get_env_int("PREFETCH_FACTOR", 4))
    dataloader_persistent_workers: bool = True
    dataloader_drop_last: bool = False
    logging_strategy: str = field(default_factory=lambda: _get_env("LOGGING_STRATEGY", "epoch"))
    logging_steps: int = field(default_factory=lambda: _get_env_int("LOGGING_STEPS", 100))
    disable_tqdm: bool = field(default_factory=lambda: _get_env_bool("DISABLE_TQDM", True))

@dataclass
class GenerationConfig:
    """Phase 1 generation + feature caching.

    ``hf_micro_batch_size`` trades throughput vs. peak VRAM on the HF forward
    used after vLLM decode. Lower if you see CUDA OOM during feature extraction.
    ``chunk_size`` trades number of chunk files vs. RAM when loading caches.
    """

    cache_dir: str = field(default_factory=lambda: _get_env("CACHE_DIR", f"{ARTIFACTS_ROOT}/cached_features"))
    max_new_tokens: int = field(default_factory=lambda: _get_env_int("GEN_MAX_NEW_TOKENS", 320))
    do_sample: bool = field(default_factory=lambda: _get_env_bool("GEN_DO_SAMPLE", False))
    temperature: float = field(default_factory=lambda: _get_env_float("GEN_TEMPERATURE", 1.0))
    batch_size: int = field(default_factory=lambda: _get_env_int("GEN_BATCH_SIZE", 512))
    skip_existing: bool = field(default_factory=lambda: _get_env_bool("GEN_SKIP_EXISTING", True))
    chunk_size: int = field(default_factory=lambda: _get_env_int("GEN_CHUNK_SIZE", 5000))
    save_hidden_states: bool = field(
        default_factory=lambda: _get_env_bool("GEN_SAVE_HIDDEN_STATES", True)
    )
    save_token_probs: bool = field(
        default_factory=lambda: _get_env_bool("GEN_SAVE_TOKEN_PROBS", True)
    )
    save_attention_weights: bool = field(
        default_factory=lambda: _get_env_bool("GEN_SAVE_ATTENTION_WEIGHTS", True)
    )
    hf_micro_batch_size: int = field(
        default_factory=lambda: _get_env_int("HF_MICRO_BATCH_SIZE", 2)
    )
    backend: str = field(default_factory=lambda: _get_env("BACKEND", "vllm"))

@dataclass
class VLLMConfig:
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = field(
        default_factory=lambda: _get_env_float("VLLM_GPU_MEMORY_UTILIZATION", 0.5)
    )
    max_model_len: int = field(
        default_factory=lambda: _get_env_int("VLLM_MAX_MODEL_LEN", 4096)
    )
    enforce_eager: bool = False
    swap_space: int = 4
    dtype: str = "auto"
    seed: int = GLOBAL_SEED
    attention_backend: str = field(
        default_factory=lambda: _get_env("VLLM_ATTENTION_BACKEND", "FLASHINFER")
    )

@dataclass
class JudgeConfig:
    judge_model_path: str = field(
        default_factory=lambda: f"{MODELS_ROOT}/{_get_env('JUDGE_MODEL_NAME', 'Mistral-Small-3.2-24B-Instruct-2506')}"
    )
    judge_backend: str = field(default_factory=lambda: _get_env("BACKEND", "vllm"))
    judge_max_new_tokens: int = 192
    judge_batch_size: int = field(default_factory=lambda: _get_env_int("JUDGE_BATCH_SIZE", 512))
    guided_decoding_enabled: bool = field(
        default_factory=lambda: _get_env_bool("JUDGE_GUIDED_DECODING_ENABLED", True)
    )
    guided_json_with_analysis: bool = field(
        default_factory=lambda: _get_env_bool("JUDGE_GUIDED_JSON_WITH_ANALYSIS", True)
    )
    thinking_token_budget: Optional[int] = field(
        default_factory=lambda: _get_env_int("JUDGE_THINKING_TOKEN_BUDGET", 0) or None
    )

@dataclass
class OutputConfig:
    output_dir: str = field(default_factory=lambda: _get_env("OUTPUT_DIR", RESULTS_ROOT))
    log_dir: str = field(default_factory=lambda: LOGS_ROOT)
    log_level: str = field(default_factory=lambda: _get_env("LOG_LEVEL", "INFO").upper())
    log_format: str = field(
        default_factory=lambda: _get_env(
            "LOG_FORMAT",
            "%(asctime)s [%(levelname)s] %(message)s",
            # "%(asctime)s [%(levelname)s] %(name)s | %(message)s",

        )
    )
    log_datefmt: str = field(
        default_factory=lambda: _get_env("LOG_DATEFMT", "%Y-%m-%d %H:%M:%S")
    )
    log_banner_width: int = field(default_factory=lambda: _get_env_int("LOG_BANNER_WIDTH", 72))
    save_final_model: bool = True
    final_model_subdir: str = "final_model"

@dataclass
class CacheConfig:
    """RAM budget (GB) for CachedFeatureDataset chunk caches.

    Set via environment variables exported by common.sh:
      CACHE_TRAIN_MEM_BUDGET_GB  — chunk cache for the training split
      CACHE_VAL_MEM_BUDGET_GB    — chunk cache for the validation split
      CACHE_EVAL_MEM_BUDGET_GB   — chunk cache for evaluation / OOD splits
    """
    train_mem_budget_gb: float = field(
        default_factory=lambda: _get_env_float("CACHE_TRAIN_MEM_BUDGET_GB", 87.5)
    )
    val_mem_budget_gb: float = field(
        default_factory=lambda: _get_env_float("CACHE_VAL_MEM_BUDGET_GB", 20.0)
    )
    eval_mem_budget_gb: float = field(
        default_factory=lambda: _get_env_float("CACHE_EVAL_MEM_BUDGET_GB", 100.0)
    )

@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    head: HeadConfig = field(default_factory=HeadConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    vllm: VLLMConfig = field(default_factory=VLLMConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
