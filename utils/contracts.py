from __future__ import annotations

from typing import Any, TypedDict


class ThresholdProtocol(TypedDict, total=False):
    mode: str
    default_threshold: float
    tune_split: str | None
    eval_split: str
    applied_thresholds: dict[str, Any]


class EvalReport(TypedDict, total=False):
    method_type: str
    head_type: str
    dataset_name: str
    split: str
    total_samples: int
    total_claims: int
    overall_metrics: dict[str, Any]
    per_difficulty_metrics: dict[str, Any]
    threshold_protocol: ThresholdProtocol
    efficiency: dict[str, Any]
    diagnostics: dict[str, Any]
    diagnostics_path: str
    predictions_path: str


class TrainResultReport(TypedDict, total=False):
    head_type: str
    cache_dir: str
    train_samples: int
    eval_samples: int
    train_metrics: dict[str, Any]
    eval_metrics: dict[str, Any]
    efficiency: dict[str, Any]
    head_config: dict[str, Any]
    results_path: str
    final_model_dir: str


class GenerateManifest(TypedDict, total=False):
    split: str
    dataset_name: str
    total_samples: int
    total_claims: int
    num_chunks: int
    sample_pending: int
    claims_pending: int
    dropped_samples_generation: int
    phase_status: dict[str, Any]


class JudgeRunSummary(TypedDict, total=False):
    split: str
    total_samples: int
    sample_pending: int
    total_claims: int
    claim_pending: int
    samples_judged: int
    dropped_samples: int


class NecessitySummary(TypedDict, total=False):
    model: str
    dataset: str
    split: str
    n_samples: int
    used_judge: bool
    judge_model: str | None
    avg_reasoning_steps: float
    exp1_accuracy: dict[str, Any]
    exp2_error_localization: dict[str, Any]
    exp3_selective_answering: dict[str, Any]
