from __future__ import annotations

from typing import Any


def resolve_threshold_mode(args, train_results_threshold: float | None) -> str:
    if bool(getattr(args, "fit_thresholds_on_eval_split", False)):
        return "fit_on_eval_split"
    if bool(getattr(args, "threshold_source_report", None)):
        return "source_report"
    if train_results_threshold is not None:
        return "train_results"
    tune_split = getattr(args, "threshold_tune_split", None)
    eval_split = getattr(args, "split", None)
    if tune_split and tune_split != eval_split:
        return "tuned_on_split"
    return "fixed"


def build_threshold_protocol(
    *,
    mode: str,
    default_threshold: float,
    eval_split: str,
    tune_split: str | None,
    sample_threshold: float,
    sample_temperature: float,
    difficulty_thresholds: dict[int, float] | None,
    extras: dict[str, Any] | None = None,
) -> dict[str, Any]:
    protocol = {
        "mode": mode,
        "default_threshold": float(default_threshold),
        "tune_split": tune_split,
        "eval_split": eval_split,
        "applied_thresholds": {
            "default": float(default_threshold),
            "sample": float(sample_threshold),
            "sample_temperature": float(sample_temperature),
            "difficulty": {
                str(k): float(v)
                for k, v in (difficulty_thresholds or {}).items()
            },
        },
    }
    if extras:
        protocol.update(dict(extras))
    return protocol
