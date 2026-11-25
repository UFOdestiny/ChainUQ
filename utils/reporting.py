from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from utils.cache_io import write_json_atomic

REPORT_SCHEMA_VERSION = "1.0"


def _with_schema_metadata(payload: Mapping[str, Any], kind: str) -> dict[str, Any]:
    obj = dict(payload)
    obj.setdefault("report_schema", {
        "kind": kind,
        "version": REPORT_SCHEMA_VERSION,
    })
    return obj


def _as_path(path: str | Path) -> Path:
    if isinstance(path, Path):
        return path
    return Path(path)


def write_report(path: str | Path, payload: Mapping[str, Any], *, kind: str = "report") -> Path:
    report_path = _as_path(path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(report_path, _with_schema_metadata(payload, kind), indent=2)
    return report_path


def write_predictions(path: str | Path, payload: Mapping[str, Any]) -> Path:
    pred_path = _as_path(path)
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(pred_path, _with_schema_metadata(payload, "predictions"), indent=2)
    return pred_path


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> Path:
    manifest_path = _as_path(path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(manifest_path, _with_schema_metadata(payload, "manifest"), indent=2)
    return manifest_path


def write_index(path: str | Path, payload: list) -> Path:
    index_path = _as_path(path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(index_path, payload, indent=2)
    return index_path


def write_training_args(path: str | Path, payload: Mapping[str, Any]) -> Path:
    out_path = _as_path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(out_path, _with_schema_metadata(payload, "training_args"), indent=2)
    return out_path
