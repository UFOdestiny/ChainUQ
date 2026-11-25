"""Shared logging helpers for ChainUQ workflow scripts."""

from __future__ import annotations

import logging
import math
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
DEFAULT_LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"
DEFAULT_BANNER_WIDTH = 72


def _coerce_level(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return getattr(logging, value.upper(), logging.INFO)
    return logging.INFO


def _cfg_output(cfg: Any) -> Any:
    return getattr(cfg, "output", None) if cfg is not None else None


def configure_logging(
    cfg: Any = None,
    *,
    level: Optional[Any] = None,
    force: bool = False,
    logger_levels: Optional[dict[str, Any]] = None,
) -> logging.Logger:
    output_cfg = _cfg_output(cfg)
    resolved_level = level or getattr(output_cfg, "log_level", DEFAULT_LOG_LEVEL)
    resolved_format = getattr(output_cfg, "log_format", DEFAULT_LOG_FORMAT)
    resolved_datefmt = getattr(output_cfg, "log_datefmt", DEFAULT_LOG_DATEFMT)

    logging.basicConfig(
        level=_coerce_level(resolved_level),
        format=resolved_format,
        datefmt=resolved_datefmt,
        force=force,
    )

    for logger_name, logger_level in (logger_levels or {}).items():
        logging.getLogger(logger_name).setLevel(_coerce_level(logger_level))

    return logging.getLogger()


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def set_logger_level(name: str, level: Any) -> None:
    logging.getLogger(name).setLevel(_coerce_level(level))


def compact_value(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        magnitude = abs(value)
        if magnitude >= 1000:
            return f"{value:.2f}"
        if magnitude >= 1:
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return f"{value:.6f}".rstrip("0").rstrip(".")
    if isinstance(value, (int, bool)):
        return str(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        parts = [f"{key}={compact_value(item)}" for key, item in value.items()]
        return "{" + ", ".join(parts) + "}"
    if isinstance(value, (list, tuple, set)):
        return "[" + ", ".join(compact_value(item) for item in value) + "]"
    return str(value)


def format_fields(fields: Mapping[str, Any], *, include_none: bool = False) -> str:
    parts = []
    for key, value in fields.items():
        if value is None and not include_none:
            continue
        parts.append(f"{key}={compact_value(value)}")
    return " ".join(parts)


def summarize_mapping(
    mapping: Mapping[str, Any],
    *,
    include: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    prefixes_to_strip: Iterable[str] = (),
) -> OrderedDict[str, Any]:
    selected = OrderedDict()
    exclude_set = set(exclude or ())
    items = ((key, mapping.get(key)) for key in include) if include else mapping.items()

    for key, value in items:
        if key is None or key in exclude_set:
            continue
        clean_key = key
        for prefix in prefixes_to_strip:
            if clean_key.startswith(prefix):
                clean_key = clean_key[len(prefix) :]
                break
        selected[clean_key] = value

    return selected


@dataclass
class WorkflowLogger:
    logger: logging.Logger
    phase: str
    width: int = DEFAULT_BANNER_WIDTH

    def _event_name(self, name: str) -> str:
        return f"{self.phase}.{name}" if self.phase else name

    def event(self, name: str, *, level: int = logging.INFO, **fields: Any) -> None:
        message = self._event_name(name)
        suffix = format_fields(fields)
        if suffix:
            message = f"{message} | {suffix}"
        self.logger.log(level, message)

    def warning(self, name: str, **fields: Any) -> None:
        self.event(name, level=logging.WARNING, **fields)

    def stage_start(self, name: str, **fields: Any) -> None:
        self.event(f"{name}.start", **fields)

    def stage_end(self, name: str, **fields: Any) -> None:
        self.event(f"{name}.done", **fields)

    def metrics(
        self,
        name: str,
        metrics: Mapping[str, Any],
        *,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
        prefixes_to_strip: Iterable[str] = (),
    ) -> None:
        selected = summarize_mapping(
            metrics,
            include=include,
            exclude=exclude,
            prefixes_to_strip=prefixes_to_strip,
        )
        self.event(name, **selected)

    def artifact(self, name: str, path: Any, **fields: Any) -> None:
        self.event(name, path=path, **fields)

    def header(self, title: str, **fields: Any) -> None:
        self.logger.info("=" * self.width)
        self.logger.info(title)
        for key, value in fields.items():
            self.logger.info("%-16s %s", f"{key}:", compact_value(value))
        self.logger.info("=" * self.width)
