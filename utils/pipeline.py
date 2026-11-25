from __future__ import annotations

import time
from dataclasses import dataclass
from collections.abc import Callable, Mapping
from typing import Any, Optional

from utils.log import WorkflowLogger


@dataclass
class StageSpec:
    name: str
    fn: Callable[[Any], Any]
    input_contract: Optional[Callable[[Any], None]] = None
    output_contract: Optional[Callable[[Any], None]] = None
    retries: int = 0
    retry_on: tuple[type[BaseException], ...] = (Exception,)
    include_duration: bool = True
    start_fields: Optional[Callable[[Any], Mapping[str, Any]]] = None
    result_fields: Optional[Callable[[Any], Mapping[str, Any]]] = None


class PipelineRunner:
    def __init__(self, workflow: WorkflowLogger):
        self.workflow = workflow
        self._stages: dict[str, StageSpec] = {}

    def register_stage(self, spec: StageSpec) -> None:
        self._stages[spec.name] = spec

    def run(self, name: str, payload: Any = None) -> Any:
        if name not in self._stages:
            raise KeyError(f"Stage '{name}' is not registered")
        spec = self._stages[name]
        return self._execute(spec, payload)

    def _execute(self, spec: StageSpec, payload: Any) -> Any:
        if spec.input_contract is not None:
            spec.input_contract(payload)

        start_fields = dict(spec.start_fields(payload) if spec.start_fields else {})
        self.workflow.stage_start(spec.name, **start_fields)

        started_at = time.time()
        attempts = 0
        last_error: Optional[BaseException] = None

        while attempts <= max(0, int(spec.retries)):
            attempts += 1
            try:
                result = spec.fn(payload)
                if spec.output_contract is not None:
                    spec.output_contract(result)

                done_fields: dict[str, Any] = {"attempts": attempts}
                if spec.result_fields is not None:
                    done_fields.update(dict(spec.result_fields(result)))
                if spec.include_duration:
                    done_fields["duration_s"] = time.time() - started_at
                self.workflow.stage_end(spec.name, **done_fields)
                return result
            except spec.retry_on as error:
                last_error = error
                if attempts > max(0, int(spec.retries)):
                    failed_fields: dict[str, Any] = {
                        "status": "failed",
                        "attempts": attempts,
                        "error_type": type(error).__name__,
                    }
                    if spec.include_duration:
                        failed_fields["duration_s"] = time.time() - started_at
                    self.workflow.stage_end(spec.name, **failed_fields)
                    raise
                self.workflow.warning(
                    f"{spec.name}.retry",
                    attempt=attempts,
                    max_attempts=max(0, int(spec.retries)) + 1,
                    error_type=type(error).__name__,
                )

        if last_error is not None:
            raise last_error
        raise RuntimeError(f"Stage '{spec.name}' failed without raising a retryable exception")

