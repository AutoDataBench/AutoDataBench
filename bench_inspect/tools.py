"""Inspect tool wrappers around the benchmark operations."""

from __future__ import annotations

import json
from typing import Any

from inspect_ai.tool import ToolError, tool

from bench_core.backends import DataBackend, ModelBackend, TrainEvalBackend
from bench_core.task_def import FormatValidator

from . import operations
from .state import current


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def make_train_and_eval(backend: TrainEvalBackend, validator: FormatValidator):
    @tool(name="train_and_eval")
    def factory():
        async def execute(dataset_path: str, n_seeds: int = 1) -> str:
            """Train on a candidate dataset and return validation results."""
            try:
                result = await operations.train_and_eval(
                    backend, validator, dataset_path, n_seeds
                )
                return _json(result)
            except operations.QuotaExceededError as error:
                raise ToolError(str(error)) from error

        return execute

    return factory


def make_model_call(backend: ModelBackend):
    @tool(name="small_model_call")
    def factory():
        async def execute(
            prompt: str,
            inputs: list[str],
            sampling: dict[str, Any] | None = None,
        ) -> str:
            """Generate responses for a batch of inputs with the small model."""
            try:
                return _json(
                    await operations.model_call(backend, prompt, inputs, sampling)
                )
            except operations.QuotaExceededError as error:
                raise ToolError(str(error)) from error

        return execute

    return factory


def make_model_embed(backend: ModelBackend):
    @tool(name="small_model_embed")
    def factory():
        async def execute(texts: list[str]) -> str:
            """Embed a batch of texts with the small model."""
            try:
                return _json(await operations.model_embed(backend, texts))
            except operations.QuotaExceededError as error:
                raise ToolError(str(error)) from error

        return execute

    return factory


def make_data_read(backend: DataBackend):
    @tool(name="data_read")
    def factory():
        async def execute(dataset_path: str, where: dict[str, Any], n: int) -> str:
            """Read up to 500 matching rows from a dataset."""
            try:
                return _json(
                    await operations.data_read(backend, dataset_path, where, n)
                )
            except operations.QuotaExceededError as error:
                raise ToolError(str(error)) from error

        return execute

    return factory


def make_data_stats(backend: DataBackend):
    @tool(name="data_stats")
    def factory():
        async def execute(dataset_path: str, axes: list[str]) -> str:
            """Return aggregate statistics for a dataset."""
            return _json(await backend.stats(dataset_path, axes))

        return execute

    return factory


def make_quota_remaining():
    @tool(name="quota_remaining")
    def factory():
        async def execute() -> str:
            """Return resource usage and the best validation result."""
            state = current()
            return _json(
                {
                    "quotas": state.quota.snapshot(),
                    "best": {
                        "dataset_path": state.best.dataset_path,
                        "score": state.best.score,
                    },
                }
            )

        return execute

    return factory


def make_validate_format(validator: FormatValidator):
    @tool(name="validate_format")
    def factory():
        async def execute(dataset_path: str) -> str:
            """Validate a candidate dataset without consuming quota."""
            return _json(validator.validate(dataset_path))

        return execute

    return factory


def make_submit_final(backend: DataBackend):
    @tool(name="submit_final")
    def factory():
        async def execute(dataset_path: str) -> str:
            """Submit the final dataset for held-out scoring."""
            return _json(await operations.submit(backend, dataset_path))

        return execute

    return factory

