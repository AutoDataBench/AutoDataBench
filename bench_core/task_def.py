"""Contract implemented by every benchmark task."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from .quota import QuotaConfig


class FormatValidator(Protocol):
    def validate(self, dataset_path: str) -> ValidationResult:
        """Validate an agent-produced dataset."""


class ValidationResult(dict[str, Any]):
    """Serializable validation result returned by task validators."""


BackendFactory = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class TaskDefinition:
    """Everything a runtime needs to construct one benchmark task."""

    task_id: str
    version: str
    description: str
    quota: QuotaConfig
    prompts: dict[str, str]
    validator: FormatValidator
    backend_factory: BackendFactory
    data: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    training: dict[str, Any] = field(default_factory=dict)
    agent_limits: dict[str, Any] = field(default_factory=dict)
    private_paths: list[str] = field(default_factory=list)
