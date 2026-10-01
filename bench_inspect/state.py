"""State shared by tools, solvers, and scorers within one sample."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path

from bench_core.quota import QuotaConfig, QuotaState
from bench_core.snapshot import BestSoFarSnapshot


@dataclass
class RuntimeState:
    quota: QuotaState
    best: BestSoFarSnapshot
    submitted_path: str | None = None
    workspace: Path | None = None


_state: ContextVar[RuntimeState | None] = ContextVar("autodatabench_state", default=None)


def initialize(
    config: QuotaConfig,
    *,
    minimize: bool = False,
    workspace: Path | None = None,
) -> RuntimeState:
    state = RuntimeState(
        quota=QuotaState(config),
        best=BestSoFarSnapshot(minimize=minimize),
        workspace=workspace,
    )
    _state.set(state)
    return state


def current() -> RuntimeState:
    state = _state.get()
    if state is None:
        raise RuntimeError("runtime state is not initialized")
    return state


def resolve_workspace_path(path: str) -> str:
    """Map the public /workspace alias to this sample's private directory."""
    workspace = current().workspace
    if workspace is None:
        return path
    if path == "/workspace":
        return str(workspace)
    if path.startswith("/workspace/"):
        return str(workspace / path.removeprefix("/workspace/"))
    return path
