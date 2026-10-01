"""State shared by tools, solvers, and scorers within one sample."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass

from bench_core.quota import QuotaConfig, QuotaState
from bench_core.snapshot import BestSoFarSnapshot


@dataclass
class RuntimeState:
    quota: QuotaState
    best: BestSoFarSnapshot
    submitted_path: str | None = None


_state: ContextVar[RuntimeState | None] = ContextVar("autodatabench_state", default=None)


def initialize(config: QuotaConfig, *, minimize: bool = False) -> RuntimeState:
    state = RuntimeState(
        quota=QuotaState(config),
        best=BestSoFarSnapshot(minimize=minimize),
    )
    _state.set(state)
    return state


def current() -> RuntimeState:
    state = _state.get()
    if state is None:
        raise RuntimeError("runtime state is not initialized")
    return state
