"""Inspect solvers for runtime initialization and fallback submission."""

from inspect_ai.solver import Solver, solver

from bench_core.backends import DataBackend
from bench_core.quota import QuotaConfig

from .operations import auto_submit
from .state import initialize


@solver
def initialize_runtime(config: QuotaConfig, *, minimize: bool = False) -> Solver:
    async def solve(state, generate):
        initialize(config, minimize=minimize)
        return state

    return solve


@solver
def submit_best_on_exit(backend: DataBackend) -> Solver:
    async def solve(state, generate):
        state.metadata["submit_source"] = await auto_submit(backend)
        return state

    return solve

