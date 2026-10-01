"""Inspect solvers for runtime initialization and fallback submission."""

import shutil
import tempfile
from pathlib import Path

from inspect_ai.solver import Solver, solver

from bench_core.backends import DataBackend
from bench_core.quota import QuotaConfig

from .operations import auto_submit
from .state import initialize


@solver
def initialize_runtime(
    config: QuotaConfig,
    *,
    minimize: bool = False,
    pool_path: str | None = None,
    prompts: dict[str, str] | None = None,
) -> Solver:
    async def solve(state, generate):
        workspace = Path(tempfile.mkdtemp(prefix="autodatabench-"))
        if pool_path is not None:
            shutil.copyfile(pool_path, workspace / Path(pool_path).name)
        prompt_files = {
            "background": "background.md",
            "format": "format_spec.md",
            "task": "task_prompt.md",
        }
        for name, text in (prompts or {}).items():
            (workspace / prompt_files.get(name, f"{name}.md")).write_text(text)
        initialize(config, minimize=minimize, workspace=workspace)
        return state

    return solve


@solver
def submit_best_on_exit(backend: DataBackend) -> Solver:
    async def solve(state, generate):
        state.metadata["submit_source"] = await auto_submit(backend)
        return state

    return solve
