"""Inspect solvers for runtime initialization and fallback submission."""

import shutil
from pathlib import Path

from inspect_ai.solver import Solver, solver

from bench_core.backends import DataBackend
from bench_core.quota import QuotaConfig
from bench_core.sandbox import lock_paths_from_agent
from bench_core.artifacts import RunArtifacts

from .operations import auto_submit
from .state import current, initialize


@solver
def initialize_runtime(
    config: QuotaConfig,
    *,
    minimize: bool = False,
    data_paths: list[str] | None = None,
    prompts: dict[str, str] | None = None,
    private_paths: list[str] | None = None,
    task_id: str = "benchmark",
    output_root: str = "runs",
    task_config: dict | None = None,
) -> Solver:
    async def solve(state, generate):
        artifacts = RunArtifacts.create(task_id, output_root)
        workspace = artifacts.workspace
        for source in data_paths or []:
            shutil.copyfile(source, workspace / Path(source).name)
        prompt_files = {
            "background": "background.md",
            "format": "format_spec.md",
            "task": "task_prompt.md",
        }
        for name, text in (prompts or {}).items():
            (workspace / prompt_files.get(name, f"{name}.md")).write_text(text)
        lock_paths_from_agent(private_paths or [])
        initialize(config, minimize=minimize, workspace=workspace, artifacts=artifacts)
        artifacts.save_config(task_config or {})
        artifacts.event("run_started", task_id=task_id, workspace=str(workspace))
        state.metadata["run_dir"] = str(artifacts.run_dir)
        return state

    return solve


@solver
def submit_best_on_exit(backend: DataBackend) -> Solver:
    async def solve(state, generate):
        state.metadata["submit_source"] = await auto_submit(backend)
        runtime = current()
        if runtime.artifacts is not None:
            runtime.artifacts.save_messages(state.messages)
            runtime.artifacts.event(
                "dataset_submitted",
                source=state.metadata["submit_source"],
                dataset_path=runtime.submitted_path,
            )
        return state

    return solve
