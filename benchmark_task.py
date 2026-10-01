"""Inspect entry points for AutoDataBench tasks."""

from inspect_ai import task

from bench_inspect import task_from_definition
from bench_tasks.retrieval_v1 import load_task


@task
def retrieval_v1():
    return task_from_definition(load_task(), sandbox="local")
