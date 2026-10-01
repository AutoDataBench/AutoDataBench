"""Inspect entry points for AutoDataBench tasks."""

from inspect_ai import task

from bench_inspect import task_from_definition
from bench_tasks.retrieval_v1 import load_task
from bench_tasks.knowledge_injection_v1 import load_task as load_knowledge_injection
from bench_tasks.function_call_v1 import load_task as load_function_call


@task
def retrieval_v1():
    return task_from_definition(load_task(), sandbox="local")


@task
def knowledge_injection_v1():
    return task_from_definition(load_knowledge_injection(), sandbox="local")


@task
def function_call_v1():
    return task_from_definition(load_function_call(), sandbox="local")
