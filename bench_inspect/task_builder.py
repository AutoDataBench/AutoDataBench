"""Build an Inspect task from a framework-independent task definition."""

from inspect_ai import Task
from inspect_ai.dataset import Sample
from inspect_ai.util import SandboxEnvironmentSpec

from bench_core.task_def import TaskDefinition

from .agent import build_agent
from .scorer import test_scorer
from .solver import initialize_runtime, submit_best_on_exit


def task_from_definition(
    definition: TaskDefinition,
    *,
    sandbox: str | None = None,
) -> Task:
    backends = definition.backend_factory(
        data=definition.data,
        model=definition.model,
        training=definition.training,
    )
    instructions = "\n\n".join(definition.prompts.values())
    agent = build_agent(
        data=backends["data"],
        model=backends["model"],
        training=backends["training"],
        validator=definition.validator,
        instructions=instructions,
        use_python=sandbox is not None,
    )
    limits = definition.agent_limits
    task_options = {}
    if sandbox is not None:
        task_options["sandbox"] = SandboxEnvironmentSpec(sandbox)
    return Task(
        dataset=[
            Sample(
                input=definition.prompts.get("task", instructions),
                id=definition.task_id,
            )
        ],
        solver=[
            initialize_runtime(
                definition.quota,
                minimize=bool(definition.training.get("minimize", False)),
                data_paths=[
                    path
                    for name, path in definition.data.items()
                    if name in {"pool_path", "sources_path"}
                ],
                prompts=definition.prompts,
                private_paths=definition.private_paths,
            ),
            agent,
            submit_best_on_exit(backends["data"]),
        ],
        scorer=test_scorer(backends["scorer"]),
        message_limit=limits.get("message_limit"),
        token_limit=limits.get("token_limit"),
        time_limit=limits.get("time_limit"),
        **task_options,
    )
