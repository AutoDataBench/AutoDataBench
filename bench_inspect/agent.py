"""Assembly of the AutoDataBench agent and its tools."""

from pathlib import Path

from inspect_ai.agent import as_solver, deepagent
from inspect_ai.tool import memory

from bench_core.backends import DataBackend, ModelBackend, TrainEvalBackend
from bench_core.task_def import FormatValidator

from .tools import (
    make_data_read,
    make_data_stats,
    make_model_call,
    make_model_embed,
    make_quota_remaining,
    make_submit_final,
    make_train_and_eval,
    make_validate_format,
)
from .bridge import BridgeHandler
from .python_tool import make_python_tool

PROMPTS_DIR = Path(__file__).parent / "prompts"


def _universal_instructions() -> str:
    return "\n\n".join(
        (PROMPTS_DIR / filename).read_text()
        for filename in ("system_base.md", "strategy_primer.md")
    )


def build_agent(
    *,
    data: DataBackend,
    model: ModelBackend,
    training: TrainEvalBackend,
    validator: FormatValidator,
    instructions: str,
    use_python: bool = False,
):
    tools = [
        make_train_and_eval(
            training,
            validator,
            model,
        )(),
        make_model_call(model)(),
        make_model_embed(model)(),
        make_data_read(data)(),
        make_data_stats(data)(),
        make_quota_remaining()(),
        make_validate_format(validator)(),
        make_submit_final(data)(),
        memory(),
    ]
    if use_python:
        tools.append(make_python_tool(BridgeHandler(model, training, validator))())
    return as_solver(
        deepagent(
            tools=tools,
            instructions=_universal_instructions() + "\n\n" + instructions,
        )
    )
