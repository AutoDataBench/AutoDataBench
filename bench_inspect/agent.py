"""Assembly of the AutoDataBench agent and its tools."""

from inspect_ai.agent import as_solver, deepagent

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


def build_agent(
    *,
    data: DataBackend,
    model: ModelBackend,
    training: TrainEvalBackend,
    validator: FormatValidator,
    instructions: str,
):
    tools = [
        make_train_and_eval(training, validator)(),
        make_model_call(model)(),
        make_model_embed(model)(),
        make_data_read(data)(),
        make_data_stats(data)(),
        make_quota_remaining()(),
        make_validate_format(validator)(),
        make_submit_final(data)(),
    ]
    return as_solver(deepagent(tools=tools, instructions=instructions))

