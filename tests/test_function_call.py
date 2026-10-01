import json
from pathlib import Path

import pytest

from bench_tasks.function_call_v1 import load_task
from bench_tasks.function_call_v1.backends import FunctionCallDataBackend
from bench_tasks.function_call_v1.validator import FunctionCallValidator
from bench_tasks.function_call_v1.scripts.eval import case_match, parse_output


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def _row(**changes):
    row = {
        "id": "one",
        "query": "What is the weather in Paris?",
        "tools": [
            {
                "name": "weather",
                "description": "Get weather",
                "parameters": {"type": "object"},
            }
        ],
        "gold_calls": [{"name": "weather", "arguments": {"city": "Paris"}}],
        "category": "simple",
    }
    row.update(changes)
    return row


def test_load_task_preserves_pool_basename(tmp_path) -> None:
    definition = load_task(data_root=str(tmp_path))
    assert definition.task_id == "function_call_v1"
    assert "`pool_agent.jsonl`" in definition.prompts["task"]
    assert definition.training["train_seed"] == 42
    assert definition.quota.axes["train_samples"].limit == 160_000


def test_validator_checks_calls_against_tools(tmp_path) -> None:
    valid = tmp_path / "valid.jsonl"
    _write_jsonl(valid, [_row()])
    validator = FunctionCallValidator()
    assert validator.validate(str(valid))["valid"] is True

    invalid = tmp_path / "invalid.jsonl"
    _write_jsonl(invalid, [_row(gold_calls=[{"name": "missing", "arguments": {}}])])
    result = validator.validate(str(invalid))
    assert result["valid"] is False
    assert "listed tool" in result["errors"][0]["msg"]


@pytest.mark.anyio
async def test_data_backend_filters_categories(tmp_path) -> None:
    data = tmp_path / "pool_agent.jsonl"
    _write_jsonl(data, [_row(), _row(id="two", category="irrelevance", gold_calls=[])])
    result = await FunctionCallDataBackend().read(
        str(data), {"category": {"$in": ["irrelevance"]}}, 10
    )
    assert [row["id"] for row in result["samples"]] == ["two"]


def test_ast_matching_accepts_parallel_call_permutations() -> None:
    gold = [
        {"name": "weather", "arguments": {"city": "Paris"}},
        {"name": "weather", "arguments": {"city": "Tokyo"}},
    ]
    predicted = parse_output(
        'result: [{"name":"weather","arguments":{"city":"Tokyo"}},'
        '{"name":"weather","arguments":{"city":"Paris"}}]'
    )
    assert case_match(predicted, gold, "parallel") is True
