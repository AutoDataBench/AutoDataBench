import json
from pathlib import Path

import pytest

from bench_tasks.knowledge_injection_v1 import load_task
from bench_tasks.knowledge_injection_v1.backends import KnowledgeDataBackend
from bench_tasks.knowledge_injection_v1.validator import KnowledgeInjectionValidator


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_load_task_uses_configured_public_paths(tmp_path) -> None:
    definition = load_task(data_root=str(tmp_path))
    assert definition.task_id == "knowledge_injection_v1"
    assert str(tmp_path / "context_pool.jsonl") in definition.prompts["task"]
    assert str(tmp_path / "sources.jsonl") in definition.prompts["task"]
    assert definition.quota.axes["train_samples"].limit == 100_000


def test_validator_requires_frozen_context_and_unique_pairs(tmp_path) -> None:
    pool = tmp_path / "context_pool.jsonl"
    _write_jsonl(pool, [{"qid": "Q1", "summary_extract": "Frozen context."}])
    valid = tmp_path / "valid.jsonl"
    row = {
        "context_qid": "Q1",
        "context": "Frozen context.",
        "question": "What fact does this context state?",
        "teacher_continuation": "It states a frozen fact.",
    }
    _write_jsonl(valid, [row])
    validator = KnowledgeInjectionValidator(str(pool))
    assert validator.validate(str(valid))["valid"] is True

    invalid = tmp_path / "invalid.jsonl"
    _write_jsonl(invalid, [row, row])
    result = validator.validate(str(invalid))
    assert result["valid"] is False
    assert "duplicate" in result["errors"][0]["msg"]


@pytest.mark.anyio
async def test_data_backend_supports_experiment_filters(tmp_path) -> None:
    data = tmp_path / "contexts.jsonl"
    _write_jsonl(
        data,
        [
            {"qid": "Q1", "earliest_anchor_year": 1920, "domains": ["history"]},
            {"qid": "Q2", "earliest_anchor_year": 1960, "domains": ["science"]},
        ],
    )
    result = await KnowledgeDataBackend().read(
        str(data), {"earliest_anchor_year": {"$gte": 1950}}, 10
    )
    assert [row["qid"] for row in result["samples"]] == ["Q2"]
