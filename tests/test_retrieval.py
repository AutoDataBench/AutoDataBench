import json

from bench_tasks.retrieval_v1 import load_task
from bench_tasks.retrieval_v1.backends import build_backends
from bench_tasks.retrieval_v1.validator import RetrievalValidator


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_retrieval_task_loads_with_external_data_root(tmp_path) -> None:
    task = load_task(data_root=str(tmp_path))
    assert task.task_id == "retrieval_v1"
    assert task.data["pool_path"] == str(tmp_path / "train.jsonl")
    assert task.training["base_model"] == "nreimers/MiniLM-L6-H384-uncased"


def test_validator_accepts_uniform_hard_negatives(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    write_rows(
        dataset,
        [
            {"query": "q1", "positive_doc": "d1", "hard_negative_docs": ["n1"]},
            {"query": "q2", "positive_doc": "d2", "hard_negative_docs": ["n2"]},
        ],
    )
    result = RetrievalValidator().validate(str(dataset))
    assert result["valid"] is True
    assert result["n_rows"] == 2


def test_validator_rejects_ragged_hard_negatives(tmp_path) -> None:
    dataset = tmp_path / "train.jsonl"
    write_rows(
        dataset,
        [
            {"query": "q1", "positive_doc": "d1", "hard_negative_docs": []},
            {"query": "q2", "positive_doc": "d2", "hard_negative_docs": ["n2"]},
        ],
    )
    result = RetrievalValidator().validate(str(dataset))
    assert result["valid"] is False
    assert "same number" in result["errors"][-1]["msg"]


def test_stub_backends_run_without_ml_dependencies(tmp_path) -> None:
    backends = build_backends(
        data={},
        model={"mode": "stub"},
        training={"mode": "stub"},
    )
    assert set(backends) == {"data", "model", "training", "scorer"}
