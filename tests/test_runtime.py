import pytest

from bench_core.quota import QuotaConfig
from bench_core.task_def import TaskDefinition
from bench_inspect import task_from_definition
from bench_inspect.operations import auto_submit, data_read, train_and_eval
from bench_inspect.state import current, initialize
from bench_core.sandbox import lock_paths_from_agent


class Validator:
    def validate(self, dataset_path):
        return {"valid": True, "n_rows": 4, "errors": [], "warnings": []}


class Data:
    async def read(self, dataset_path, where, n):
        return {"samples": [{"id": i} for i in range(n)], "n_returned": n}

    async def stats(self, dataset_path, axes):
        return {"n_samples": 4, "summary": {}}

    async def accept_submission(self, dataset_path):
        return dataset_path.endswith(".jsonl")


class Training:
    async def count_samples(self, dataset_path):
        return 4

    async def train_and_eval(self, dataset_path, n_seeds=1, checkpoint_dir=None):
        return {
            "val_scores": [0.5] * n_seeds,
            "dataset_hash": "hash",
            "n_samples_trained": 4,
        }


class Model:
    async def call(self, prompt, inputs, sampling=None):
        return {"responses": inputs, "tokens_used": len(inputs)}

    async def embed(self, texts):
        return {"embeddings": [[0.0] for _ in texts], "tokens_used": len(texts)}

    def count_tokens(self, text):
        return len(text)

    async def release(self):
        return None


class Scorer:
    async def score(self, submitted_dataset_path):
        return {"score": 0.5, "metric": "stub", "n_examples": 1}


@pytest.mark.anyio
async def test_runtime_tracks_budget_best_result_and_fallback() -> None:
    initialize(
        QuotaConfig.from_dict(
            {
                "read_samples": {"limit": 10},
                "eval_calls": {"limit": 2},
                "train_samples": {"limit": 10},
            }
        )
    )

    await data_read(Data(), "pool.jsonl", {}, 3)
    result = await train_and_eval(Training(), Validator(), "candidate.jsonl", 1)

    assert result["best_updated"] is True
    assert current().quota.spent("read_samples") == 3
    assert current().quota.spent("train_samples") == 4
    assert await auto_submit(Data()) == "auto"
    assert current().submitted_path == "candidate.jsonl"


def test_task_definition_builds_an_inspect_task() -> None:
    def backend_factory(**config):
        return {
            "data": Data(),
            "model": Model(),
            "training": Training(),
            "scorer": Scorer(),
        }

    definition = TaskDefinition(
        task_id="stub",
        version="1",
        description="Stub task",
        quota=QuotaConfig.from_dict({}),
        prompts={"task": "Improve the dataset."},
        validator=Validator(),
        backend_factory=backend_factory,
    )

    task = task_from_definition(definition)
    assert len(task.dataset) == 1
    assert task.dataset[0].id == "stub"


def test_private_paths_are_hidden_from_sandbox_uid(tmp_path) -> None:
    private = tmp_path / "test.jsonl"
    private.write_text("secret")
    private.chmod(0o644)
    lock_paths_from_agent([str(private)])
    assert private.stat().st_mode & 0o007 == 0
    assert private.stat().st_mode & 0o600 == 0o600
