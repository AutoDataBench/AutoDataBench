"""Training and validation interface."""

from typing import Any, Protocol, TypedDict


class TrainEvalResult(TypedDict, total=False):
    val_scores: list[float]
    dataset_hash: str
    n_samples_trained: int
    error_examples: list[dict[str, Any]]
    score_breakdown: dict[str, Any]
    checkpoint_path: str


class TrainEvalBackend(Protocol):
    async def train_and_eval(
        self,
        dataset_path: str,
        n_seeds: int = 1,
        checkpoint_dir: str | None = None,
    ) -> TrainEvalResult: ...

    async def count_samples(self, dataset_path: str) -> int: ...

