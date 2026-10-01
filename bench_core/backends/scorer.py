"""Held-out scoring interface."""

from typing import Protocol, TypedDict


class ScoreResult(TypedDict):
    score: float
    metric: str
    n_examples: int


class ScorerBackend(Protocol):
    async def score(self, submitted_dataset_path: str) -> ScoreResult: ...

