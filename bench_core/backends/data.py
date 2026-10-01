"""Data access interface."""

from typing import Any, Protocol, TypedDict


class DataReadResult(TypedDict):
    samples: list[dict[str, Any]]
    n_returned: int


class DataStatsResult(TypedDict):
    n_samples: int
    summary: dict[str, Any]


class DataBackend(Protocol):
    async def read(
        self, dataset_path: str, where: dict[str, Any], n: int
    ) -> DataReadResult: ...

    async def stats(self, dataset_path: str, axes: list[str]) -> DataStatsResult: ...

    async def accept_submission(self, dataset_path: str) -> bool: ...

