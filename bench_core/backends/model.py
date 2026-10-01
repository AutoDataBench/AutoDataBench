"""Small-model inference interface."""

from typing import Any, Protocol, TypedDict


class ModelCallResult(TypedDict):
    responses: list[str]
    tokens_used: int


class ModelEmbedResult(TypedDict):
    embeddings: list[list[float]]
    tokens_used: int


class ModelBackend(Protocol):
    async def call(
        self,
        prompt: str,
        inputs: list[str],
        sampling: dict[str, Any] | None = None,
    ) -> ModelCallResult: ...

    async def embed(self, texts: list[str]) -> ModelEmbedResult: ...

    def count_tokens(self, text: str) -> int: ...

    async def release(self) -> None: ...

