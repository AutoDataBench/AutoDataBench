"""Framework-neutral implementations of the agent-facing operations."""

from __future__ import annotations

from typing import Any

from bench_core.backends import DataBackend, ModelBackend, TrainEvalBackend
from bench_core.task_def import FormatValidator

from .state import current


class QuotaExceededError(RuntimeError):
    """Raised when an operation has no remaining budget."""


def _check(*names: str) -> None:
    quota = current().quota
    exhausted = [
        name for name in names if quota.has_axis(name) and quota.exhausted(name)
    ]
    if exhausted:
        raise QuotaExceededError(f"quota exhausted: {', '.join(exhausted)}")


def _debit(name: str, amount: float) -> None:
    quota = current().quota
    if quota.has_axis(name):
        quota.debit(name, amount)


async def train_and_eval(
    backend: TrainEvalBackend,
    validator: FormatValidator,
    dataset_path: str,
    n_seeds: int = 1,
) -> dict[str, Any]:
    _check("eval_calls", "train_samples")
    validation = validator.validate(dataset_path)
    if not validation.get("valid", False):
        return {"accepted": False, "validation": validation}

    n_samples = await backend.count_samples(dataset_path)
    _debit("eval_calls", n_seeds)
    _debit("train_samples", n_samples)
    try:
        result = await backend.train_and_eval(dataset_path, n_seeds=n_seeds)
    except Exception:
        quota = current().quota
        if quota.has_axis("eval_calls"):
            quota.refund("eval_calls", n_seeds)
        if quota.has_axis("train_samples"):
            quota.refund("train_samples", n_samples)
        raise

    scores = result.get("val_scores", [])
    if not scores:
        raise ValueError("training backend returned no validation scores")
    score = sum(scores) / len(scores)
    improved = current().best.consider(
        dataset_path=dataset_path,
        score=score,
        dataset_hash=result["dataset_hash"],
    )
    return {**result, "mean_score": score, "best_updated": improved}


async def model_call(
    backend: ModelBackend,
    prompt: str,
    inputs: list[str],
    sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _check("small_model_tokens")
    result = await backend.call(prompt, inputs, sampling)
    _debit("small_model_tokens", result["tokens_used"])
    return result


async def model_embed(backend: ModelBackend, texts: list[str]) -> dict[str, Any]:
    _check("small_model_tokens")
    result = await backend.embed(texts)
    _debit("small_model_tokens", result["tokens_used"])
    return result


async def data_read(
    backend: DataBackend,
    dataset_path: str,
    where: dict[str, Any],
    n: int,
    *,
    max_rows: int = 500,
) -> dict[str, Any]:
    _check("read_samples")
    result = await backend.read(dataset_path, where, min(n, max_rows))
    _debit("read_samples", result["n_returned"])
    return result


async def submit(backend: DataBackend, dataset_path: str) -> dict[str, Any]:
    state = current()
    if state.submitted_path == dataset_path:
        return {"accepted": True, "path": dataset_path, "already_submitted": True}
    accepted = await backend.accept_submission(dataset_path)
    if accepted:
        state.submitted_path = dataset_path
    return {"accepted": accepted, "path": dataset_path}


async def auto_submit(backend: DataBackend) -> str | None:
    state = current()
    if state.submitted_path is not None:
        return "agent"
    if state.best.dataset_path is None:
        return None
    if await backend.accept_submission(state.best.dataset_path):
        state.submitted_path = state.best.dataset_path
        return "auto"
    return None
