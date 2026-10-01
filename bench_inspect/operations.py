"""Framework-neutral implementations of the agent-facing operations."""

from __future__ import annotations

from typing import Any

from bench_core.backends import DataBackend, ModelBackend, TrainEvalBackend
from bench_core.task_def import FormatValidator

from .state import current, resolve_workspace_path


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
    model_backend: ModelBackend | None = None,
) -> dict[str, Any]:
    dataset_path = resolve_workspace_path(dataset_path)
    _check("eval_calls", "train_samples")
    validation = validator.validate(dataset_path)
    if not validation.get("valid", False):
        return {"accepted": False, "validation": validation}

    n_samples = await backend.count_samples(dataset_path)
    _debit("eval_calls", n_seeds)
    _debit("train_samples", n_samples)
    artifacts = current().artifacts
    eval_sequence = None
    checkpoint = None
    if artifacts is not None:
        eval_sequence, _, checkpoint = artifacts.start_eval(dataset_path)
    try:
        if model_backend is not None:
            await model_backend.release()
        result = await backend.train_and_eval(
            dataset_path,
            n_seeds=n_seeds,
            checkpoint_dir=str(checkpoint) if checkpoint is not None else None,
        )
        scores = result.get("val_scores", [])
        if not scores:
            raise ValueError("training backend returned no validation scores")
    except Exception as error:
        quota = current().quota
        if quota.has_axis("eval_calls"):
            quota.refund("eval_calls", n_seeds)
        if quota.has_axis("train_samples"):
            quota.refund("train_samples", n_samples)
        if artifacts is not None and checkpoint is not None and eval_sequence is not None:
            artifacts.fail_eval(eval_sequence, checkpoint, error)
        raise

    score = sum(scores) / len(scores)
    improved = current().best.consider(
        dataset_path=dataset_path,
        score=score,
        dataset_hash=result["dataset_hash"],
    )
    if artifacts is not None and checkpoint is not None and eval_sequence is not None:
        saved_checkpoint = artifacts.finish_eval(
            sequence=eval_sequence,
            dataset_path=dataset_path,
            dataset_hash=result["dataset_hash"],
            checkpoint=checkpoint,
            result=result,
            mean_score=score,
            best_updated=improved,
            quota=current().quota.snapshot(),
        )
        result["checkpoint_path"] = saved_checkpoint or ""
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
    dataset_path = resolve_workspace_path(dataset_path)
    _check("read_samples")
    result = await backend.read(dataset_path, where, min(n, max_rows))
    _debit("read_samples", result["n_returned"])
    return result


async def submit(
    backend: DataBackend,
    dataset_path: str,
) -> dict[str, Any]:
    dataset_path = resolve_workspace_path(dataset_path)
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
