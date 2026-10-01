"""Inspect scorer backed by a task's held-out evaluator."""

from inspect_ai.scorer import Score, Scorer, Target, mean, scorer, stderr
from inspect_ai.solver import TaskState

from bench_core.backends import ScorerBackend

from .state import current


@scorer(metrics=[mean(), stderr()])
def test_scorer(backend: ScorerBackend) -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        submitted = current().submitted_path
        if submitted is None:
            artifacts = current().artifacts
            if artifacts is not None:
                artifacts.save_results(
                    {
                        "task_id": getattr(state, "sample_id", "unknown"),
                        "submit_source": "none",
                        "test_score": None,
                        "quota": current().quota.snapshot(),
                        "evaluations": artifacts.history,
                    }
                )
            return Score.unscored(explanation="no dataset was submitted")
        runtime = current()
        run_metadata = {}
        final_dataset = submitted
        final_hash = runtime.best.dataset_hash
        if runtime.artifacts is not None:
            final_dataset, final_hash = runtime.artifacts.save_final_dataset(submitted)
            run_metadata["run_dir"] = str(runtime.artifacts.run_dir)
        try:
            result = await backend.score(submitted)
        except Exception as error:
            if runtime.artifacts is not None:
                runtime.artifacts.save_results(
                    {
                        "task_id": state.metadata.get("task_id")
                        or getattr(state, "sample_id", "unknown"),
                        "submit_source": state.metadata.get("submit_source"),
                        "final_dataset": final_dataset,
                        "final_dataset_hash": final_hash,
                        "best_validation_score": runtime.best.score,
                        "best_checkpoint": runtime.artifacts.best_checkpoint,
                        "test_score": None,
                        "scoring_error": f"{type(error).__name__}: {error}",
                        "quota": runtime.quota.snapshot(),
                        "evaluations": runtime.artifacts.history,
                    }
                )
            raise
        if runtime.artifacts is not None:
            runtime.artifacts.save_results(
                {
                    "task_id": state.metadata.get("task_id")
                    or getattr(state, "sample_id", "unknown"),
                    "submit_source": state.metadata.get("submit_source"),
                    "final_dataset": final_dataset,
                    "final_dataset_hash": final_hash,
                    "best_validation_score": runtime.best.score,
                    "best_checkpoint": runtime.artifacts.best_checkpoint,
                    "test_score": result["score"],
                    "test_metric": result["metric"],
                    "n_test_examples": result["n_examples"],
                    "quota": runtime.quota.snapshot(),
                    "evaluations": runtime.artifacts.history,
                }
            )
        return Score(
            value=result["score"],
            answer=submitted,
            explanation=(
                f"{result['metric']} = {result['score']:.4f} "
                f"on {result['n_examples']} examples"
            ),
            metadata={
                "metric": result["metric"],
                "n_examples": result["n_examples"],
                **run_metadata,
            },
        )

    return score
