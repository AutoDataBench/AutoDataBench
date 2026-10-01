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
            return Score.unscored(explanation="no dataset was submitted")
        result = await backend.score(submitted)
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
            },
        )

    return score

