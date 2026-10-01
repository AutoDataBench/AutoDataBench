"""BFCL v3 single-turn OOD evaluation."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from .scripts import eval as evaluator

NONLIVE = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]
LIVE = [
    "live_simple",
    "live_multiple",
    "live_parallel",
    "live_parallel_multiple",
    "live_irrelevance",
]


def _macro(scores: dict[str, float], categories: list[str]) -> float | None:
    values = [scores[name] for name in categories if name in scores]
    return sum(values) / len(values) if values else None


def evaluate_ood(
    checkpoint: str,
    base_model: str,
    dataset: str,
    *,
    batch_size: int = 32,
) -> dict:
    samples = [json.loads(line) for line in Path(dataset).read_text().splitlines() if line]
    with tempfile.TemporaryDirectory(prefix="function-call-ood-") as temporary:
        model = evaluator.ensure_merged(checkpoint, base_model, temporary)
        results = evaluator.run_inference(
            samples,
            model,
            base_model,
            batch_size,
            512,
            4096,
        )
    scores = evaluator.score_results(results)
    scores["nonlive_macro"] = _macro(scores["per_category"], NONLIVE)
    scores["live_macro"] = _macro(scores["per_category"], LIVE)
    scores["n_examples"] = len(samples)
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--base-model", default="models/Qwen2-1.5B-Instruct")
    parser.add_argument("--data", default="data/function_call_v1/bfcl_guard.jsonl")
    parser.add_argument("--output", default="ood_function_call.json")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    result = evaluate_ood(
        args.checkpoint,
        args.base_model,
        args.data,
        batch_size=args.batch_size,
    )
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
