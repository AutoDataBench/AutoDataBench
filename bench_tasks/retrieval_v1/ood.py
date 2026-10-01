"""OOD evaluation used for the retrieval experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

OOD_BENCHMARKS = [
    "ClimateFEVERHardNegatives",
    "CQADupstackGamingRetrieval",
    "CQADupstackUnixRetrieval",
    "Touche2020Retrieval.v3",
    "TRECCOVID",
]


def evaluate_ood(checkpoint: str, benchmarks: list[str] | None = None) -> dict:
    import mteb
    import numpy as np
    from sentence_transformers import SentenceTransformer

    names = benchmarks or OOD_BENCHMARKS
    model = SentenceTransformer(checkpoint)
    scores, per_benchmark = [], {}
    for name in names:
        try:
            results = mteb.MTEB(tasks=mteb.get_tasks(tasks=[name])).run(
                model, output_folder=None, eval_splits=["test"]
            )
            score = results[0].scores["test"][0]
            ndcg = float(score["ndcg_at_10"])
            scores.append(ndcg)
            per_benchmark[name] = {
                "ndcg@10": ndcg,
                "ndcg@1": float(score.get("ndcg_at_1", 0.0)),
                "ndcg@5": float(score.get("ndcg_at_5", 0.0)),
                "mrr@10": float(score.get("mrr_at_10", 0.0)),
                "map@10": float(score.get("map_at_10", 0.0)),
                "recall@100": float(score.get("recall_at_100", 0.0)),
            }
        except Exception as error:
            per_benchmark[name] = {"error": f"{type(error).__name__}: {error}"}
    return {
        "overall_ndcg": float(np.mean(scores)) if scores else 0.0,
        "n_benchmarks_evaluated": len(scores),
        "n_benchmarks_total": len(names),
        "per_benchmark": per_benchmark,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="ood_retrieval.json")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_dir():
        raise FileNotFoundError(checkpoint)
    result = evaluate_ood(str(checkpoint))
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
