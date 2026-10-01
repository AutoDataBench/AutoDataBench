"""Train a sentence-transformer and evaluate it with MTEB."""

from __future__ import annotations

import argparse
import json
import tempfile
import time
from pathlib import Path


def train_and_evaluate(train_path: str, config: dict, output: str | None = None):
    import numpy as np
    import torch
    from sentence_transformers import InputExample, SentenceTransformer, losses
    from torch.utils.data import DataLoader

    rows = [
        json.loads(line)
        for line in Path(train_path).read_text().splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError("training dataset is empty")
    examples = [
        InputExample(
            texts=[
                row["query"],
                row["positive_doc"],
                *row["hard_negative_docs"][:20],
            ]
        )
        for row in rows
    ]
    torch.manual_seed(42)
    np.random.seed(42)
    model = SentenceTransformer(config["base_model"])
    model.max_seq_length = config.get("max_length", 512)
    loader = DataLoader(
        examples, shuffle=True, batch_size=config.get("batch_size", 128)
    )
    loss = losses.MultipleNegativesRankingLoss(model)

    temporary = None
    if output is None:
        temporary = tempfile.TemporaryDirectory()
        output = temporary.name
    started = time.time()
    model.fit(
        train_objectives=[(loader, loss)],
        epochs=config.get("epochs", 1),
        warmup_steps=int(
            len(loader)
            * config.get("epochs", 1)
            * config.get("warmup_ratio", 0.1)
        ),
        optimizer_params={"lr": config.get("learning_rate", 5e-5)},
        output_path=output,
        show_progress_bar=False,
        use_amp=torch.cuda.is_available(),
    )
    result = evaluate(model, config["benchmarks"])
    result.update(
        {
            "dataset_size": len(rows),
            "training_seconds": round(time.time() - started, 2),
            "checkpoint_path": None if temporary else output,
        }
    )
    if temporary:
        temporary.cleanup()
    return result


def evaluate(model, benchmarks):
    import mteb
    import numpy as np

    per_benchmark = {}
    scores = []
    for name in benchmarks:
        tasks = mteb.get_tasks(tasks=[name])
        results = mteb.MTEB(tasks=tasks).run(
            model, output_folder=None, eval_splits=["test"]
        )
        score = results[0].scores["test"][0]
        ndcg = float(score["ndcg_at_10"])
        scores.append(ndcg)
        per_benchmark[name] = {"ndcg@10": ndcg}
    return {
        "overall_ndcg": float(np.mean(scores)),
        "per_benchmark": per_benchmark,
        "n_benchmarks_evaluated": len(scores),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    print(json.dumps(train_and_evaluate(args.train, json.loads(args.config), args.output)))


if __name__ == "__main__":
    main()

