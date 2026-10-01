"""Backend implementations for the retrieval task."""

from __future__ import annotations

import asyncio
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import httpx


class JsonlDataBackend:
    async def read(self, dataset_path: str, where: dict[str, Any], n: int):
        rows = []
        with Path(dataset_path).open() as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if all(row.get(key) == value for key, value in where.items()):
                    rows.append(row)
                if len(rows) == n:
                    break
        return {"samples": rows, "n_returned": len(rows)}

    async def stats(self, dataset_path: str, axes: list[str]):
        query_lengths = []
        document_lengths = []
        negative_counts = []
        subset_counts = {}
        n_samples = 0
        with Path(dataset_path).open() as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                n_samples += 1
                if "length" in axes:
                    query_lengths.append(len(row["query"]))
                    document_lengths.append(len(row["positive_doc"]))
                if "negatives" in axes:
                    negative_counts.append(len(row["hard_negative_docs"]))
                if "subsets" in axes and "subset" in row:
                    subset = row["subset"]
                    subset_counts[subset] = subset_counts.get(subset, 0) + 1
        summary = {}
        if "length" in axes:
            summary["query_length"] = _summary(query_lengths)
            summary["document_length"] = _summary(document_lengths)
        if "negatives" in axes:
            summary["hard_negatives"] = _summary(negative_counts)
        if "subsets" in axes:
            summary["subsets"] = subset_counts
        return {"n_samples": n_samples, "summary": summary}

    async def accept_submission(self, dataset_path: str) -> bool:
        return Path(dataset_path).is_file()


def _summary(values: list[int]) -> dict[str, float]:
    if not values:
        return {"min": 0, "max": 0, "mean": 0.0}
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values)}


class HttpModelBackend:
    """OpenAI-compatible generation and embedding client."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.generation_model = config["generation_model"]
        self.generation_url = config["generation_url"].rstrip("/")
        self.embedding_model = config["embedding_model"]
        self.embedding_url = config["embedding_url"].rstrip("/")

    async def call(self, prompt, inputs, sampling=None):
        options = sampling or {}
        async with httpx.AsyncClient(timeout=300) as client:
            responses = await asyncio.gather(
                *[
                    client.post(
                        f"{self.generation_url}/chat/completions",
                        json={
                            "model": self.generation_model,
                            "messages": [
                                {"role": "system", "content": prompt},
                                {"role": "user", "content": value},
                            ],
                            **options,
                        },
                    )
                    for value in inputs
                ]
            )
        for response in responses:
            response.raise_for_status()
        payloads = [response.json() for response in responses]
        return {
            "responses": [
                payload["choices"][0]["message"]["content"] for payload in payloads
            ],
            "tokens_used": sum(
                payload.get("usage", {}).get(
                    "total_tokens",
                    self.count_tokens(prompt)
                    + self.count_tokens(payload["choices"][0]["message"]["content"]),
                )
                for payload in payloads
            ),
        }

    async def embed(self, texts):
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f"{self.embedding_url}/embeddings",
                json={"model": self.embedding_model, "input": texts},
            )
        response.raise_for_status()
        payload = response.json()
        vectors = [
            item["embedding"]
            for item in sorted(payload["data"], key=lambda item: item["index"])
        ]
        return {
            "embeddings": vectors,
            "tokens_used": payload.get("usage", {}).get(
                "total_tokens", sum(self.count_tokens(text) for text in texts)
            ),
        }

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    async def release(self):
        return None


class StubModelBackend:
    async def call(self, prompt, inputs, sampling=None):
        return {
            "responses": [f"stub:{value[:32]}" for value in inputs],
            "tokens_used": sum(self.count_tokens(value) for value in inputs),
        }

    async def embed(self, texts):
        vectors = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rng = random.Random(seed)
            vectors.append([rng.uniform(-1, 1) for _ in range(8)])
        return {
            "embeddings": vectors,
            "tokens_used": sum(self.count_tokens(text) for text in texts),
        }

    def count_tokens(self, text):
        return max(1, len(text) // 4)

    async def release(self):
        return None


class RetrievalTrainingBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    async def count_samples(self, dataset_path):
        with Path(dataset_path).open() as file:
            return sum(bool(line.strip()) for line in file)

    async def train_and_eval(self, dataset_path, n_seeds=1, checkpoint_dir=None):
        result = await asyncio.to_thread(
            _run_trainer, dataset_path, self.config, checkpoint_dir
        )
        return {
            "val_scores": [result["overall_ndcg"]] * n_seeds,
            "dataset_hash": _hash(dataset_path),
            "n_samples_trained": result["dataset_size"],
            "score_breakdown": result["per_benchmark"],
            "checkpoint_path": result.get("checkpoint_path", ""),
        }


class RetrievalScorer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    async def score(self, submitted_dataset_path):
        result = await asyncio.to_thread(
            _run_trainer, submitted_dataset_path, self.config, None
        )
        return {
            "score": result["overall_ndcg"],
            "metric": "nDCG@10",
            "n_examples": result["n_benchmarks_evaluated"],
        }


class StubTrainingBackend:
    async def count_samples(self, dataset_path):
        with Path(dataset_path).open() as file:
            return sum(bool(line.strip()) for line in file)

    async def train_and_eval(self, dataset_path, n_seeds=1, checkpoint_dir=None):
        digest = _hash(dataset_path)
        score = int(digest[:8], 16) / 0xFFFFFFFF
        return {
            "val_scores": [score] * n_seeds,
            "dataset_hash": digest,
            "n_samples_trained": await self.count_samples(dataset_path),
        }


class StubScorer:
    async def score(self, submitted_dataset_path):
        digest = _hash(submitted_dataset_path)
        return {
            "score": int(digest[:8], 16) / 0xFFFFFFFF,
            "metric": "stub_nDCG@10",
            "n_examples": 0,
        }


def _hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_trainer(dataset_path, config, checkpoint_dir):
    command = [
        sys.executable,
        "-m",
        "bench_tasks.retrieval_v1.trainer",
        "--train",
        dataset_path,
        "--config",
        json.dumps(config),
    ]
    if checkpoint_dir:
        command.extend(["--output", checkpoint_dir])
    process = subprocess.run(
        command, capture_output=True, text=True, timeout=18_000, check=False
    )
    if process.returncode:
        raise RuntimeError(process.stderr[-2000:])
    return json.loads(process.stdout.strip().splitlines()[-1])


def build_backends(*, data, model, training):
    model_backend = (
        StubModelBackend() if model.get("mode") == "stub" else HttpModelBackend(model)
    )
    if training.get("mode") == "stub":
        training_backend = StubTrainingBackend()
        scorer = StubScorer()
    else:
        training_backend = RetrievalTrainingBackend(training)
        scorer = RetrievalScorer(training)
    return {
        "data": JsonlDataBackend(),
        "model": model_backend,
        "training": training_backend,
        "scorer": scorer,
    }
