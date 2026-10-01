"""Backends for the knowledge-injection benchmark."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from bench_tasks.retrieval_v1.backends import HttpModelBackend, StubModelBackend

TASK_DIR = Path(__file__).parent


class KnowledgeDataBackend:
    async def read(self, dataset_path: str, where: dict[str, Any], n: int):
        rows = []
        with Path(dataset_path).open() as file:
            for line in file:
                row = json.loads(line)
                if _matches(row, where):
                    rows.append(row)
                if len(rows) == n:
                    break
        return {"samples": rows, "n_returned": len(rows)}

    async def stats(self, dataset_path: str, axes: list[str]):
        values = {axis: [] for axis in axes}
        n_samples = 0
        with Path(dataset_path).open() as file:
            for line in file:
                row = json.loads(line)
                n_samples += 1
                for axis in axes:
                    if axis in row:
                        value = row[axis]
                        values[axis].extend(value if isinstance(value, list) else [value])
        summary = {}
        for axis, items in values.items():
            if not items:
                continue
            if all(isinstance(item, (int, float)) for item in items):
                summary[axis] = {"min": min(items), "max": max(items), "mean": sum(items) / len(items)}
            else:
                unique = sorted({str(item) for item in items})
                summary[axis] = {"unique_count": len(unique), "sample_values": unique[:20]}
        return {"n_samples": n_samples, "summary": summary}

    async def accept_submission(self, dataset_path: str) -> bool:
        return Path(dataset_path).is_file()


def _matches(row: dict, filters: dict) -> bool:
    for key, condition in filters.items():
        value = row.get(key)
        if not isinstance(condition, dict):
            if condition not in value if isinstance(value, list) else value != condition:
                return False
            continue
        for operator, expected in condition.items():
            if operator == "$gte" and value < expected:
                return False
            if operator == "$lte" and value > expected:
                return False
            if operator == "$gt" and value <= expected:
                return False
            if operator == "$lt" and value >= expected:
                return False
            if operator == "$in" and value not in expected:
                return False
            if operator == "$contains" and expected not in value:
                return False
    return True


class KnowledgeTrainingBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    async def count_samples(self, dataset_path: str) -> int:
        with Path(dataset_path).open() as file:
            return sum(bool(line.strip()) for line in file)

    async def train_and_eval(self, dataset_path, n_seeds=1, checkpoint_dir=None):
        result = await asyncio.to_thread(_run_pipeline, dataset_path, self.config, checkpoint_dir)
        return {
            "val_scores": [result["macro_normalized"]] * n_seeds,
            "dataset_hash": _hash(dataset_path),
            "n_samples_trained": await self.count_samples(dataset_path),
            "score_breakdown": {
                "novel_normalized": result["novel_normalized"],
                "retention_normalized": result["retention_normalized"],
            },
            "checkpoint_path": result.get("checkpoint_path", ""),
        }


class KnowledgeScorer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    async def score(self, submitted_dataset_path: str):
        result = await asyncio.to_thread(_run_pipeline, submitted_dataset_path, self.config, None)
        return {"score": result["macro_normalized"], "metric": "macro_normalized", "n_examples": 5400}


class StubTrainingBackend:
    async def count_samples(self, dataset_path):
        with Path(dataset_path).open() as file:
            return sum(bool(line.strip()) for line in file)

    async def train_and_eval(self, dataset_path, n_seeds=1, checkpoint_dir=None):
        score = int(_hash(dataset_path)[:8], 16) / 0xFFFFFFFF
        return {"val_scores": [score] * n_seeds, "dataset_hash": _hash(dataset_path), "n_samples_trained": await self.count_samples(dataset_path)}


class StubScorer:
    async def score(self, submitted_dataset_path):
        return {"score": int(_hash(submitted_dataset_path)[:8], 16) / 0xFFFFFFFF, "metric": "stub_macro_normalized", "n_examples": 0}


def _hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_pipeline(dataset_path: str, config: dict, checkpoint_dir: str | None):
    with tempfile.TemporaryDirectory(prefix="knowledge-injection-") as output:
        command = [
            sys.executable,
            str(TASK_DIR / "scripts" / "run_benchmark.py"),
            "--submission", dataset_path,
            "--output", output,
            "--model", config["base_model"],
            "--num-gpus", str(config.get("num_gpus", 1)),
        ]
        process = subprocess.run(command, capture_output=True, text=True, timeout=14_400)
        if process.returncode:
            raise RuntimeError(process.stderr[-3000:] or process.stdout[-3000:])
        result = json.loads((Path(output) / "result.json").read_text())
        if checkpoint_dir:
            source = Path(output) / "training" / "final_adapter"
            if source.is_dir():
                import shutil
                shutil.copytree(source, Path(checkpoint_dir), dirs_exist_ok=True)
                result["checkpoint_path"] = str(Path(checkpoint_dir))
        return result


def build_backends(*, data, model, training):
    model_backend = StubModelBackend() if model.get("mode") == "stub" else HttpModelBackend(model)
    if training.get("mode") == "stub":
        training_backend, scorer = StubTrainingBackend(), StubScorer()
    else:
        training_backend, scorer = KnowledgeTrainingBackend(training), KnowledgeScorer(training)
    return {"data": KnowledgeDataBackend(), "model": model_backend, "training": training_backend, "scorer": scorer}
