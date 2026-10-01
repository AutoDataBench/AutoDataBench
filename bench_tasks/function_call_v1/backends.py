"""Backends for the function-calling benchmark."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from bench_tasks.retrieval_v1.backends import HttpModelBackend, StubModelBackend

TASK_DIR = Path(__file__).parent


class FunctionCallDataBackend:
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
                    if axis == "n_tools":
                        values[axis].append(len(row.get("tools", [])))
                    elif axis == "n_calls":
                        values[axis].append(len(row.get("gold_calls", [])))
                    elif axis in row:
                        values[axis].append(row[axis])
        summary = {}
        for axis, items in values.items():
            if not items:
                continue
            if all(isinstance(item, (int, float)) for item in items):
                summary[axis] = {"min": min(items), "max": max(items), "mean": sum(items) / len(items)}
            else:
                unique = sorted({json.dumps(item, sort_keys=True) if isinstance(item, (list, dict)) else str(item) for item in items})
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
            try:
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
            except TypeError:
                return False
    return True


class FunctionCallTrainingBackend:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    async def count_samples(self, dataset_path: str) -> int:
        with Path(dataset_path).open() as file:
            return sum(bool(line.strip()) for line in file)

    async def train_and_eval(self, dataset_path, n_seeds=1, checkpoint_dir=None):
        result = await asyncio.to_thread(_run_pipeline, dataset_path, self.config, checkpoint_dir)
        return {
            "val_scores": [result["macro"]] * n_seeds,
            "dataset_hash": _hash(dataset_path),
            "n_samples_trained": await self.count_samples(dataset_path),
            "score_breakdown": result.get("per_category", {}),
            "checkpoint_path": result.get("checkpoint_path", ""),
        }


class FunctionCallScorer:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config

    async def score(self, submitted_dataset_path: str):
        result = await asyncio.to_thread(_run_pipeline, submitted_dataset_path, self.config, None)
        return {"score": result["macro"], "metric": "macro_ast", "n_examples": 2000}


class StubTrainingBackend:
    async def count_samples(self, dataset_path):
        with Path(dataset_path).open() as file:
            return sum(bool(line.strip()) for line in file)

    async def train_and_eval(self, dataset_path, n_seeds=1, checkpoint_dir=None):
        score = int(_hash(dataset_path)[:8], 16) / 0xFFFFFFFF
        return {"val_scores": [score] * n_seeds, "dataset_hash": _hash(dataset_path), "n_samples_trained": await self.count_samples(dataset_path)}


class StubScorer:
    async def score(self, submitted_dataset_path):
        return {"score": int(_hash(submitted_dataset_path)[:8], 16) / 0xFFFFFFFF, "metric": "stub_macro_ast", "n_examples": 0}


def _hash(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run_pipeline(dataset_path: str, config: dict, checkpoint_dir: str | None):
    with tempfile.TemporaryDirectory(prefix="function-call-") as output:
        output_path = Path(output)
        environment = dict(os.environ)
        environment.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        environment["CUDA_VISIBLE_DEVICES"] = str(config.get("train_cuda_visible_devices", "0"))
        train = [
            sys.executable, str(TASK_DIR / "scripts" / "train.py"),
            "--data", dataset_path, "--output", output,
            "--seed", str(config.get("train_seed", 42)),
            "--base_model", config["base_model"],
            "--epochs", str(config.get("epochs", 1)),
            "--lr", str(config.get("learning_rate", 5e-5)),
            "--batch_size", str(config.get("batch_size", 4)),
            "--grad_accum", str(config.get("grad_accum", 2)),
            "--lora_r", str(config.get("lora_rank", 32)),
            "--lora_alpha", str(config.get("lora_alpha", 64)),
            "--max_seq_len", str(config.get("max_seq_len", 2048)),
        ]
        if config.get("no_grad_ckpt", False):
            train.append("--no_grad_ckpt")
        _run(train, environment, 7200)
        _run(
            [sys.executable, str(TASK_DIR / "scripts" / "eval.py"),
             "--adapter", str(output_path / "adapter"),
             "--base_model", config["base_model"],
             "--data", config["test_path"], "--output", output],
            os.environ.copy(), 3600,
        )
        scores = json.loads((output_path / "test_scores.json").read_text())
        if checkpoint_dir:
            destination = Path(checkpoint_dir) / "adapter"
            shutil.copytree(output_path / "adapter", destination, dirs_exist_ok=True)
            scores["checkpoint_path"] = str(destination)
        return scores


def _run(command: list[str], environment: dict, timeout: int) -> None:
    process = subprocess.run(command, capture_output=True, text=True, timeout=timeout, env=environment)
    if process.returncode:
        raise RuntimeError(process.stderr[-3000:] or process.stdout[-3000:])


def build_backends(*, data, model, training):
    model_backend = StubModelBackend() if model.get("mode") == "stub" else HttpModelBackend(model)
    if training.get("mode") == "stub":
        training_backend, scorer = StubTrainingBackend(), StubScorer()
    else:
        training_backend, scorer = FunctionCallTrainingBackend(training), FunctionCallScorer(training)
    return {"data": FunctionCallDataBackend(), "model": model_backend, "training": training_backend, "scorer": scorer}
