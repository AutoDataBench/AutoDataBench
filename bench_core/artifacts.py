"""Durable outputs for one benchmark run."""

from __future__ import annotations

import hashlib
import json
import shutil
import threading
import time
import uuid
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


class RunArtifacts:
    def __init__(self, run_dir: Path) -> None:
        self.run_dir = run_dir
        self.artifacts_dir = run_dir / "artifacts"
        self.workspace = run_dir / "workspace"
        self.checkpoints = self.artifacts_dir / "checkpoints"
        self.trajectory = self.artifacts_dir / "trajectory.jsonl"
        self.history: list[dict[str, Any]] = []
        self.eval_sequence = 0
        self.best_checkpoint: str | None = None
        self._lock = threading.Lock()
        self.workspace.mkdir(parents=True)
        self.checkpoints.mkdir(parents=True)
        self.artifacts_dir.chmod(0o700)

    @classmethod
    def create(cls, task_id: str, output_root: str | Path = "runs") -> RunArtifacts:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        return cls(Path(output_root).absolute() / f"{task_id}_{timestamp}_{suffix}")

    def save_config(self, config: dict[str, Any]) -> None:
        self._write_json(self.artifacts_dir / "config.json", config)

    def start_eval(self, dataset_path: str) -> tuple[int, str, Path]:
        self.eval_sequence += 1
        digest = _sha256(Path(dataset_path))
        checkpoint = self.checkpoints / f"eval_{self.eval_sequence:03d}_{digest[:8]}"
        checkpoint.mkdir()
        self.event(
            "train_eval_started",
            sequence=self.eval_sequence,
            dataset_hash=digest,
            dataset_path=dataset_path,
        )
        return self.eval_sequence, digest, checkpoint

    def finish_eval(
        self,
        *,
        sequence: int,
        dataset_path: str,
        dataset_hash: str,
        checkpoint: Path,
        result: dict[str, Any],
        mean_score: float,
        best_updated: bool,
        quota: dict[str, Any],
    ) -> str | None:
        checkpoint_path = None
        if best_updated:
            best_dataset = self.artifacts_dir / "best_dataset.jsonl"
            shutil.copy2(dataset_path, best_dataset)
            best = self.artifacts_dir / "best_checkpoint"
            if best.exists():
                shutil.rmtree(best)
            if any(checkpoint.iterdir()):
                checkpoint.rename(best)
                checkpoint_path = str(best)
                self.best_checkpoint = checkpoint_path
            else:
                checkpoint.rmdir()
                self.best_checkpoint = None
            self._write_json(
                self.artifacts_dir / "best.json",
                {
                    "sequence": sequence,
                    "dataset": str(best_dataset),
                    "dataset_hash": dataset_hash,
                    "checkpoint": self.best_checkpoint,
                    "score": mean_score,
                },
            )
        else:
            shutil.rmtree(checkpoint, ignore_errors=True)

        record = {
            "sequence": sequence,
            "dataset_hash": dataset_hash,
            "val_scores": result.get("val_scores", []),
            "mean_score": mean_score,
            "score_breakdown": result.get("score_breakdown", {}),
            "n_samples_trained": result.get("n_samples_trained"),
            "best_updated": best_updated,
            "checkpoint": checkpoint_path,
            "quota": quota,
        }
        self.history.append(record)
        self._write_json(self.artifacts_dir / "evaluations.json", self.history)
        self.event("train_eval_finished", **record)
        return checkpoint_path

    def fail_eval(self, sequence: int, checkpoint: Path, error: Exception) -> None:
        shutil.rmtree(checkpoint, ignore_errors=True)
        self.event(
            "train_eval_failed",
            sequence=sequence,
            error=f"{type(error).__name__}: {error}",
        )

    def save_messages(self, messages: list[Any]) -> None:
        serialised = [_sanitise(_json_value(message)) for message in messages]
        self._write_json(self.artifacts_dir / "messages.json", serialised)

    def save_final_dataset(self, source: str) -> tuple[str, str]:
        destination = self.artifacts_dir / "final_dataset.jsonl"
        shutil.copy2(source, destination)
        return str(destination), _sha256(destination)

    def save_results(self, results: dict[str, Any]) -> None:
        self._write_json(self.artifacts_dir / "results.json", results)
        self.event("run_finished", **results)

    def event(self, event_type: str, **fields: Any) -> None:
        record = {"timestamp": time.time(), "type": event_type, **fields}
        line = json.dumps(record, ensure_ascii=False, default=_json_value) + "\n"
        with self._lock:
            with self.trajectory.open("a") as file:
                file.write(line)

    @staticmethod
    def _write_json(path: Path, value: Any) -> None:
        path.write_text(
            json.dumps(_sanitise(value), indent=2, ensure_ascii=False, default=_json_value)
            + "\n"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value):
        return asdict(value)
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=False)
    if isinstance(value, (set, tuple)):
        return list(value)
    return repr(value)


def _sanitise(value: Any) -> Any:
    secret_keys = {
        "api_key", "apikey", "authorization", "access_token", "refresh_token"
    }
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if str(key).lower() in secret_keys
            else _sanitise(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitise(item) for item in value]
    return value
