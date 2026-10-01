"""Knowledge-injection benchmark."""

from pathlib import Path

import yaml

from bench_core.quota import QuotaConfig
from bench_core.task_def import TaskDefinition

from .backends import build_backends
from .validator import KnowledgeInjectionValidator

TASK_DIR = Path(__file__).parent


def load_task(*, data_root: str | None = None) -> TaskDefinition:
    config = yaml.safe_load((TASK_DIR / "config.yaml").read_text())
    data = dict(config["data"])
    if data_root is not None:
        data["pool_path"] = str(Path(data_root) / "context_pool.jsonl")
        data["sources_path"] = str(Path(data_root) / "sources.jsonl")

    prompts = {
        "background": (TASK_DIR / "background.md").read_text(),
        "task": (TASK_DIR / "task_prompt.md").read_text(),
        "format": (TASK_DIR / "format_spec.md").read_text(),
    }
    prompts["task"] = (
        prompts["task"]
        .replace("{POOL_PATH}", data["pool_path"])
        .replace("{SOURCES_PATH}", data["sources_path"])
    )
    return TaskDefinition(
        task_id=config["task_id"],
        version=str(config["version"]),
        description=config["description"],
        quota=QuotaConfig.from_dict(config["quota"]),
        prompts=prompts,
        validator=KnowledgeInjectionValidator(data["pool_path"]),
        backend_factory=build_backends,
        data=data,
        model=config["model"],
        training=config["training"],
        agent_limits=config.get("agent_limits", {}),
    )


__all__ = ["load_task"]
