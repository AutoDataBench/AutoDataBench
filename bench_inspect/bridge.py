"""Host side of calls made by ``bench_helpers.bridge``."""

from __future__ import annotations

import json
import re

from inspect_ai.util import sandbox as sandbox_env

from bench_core.backends import ModelBackend, TrainEvalBackend
from bench_core.task_def import FormatValidator

from . import operations
from .state import resolve_workspace_path

BRIDGE_EXIT_CODE = 99
BRIDGE_MARKER_PREFIX = "BENCH_BRIDGE_CALL:"
_MARKER = re.compile(r"BENCH_BRIDGE_CALL:(\d+)")


def extract_bridge_seq(output: str) -> int | None:
    match = _MARKER.search(output)
    return int(match.group(1)) if match else None


async def read_bridge_request(seq: int, bridge_dir: str) -> dict:
    content = await sandbox_env().read_file(f"{bridge_dir}/req_{seq}.json")
    return json.loads(content)


async def write_bridge_response(seq: int, response: dict, bridge_dir: str) -> None:
    await sandbox_env().write_file(
        f"{bridge_dir}/resp_{seq}.json",
        json.dumps(response, ensure_ascii=False),
    )


async def cleanup_bridge_state(bridge_dir: str) -> None:
    await sandbox_env().exec(cmd=["rm", "-rf", bridge_dir], timeout=5)


class BridgeHandler:
    def __init__(
        self,
        model: ModelBackend,
        training: TrainEvalBackend,
        validator: FormatValidator,
    ) -> None:
        self.model = model
        self.training = training
        self.validator = validator

    async def handle(self, request: dict) -> dict:
        try:
            result = await self._dispatch(request["method"], request.get("params", {}))
            return {"jsonrpc": "2.0", "result": result, "seq": request.get("seq")}
        except Exception as error:
            return {
                "jsonrpc": "2.0",
                "error": f"{type(error).__name__}: {error}",
                "seq": request.get("seq"),
            }

    async def _dispatch(self, method: str, params: dict) -> dict:
        if method == "small_model_call":
            return await operations.model_call(
                self.model,
                params["prompt"],
                params["inputs"],
                params.get("sampling"),
            )
        if method == "small_model_embed":
            return await operations.model_embed(self.model, params["texts"])
        if method == "train_and_eval":
            return await operations.train_and_eval(
                self.training,
                self.validator,
                params["dataset_path"],
                params.get("n_seeds", 1),
                self.model,
            )
        if method == "validate_format":
            path = resolve_workspace_path(params["dataset_path"])
            return self.validator.validate(path)
        raise ValueError(f"unknown bridge method: {method}")
