"""Calls from sandboxed Python to benchmark backends."""

from __future__ import annotations

import json
import os
import sys
from typing import Any

BRIDGE_DIR = "/tmp/bench_bridge"
BRIDGE_EXIT_CODE = 99
BRIDGE_MARKER_PREFIX = "BENCH_BRIDGE_CALL:"
_call_seq = 0


def _call(method: str, params: dict[str, Any]) -> dict:
    global _call_seq
    _call_seq += 1
    os.makedirs(BRIDGE_DIR, exist_ok=True)
    response_path = os.path.join(BRIDGE_DIR, f"resp_{_call_seq}.json")
    if os.path.exists(response_path):
        with open(response_path) as file:
            response = json.load(file)
        if "error" in response:
            raise RuntimeError(f"Bridge call {method} failed: {response['error']}")
        return response.get("result", {})

    request_path = os.path.join(BRIDGE_DIR, f"req_{_call_seq}.json")
    with open(request_path, "w") as file:
        json.dump(
            {"jsonrpc": "2.0", "method": method, "params": params, "seq": _call_seq},
            file,
            ensure_ascii=False,
        )
    print(f"{BRIDGE_MARKER_PREFIX}{_call_seq}", flush=True)
    sys.exit(BRIDGE_EXIT_CODE)


def small_model_call(
    prompt: str,
    inputs: list[str],
    sampling: dict[str, Any] | None = None,
) -> dict:
    params: dict[str, Any] = {"prompt": prompt, "inputs": inputs}
    if sampling is not None:
        params["sampling"] = sampling
    return _call("small_model_call", params)


def small_model_embed(texts: list[str]) -> dict:
    return _call("small_model_embed", {"texts": texts})


def train_and_eval(dataset_path: str, n_seeds: int = 1) -> dict:
    return _call("train_and_eval", {"dataset_path": dataset_path, "n_seeds": n_seeds})


def validate_format(dataset_path: str) -> dict:
    return _call("validate_format", {"dataset_path": dataset_path})
