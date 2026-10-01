"""Security settings for agent-written Python."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

BENCH_HELPERS_DIR = str(Path(__file__).parent.parent)


@dataclass
class SandboxExecConfig:
    network_isolation: bool = True
    drop_privileges: bool = True
    clean_env: bool = True
    max_output_chars: int = 16_000
    timeout: int = 3600
    max_bridge_rounds: int = 500
    sandbox_uid: int = 999
    sandbox_gid: int = 998


def build_exec_cmd(config: SandboxExecConfig) -> list[str]:
    """Return the command used to execute a Python program from stdin."""
    python = sys.executable or shutil.which("python3") or "python3"
    if not (config.network_isolation or config.drop_privileges or config.clean_env):
        return [python, "-"]

    command: list[str] = []
    if config.clean_env:
        command += [
            "env", "-i",
            "PATH=/usr/local/bin:/usr/bin:/bin",
            "HOME=/tmp",
            f"PYTHONPATH={BENCH_HELPERS_DIR}",
            "TMPDIR=/tmp",
        ]
    if config.network_isolation:
        command += ["unshare", "--net"]
    if config.drop_privileges:
        program = (
            "import os,sys; code=sys.stdin.read(); os.setgroups([]); "
            f"os.setgid({config.sandbox_gid}); os.setuid({config.sandbox_uid}); "
            "exec(code)"
        )
        command += [python, "-c", program]
    else:
        command += [python, "-"]
    return command


def format_exec_output(result) -> str:
    return ((result.stderr + "\n") if result.stderr else "") + result.stdout


def truncate_output(output: str, limit: int) -> str:
    if len(output) <= limit:
        return output
    return (
        output[:limit]
        + f"\n\n[OUTPUT TRUNCATED: {len(output)} characters; showing {limit}. "
        "Write large results to a file instead.]"
    )
