"""Sandboxed Python tool with bridge replay."""

from __future__ import annotations

from pathlib import Path

from inspect_ai.tool import tool
from inspect_ai.tool._tools._execute import code_viewer
from inspect_ai.util import sandbox as sandbox_env

from bench_core.sandbox import (
    SandboxExecConfig,
    build_exec_cmd,
    format_exec_output,
    truncate_output,
)

from .bridge import (
    BRIDGE_EXIT_CODE,
    BridgeHandler,
    cleanup_bridge_state,
    extract_bridge_seq,
    read_bridge_request,
    write_bridge_response,
)
from .state import current

HELPERS_ROOT = str(Path(__file__).parent.parent)


def make_python_tool(handler: BridgeHandler, config: SandboxExecConfig | None = None):
    config = config or SandboxExecConfig()
    command = build_exec_cmd(config)

    @tool(name="python", viewer=code_viewer("python", "code"), parallel=True)
    def python_tool():
        async def execute(code: str) -> str:
            """Execute Python in /workspace; use bench_helpers for JSONL and backend calls."""
            workspace = current().workspace
            if workspace is None:
                raise RuntimeError("workspace is not initialized")
            if config.drop_privileges:
                workspace.chmod(0o777)
            bridge_dir = str(workspace / ".bench_bridge")
            await cleanup_bridge_state(bridge_dir)

            rewritten = code.replace("/workspace", str(workspace))
            program = (
                f"import os, sys; os.chdir({str(workspace)!r}); "
                f"sys.path.insert(0, {HELPERS_ROOT!r})\n"
                "import bench_helpers.bridge as _bench_bridge\n"
                f"_bench_bridge.BRIDGE_DIR = {bridge_dir!r}\n"
                + rewritten
            )
            output = ""
            for _ in range(config.max_bridge_rounds):
                try:
                    result = await sandbox_env().exec(
                        cmd=command,
                        input=program,
                        timeout=config.timeout,
                    )
                except TimeoutError:
                    return f"[TIMEOUT] Execution exceeded {config.timeout} seconds"
                except Exception as error:
                    return f"[ERROR] {type(error).__name__}: {error}"

                output = format_exec_output(result)
                if result.returncode != BRIDGE_EXIT_CODE:
                    return truncate_output(output, config.max_output_chars)
                seq = extract_bridge_seq(output)
                if seq is None:
                    break
                request = await read_bridge_request(seq, bridge_dir)
                response = await handler.handle(request)
                await write_bridge_response(seq, response, bridge_dir)

            message = (
                f"[BRIDGE ROUND LIMIT EXCEEDED] A python call is limited to "
                f"{config.max_bridge_rounds} backend calls. Batch inputs or split the code "
                "across multiple python calls.\n\n"
                + output
            )
            return truncate_output(message, config.max_output_chars)

        return execute

    return python_tool
