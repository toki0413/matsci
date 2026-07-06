"""Bash tool — run shell commands inside the workspace.

Used by the Coder agent to run tests, builds, and git operations.
Always requires approval.
"""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.security import ContainerExecutor, SandboxError, SandboxExecutor, get_executor
from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class BashToolInput(BaseModel):
    action: Literal["run", "stream"] = Field(default="run")
    command: list[str] = Field(..., description="Command as a list of arguments")
    working_dir: str | None = Field(default=None)
    timeout: float = Field(default=300.0, gt=0)
    capture_output: bool = Field(default=True)
    stream: bool = Field(
        default=False,
        description="Stream stdout/stderr line-by-line while the command runs",
    )


class BashTool(HuginnTool):
    """Run shell commands in the workspace."""

    name = "bash_tool"
    category = "core"
    description = (
        "Run a shell command as a list of arguments inside the workspace. "
        "Use for tests, builds, git, and other command-line tasks."
    )
    destructive = True
    input_schema = BashToolInput

    def _stream_command(
        self,
        command: list[str],
        cwd: Path,
        timeout: float,
    ) -> tuple[str, str, int]:
        """Run a command and stream stdout/stderr line-by-line.

        Returns the accumulated stdout, stderr, and exit code.
        """
        stdout_chunks: list[str] = []
        stderr_chunks: list[str] = []

        def _reader(pipe, buffer: list[str]) -> None:
            try:
                for line in iter(pipe.readline, ""):
                    buffer.append(line)
                    # Emit line in real-time so users see progress.
                    print(line, end="", flush=True)
            finally:
                pipe.close()

        with subprocess.Popen(
            command,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        ) as proc:
            t_out = threading.Thread(target=_reader, args=(proc.stdout, stdout_chunks))
            t_err = threading.Thread(target=_reader, args=(proc.stderr, stderr_chunks))
            t_out.start()
            t_err.start()
            try:
                returncode = proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                t_out.join()
                t_err.join()
                return "".join(stdout_chunks), "".join(stderr_chunks), -1
            t_out.join()
            t_err.join()
        return "".join(stdout_chunks), "".join(stderr_chunks), returncode

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = BashToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )

        if not input_data.command:
            return ToolResult(data=None, success=False, error="Empty command.")

        # Use the Rust sandbox runner when the compiled extension is available.
        try:
            from huginn_ext.sandbox import (
                run_sandboxed,  # type: ignore[import-not-found]
            )

            allowed_base_dirs = [str(work_dir.resolve()), str(Path.cwd().resolve())]
            result = run_sandboxed(
                command=input_data.command[0],
                args=input_data.command[1:],
                cwd=str(work_dir),
                timeout=input_data.timeout,
                allowed_base_dirs=allowed_base_dirs,
            )
            if not result["success"]:
                error = (
                    result.get("stderr")
                    or result.get("message")
                    or "Sandboxed command failed."
                )
            else:
                error = None
            return ToolResult(
                data={
                    "command": input_data.command,
                    "returncode": result["returncode"],
                    "stdout": result["stdout"],
                    "stderr": result["stderr"],
                    "message": result["message"],
                    "timed_out": result["timed_out"],
                },
                success=result["success"],
                error=error,
            )
        except Exception:
            # Rust extension not available; proceed to the configured backend.
            pass

        try:
            executor = get_executor()
        except SandboxError as exc:
            return ToolResult(
                data=None, success=False, error=f"Execution blocked: {exc}"
            )

        if isinstance(executor, ContainerExecutor):
            result = executor.run(
                input_data.command,
                cwd=work_dir,
                timeout=input_data.timeout,
                capture_output=True,
                text=True,
            )
            return ToolResult(
                data={
                    "command": input_data.command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "message": (
                        "Command succeeded." if result.success else "Command failed."
                    ),
                    "container": True,
                },
                success=result.success,
            )

        # SandboxExecutor path — uses executable whitelist + work-dir validation.
        if isinstance(executor, SandboxExecutor):
            try:
                result = executor.run(
                    input_data.command,
                    cwd=work_dir,
                    timeout=input_data.timeout,
                    capture_output=input_data.capture_output,
                    text=True,
                )
                return ToolResult(
                    data={
                        "command": input_data.command,
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr,
                        "message": (
                            "Command succeeded."
                            if result.returncode == 0
                            else "Command failed."
                        ),
                        "sandbox": True,
                    },
                    success=result.returncode == 0,
                )
            except SandboxError as e:
                return ToolResult(
                    data=None, success=False,
                    error=f"Sandbox blocked command: {e}",
                )
            except Exception as e:
                return ToolResult(
                    data=None, success=False,
                    error=f"Sandbox execution failed: {e}",
                )
