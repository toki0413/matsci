"""Bash tool — run shell commands inside the workspace.

Used by the Coder agent to run tests, builds, and git operations.
Always requires approval.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from matsci_agent.tools.base import MatSciTool
from matsci_agent.types import ToolResult, ToolContext


class BashToolInput(BaseModel):
    action: Literal["run"] = Field(default="run")
    command: list[str] = Field(..., description="Command as a list of arguments")
    working_dir: str | None = Field(default=None)
    timeout: float = Field(default=300.0, gt=0)
    capture_output: bool = Field(default=True)


class BashTool(MatSciTool):
    """Run shell commands in the workspace."""

    name = "bash_tool"
    description = (
        "Run a shell command as a list of arguments inside the workspace. "
        "Use for tests, builds, git, and other command-line tasks."
    )
    destructive = True
    input_schema = BashToolInput

    def call(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        input_data = BashToolInput(**args)
        work_dir = Path(input_data.working_dir) if input_data.working_dir else Path.cwd()

        if not input_data.command:
            return ToolResult(data=None, success=False, error="Empty command.")

        exe = shutil.which(input_data.command[0])
        if exe is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Executable not found: {input_data.command[0]}",
            )

        try:
            result = subprocess.run(
                input_data.command,
                cwd=str(work_dir),
                capture_output=input_data.capture_output,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=input_data.timeout,
            )
            return ToolResult(
                data={
                    "command": input_data.command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "message": "Command succeeded." if result.returncode == 0 else "Command failed.",
                },
                success=result.returncode == 0,
            )
        except subprocess.TimeoutExpired as e:
            return ToolResult(
                data={
                    "command": input_data.command,
                    "returncode": -1,
                    "stdout": e.stdout or "",
                    "stderr": e.stderr or "",
                    "message": f"Command timed out after {input_data.timeout}s.",
                },
                success=False,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Bash tool failed: {e}")
