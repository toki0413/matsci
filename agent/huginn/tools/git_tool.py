"""Git tool — read-only git introspection for the Coder agent.

Provides status, diff, and log without mutating the repository.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class GitToolInput(BaseModel):
    action: Literal["status", "diff", "log"] = Field(default="status")
    working_dir: str | None = Field(default=None)
    max_lines: int = Field(default=200, ge=1)


class GitTool(HuginnTool):
    """Read-only git introspection."""

    name = "git_tool"
    description = "Run read-only git commands: status, diff, log."
    input_schema = GitToolInput

    def is_read_only(self, args: GitToolInput) -> bool:
        return True

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = GitToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )

        cmd_map = {
            "status": ["git", "status", "--short"],
            "diff": ["git", "diff"],
            "log": ["git", "log", "--oneline", "-20"],
        }
        cmd = cmd_map.get(input_data.action)
        if cmd is None:
            return ToolResult(
                data=None,
                success=False,
                error=f"Unknown git action: {input_data.action}",
            )

        try:
            result = subprocess.run(
                cmd,
                cwd=str(work_dir),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=30.0,
            )
            output = (result.stdout + result.stderr).splitlines()[
                : input_data.max_lines
            ]
            return ToolResult(
                data={
                    "action": input_data.action,
                    "output": "\n".join(output),
                    "truncated": len(result.stdout.splitlines()) > input_data.max_lines,
                    "message": f"Git {input_data.action} completed.",
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Git tool failed: {e}")
