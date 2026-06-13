"""File read tool — read text files with optional line range.

A read-only tool for inspecting source code, logs, and configuration files.
Safe to auto-execute.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from matsci_agent.tools.base import MatSciTool
from matsci_agent.types import ToolResult, ToolContext


class FileReadToolInput(BaseModel):
    action: Literal["read"] = Field(default="read")
    file_path: str = Field(..., description="Path to file")
    line_offset: int = Field(default=1, ge=1, description="1-based start line")
    n_lines: int | None = Field(default=None, description="Number of lines to read")
    working_dir: str | None = Field(default=None)


class FileReadTool(MatSciTool):
    """Read text files."""

    name = "file_read_tool"
    description = "Read the contents of a text file, optionally with a line range."
    input_schema = FileReadToolInput

    def is_read_only(self, args: FileReadToolInput) -> bool:
        return True

    def call(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        input_data = FileReadToolInput(**args)
        work_dir = Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        path = work_dir / input_data.file_path
        if not path.is_absolute():
            path = path.resolve()

        if not path.exists():
            return ToolResult(data=None, success=False, error=f"File not found: {path}")
        if not path.is_file():
            return ToolResult(data=None, success=False, error=f"Not a file: {path}")

        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            total = len(lines)
            start = input_data.line_offset
            end = total + 1 if input_data.n_lines is None else start + input_data.n_lines
            selected = lines[start - 1 : end - 1]
            numbered = "\n".join(f"{i + start:4d}  {line}" for i, line in enumerate(selected))

            return ToolResult(
                data={
                    "file_path": str(path),
                    "total_lines": total,
                    "start_line": start,
                    "content": numbered,
                    "message": f"Read lines {start}-{min(end - 1, total)} of {total}.",
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Failed to read file: {e}")
