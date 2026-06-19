"""File write tool — create or overwrite text files.

Used by the Coder agent to create new files. Always requires approval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class FileWriteToolInput(BaseModel):
    action: Literal["write"] = Field(default="write")
    file_path: str = Field(..., description="Path to file")
    content: str = Field(..., description="Content to write")
    working_dir: str | None = Field(default=None)


class FileWriteTool(HuginnTool):
    """Write text files."""

    name = "file_write_tool"
    description = "Create or overwrite a text file with the provided content."
    destructive = True
    input_schema = FileWriteToolInput

    def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = FileWriteToolInput(**args)
        work_dir = (
            Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        )
        path = work_dir / input_data.file_path
        if not path.is_absolute():
            path = path.resolve()

        # Prevent writing outside the working directory tree
        try:
            path.relative_to(work_dir.resolve())
        except ValueError:
            return ToolResult(
                data=None,
                success=False,
                error=f"Refusing to write outside working directory: {path}",
            )

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            existed = path.exists()
            path.write_text(input_data.content, encoding="utf-8")
            return ToolResult(
                data={
                    "file_path": str(path),
                    "existed": existed,
                    "bytes_written": len(input_data.content.encode("utf-8")),
                    "message": f"Wrote {path}.",
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Failed to write file: {e}"
            )
