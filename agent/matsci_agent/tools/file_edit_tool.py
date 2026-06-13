"""File edit tool — precise string replacement in text files.

Used by the Coder agent to make minimal surgical edits. Always requires approval.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from matsci_agent.tools.base import MatSciTool
from matsci_agent.types import ToolResult, ToolContext


class FileEditToolInput(BaseModel):
    action: Literal["edit"] = Field(default="edit")
    file_path: str = Field(..., description="Path to file")
    old_string: str = Field(..., description="Exact text to replace")
    new_string: str = Field(..., description="Replacement text")
    working_dir: str | None = Field(default=None)


class FileEditTool(MatSciTool):
    """Edit text files by replacing a unique string."""

    name = "file_edit_tool"
    description = "Replace a unique substring in a text file with new content."
    destructive = True
    input_schema = FileEditToolInput

    def call(self, args: dict[str, Any], context: ToolContext | None = None) -> ToolResult:
        input_data = FileEditToolInput(**args)
        work_dir = Path(input_data.working_dir) if input_data.working_dir else Path.cwd()
        path = work_dir / input_data.file_path
        if not path.is_absolute():
            path = path.resolve()

        # Prevent editing outside the working directory tree
        try:
            path.relative_to(work_dir.resolve())
        except ValueError:
            return ToolResult(
                data=None,
                success=False,
                error=f"Refusing to edit outside working directory: {path}",
            )

        if not path.exists():
            return ToolResult(data=None, success=False, error=f"File not found: {path}")

        try:
            content = path.read_text(encoding="utf-8")
            if content.count(input_data.old_string) != 1:
                return ToolResult(
                    data=None,
                    success=False,
                    error=(
                        f"old_string occurs {content.count(input_data.old_string)} times; "
                        "must occur exactly once."
                    ),
                )
            new_content = content.replace(input_data.old_string, input_data.new_string, 1)
            path.write_text(new_content, encoding="utf-8")
            return ToolResult(
                data={
                    "file_path": str(path),
                    "replacements": 1,
                    "message": f"Edited {path}.",
                },
                success=True,
            )
        except Exception as e:
            return ToolResult(data=None, success=False, error=f"Failed to edit file: {e}")
