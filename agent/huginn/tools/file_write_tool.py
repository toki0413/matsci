"""File write tool — create or overwrite text files.

Used by the Coder agent to create new files. Always requires approval.
Supports dry-run diff preview against existing content.
"""

from __future__ import annotations

import difflib
import hashlib
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class FileWriteToolInput(BaseModel):
    action: Literal["write", "preview"] = Field(default="write")
    file_path: str = Field(..., description="Path to file")
    content: str = Field(..., description="Content to write")
    working_dir: str | None = Field(default=None)
    dry_run: bool = Field(
        default=False,
        description="If True, return a diff preview without modifying the file.",
    )


def _content_hash(text: str) -> str:
    """Short SHA-256 hash for snapshot/rollback identification."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


class FileWriteTool(HuginnTool):
    """Write text files.

    When ``dry_run`` is True (or ``action`` is ``"preview"``), the tool returns
    a unified diff against the existing content without writing. On actual
    writes, the result includes a ``snapshot_hash`` of the prior content (if any)
    so callers can track what changed.
    """

    name = "file_write_tool"
    category = "core"
    description = "Create or overwrite a text file with the provided content."
    destructive = True
    input_schema = FileWriteToolInput

    async def call(
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
            existed = path.exists()
            old_content = path.read_text(encoding="utf-8") if existed else ""

            # Preview / dry-run: return the diff without writing.
            if input_data.dry_run or input_data.action == "preview":
                diff_lines = difflib.unified_diff(
                    old_content.splitlines(keepends=True),
                    input_data.content.splitlines(keepends=True),
                    fromfile=f"a/{path.name}" if existed else "/dev/null",
                    tofile=f"b/{path.name}",
                )
                return ToolResult(
                    data={
                        "file_path": str(path),
                        "dry_run": True,
                        "existed": existed,
                        "diff": "".join(diff_lines),
                        "message": f"Preview of write to {path}.",
                    },
                    success=True,
                )

            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(input_data.content, encoding="utf-8")
            result_data: dict[str, Any] = {
                "file_path": str(path),
                "existed": existed,
                "bytes_written": len(input_data.content.encode("utf-8")),
                "message": f"Wrote {path}.",
            }
            if existed:
                result_data["snapshot_hash"] = _content_hash(old_content)
            return ToolResult(data=result_data, success=True)
        except Exception as e:
            return ToolResult(
                data=None, success=False, error=f"Failed to write file: {e}"
            )
