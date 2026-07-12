"""Glob tool — find files matching a pattern.

Read-only tool using pathlib.Path.glob() for file discovery.
The explore subagent spec already references "glob" — this makes it real.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class GlobToolInput(BaseModel):
    action: Literal["search"] = Field(default="search")
    pattern: str = Field(..., description="Glob pattern (e.g. '**/*.py', '*.csv')")
    path: str = Field(default=".", description="Base directory to search from")
    max_results: int = Field(default=100, ge=1, description="Max files to return")


class GlobTool(HuginnTool):
    """Find files matching a glob pattern."""

    name = "glob"
    category = "core"
    description = (
        "Find files matching a glob pattern. "
        "Use '**/*.py' for recursive search, '*.csv' for top-level only."
    )
    input_schema = GlobToolInput
    destructive = False
    read_only = True

    def is_read_only(self, args: GlobToolInput) -> bool:
        return True

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = GlobToolInput(**args)
        base = Path(input_data.path).resolve()

        if not base.is_dir():
            return ToolResult(data=None, success=False, error=f"Not a directory: {base}")

        # path traversal protection — keep reads inside workspace
        if context and context.workspace:
            workspace = Path(context.workspace).resolve()
            try:
                base.relative_to(workspace)
            except ValueError:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"Access denied: {base} is outside workspace",
                )

        matches = sorted(base.glob(input_data.pattern))
        files = [str(m.relative_to(base)) for m in matches if m.is_file()]

        if len(files) > input_data.max_results:
            total = len(files)
            files = files[: input_data.max_results]
            msg = f"Found {total} files, showing first {input_data.max_results}."
        else:
            msg = f"Found {len(files)} files."

        return ToolResult(
            data={"files": files, "count": len(files), "message": msg},
            success=True,
        )
