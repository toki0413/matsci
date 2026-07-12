"""Grep tool — search file contents with regex.

Read-only tool using re module for content search.
The explore subagent spec already references "grep" — this makes it real.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult


class GrepToolInput(BaseModel):
    action: Literal["search"] = Field(default="search")
    pattern: str = Field(..., description="Regular expression to search for")
    path: str = Field(default=".", description="File or directory to search in")
    file_glob: str = Field(
        default="**/*",
        description="File pattern to search within (e.g. '**/*.py')",
    )
    max_results: int = Field(default=50, ge=1, description="Max matches to return")
    case_insensitive: bool = Field(default=False)


class GrepTool(HuginnTool):
    """Search file contents with regex."""

    name = "grep"
    category = "core"
    description = (
        "Search file contents using regex. Returns matching lines with file paths and line numbers."
    )
    input_schema = GrepToolInput
    destructive = False
    read_only = True

    def is_read_only(self, args: GrepToolInput) -> bool:
        return True

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = GrepToolInput(**args)
        base = Path(input_data.path).resolve()

        if not base.exists():
            return ToolResult(data=None, success=False, error=f"Path not found: {base}")

        # path traversal protection
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

        flags = re.IGNORECASE if input_data.case_insensitive else 0
        try:
            regex = re.compile(input_data.pattern, flags)
        except re.error as e:
            return ToolResult(data=None, success=False, error=f"Invalid regex: {e}")

        # collect files to search
        if base.is_file():
            files = [base]
        else:
            files = [f for f in base.glob(input_data.file_glob) if f.is_file()]

        matches: list[dict] = []
        for f in files:
            try:
                content = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                continue
            for line_no, line in enumerate(content.splitlines(), 1):
                if regex.search(line):
                    matches.append({
                        "file": str(
                            f.relative_to(base.parent) if base.is_dir() else f.name
                        ),
                        "line": line_no,
                        "text": line[:200],  # truncate long lines
                    })
                    if len(matches) >= input_data.max_results:
                        break
            if len(matches) >= input_data.max_results:
                break

        msg = f"Found {len(matches)} matches."
        if len(matches) >= input_data.max_results:
            msg += f" (max {input_data.max_results} reached)"

        return ToolResult(
            data={"matches": matches, "count": len(matches), "message": msg},
            success=True,
        )
