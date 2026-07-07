"""Tests for BashTool."""

from __future__ import annotations

import sys

import pytest

from huginn.tools.bash_tool import BashTool
from huginn.types import ToolContext


class TestBashTool:
    async def test_run_echo(self):
        tool = BashTool()
        # Windows: use python -c "print('hello')" since echo is a shell builtin
        if sys.platform == "win32":
            cmd = ["python", "-c", "print('hello')"]
        else:
            cmd = ["echo", "hello"]
        result = await tool.call(
            {"command": cmd},
            ToolContext(session_id="test", workspace="."),
        )
        assert result.success is True
        assert "hello" in result.data["stdout"]

    async def test_stream_echo(self):
        tool = BashTool()
        result = await tool.call(
            {"command": ["python", "-c", "print('streamed')"], "stream": True},
            ToolContext(session_id="test", workspace="."),
        )
        assert result.success is True
        assert "streamed" in result.data["stdout"]
