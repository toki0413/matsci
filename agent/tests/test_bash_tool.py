"""Tests for BashTool."""

from __future__ import annotations

import sys

import pytest

from huginn.tools.bash_tool import BashTool
from huginn.types import ToolContext


class TestBashTool:
    @pytest.mark.skipif(sys.platform == "win32", reason="echo path differs on Windows")
    def test_run_echo(self):
        tool = BashTool()
        result = tool.call(
            {"command": ["echo", "hello"]},
            ToolContext(session_id="test", workspace="."),
        )
        assert result.success is True
        assert "hello" in result.data["stdout"]

    @pytest.mark.skipif(sys.platform == "win32", reason="shell path differs on Windows")
    def test_stream_echo(self, capsys):
        tool = BashTool()
        result = tool.call(
            {"command": ["python", "-c", "print('streamed')"], "stream": True},
            ToolContext(session_id="test", workspace="."),
        )
        captured = capsys.readouterr()
        assert result.success is True
        assert "streamed" in result.data["stdout"]
        assert "streamed" in captured.out
