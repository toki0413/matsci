"""Tests for the generic code execution tool."""

import os
from pathlib import Path

import pytest

from huginn.core_types import ErrorKind
from huginn.tools.adapter import _timeout_failure_hint
from huginn.tools.code_tool import CodeTool, CodeToolInput

_skip_ci = os.environ.get("HUGINN_CI", "").lower() in ("1", "true", "yes")


@pytest.mark.skipif(_skip_ci, reason="CI sandbox tmp_path not found")
def test_code_tool_execute_python(tmp_path: Path) -> None:
    """CodeTool should execute Python and return stdout + result variable."""
    tool = CodeTool()
    result = tool.call(
        {
            "action": "execute",
            "code": "x = 21\ny = x * 2\nresult = {'x': x, 'y': y}",
            "result_variable": "result",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    assert result.data["result"]["y"] == 42
    assert "42" in result.data["stdout"]


def test_code_tool_generate_only() -> None:
    """CodeTool generate action should not execute code."""
    tool = CodeTool()
    result = tool.call(
        {
            "action": "generate",
            "code": "print('should not run')",
        }
    )
    assert result.success is True
    assert result.data["code"] == "print('should not run')"


@pytest.mark.skipif(_skip_ci, reason="CI sandbox tmp_path not found")
def test_code_tool_detects_output_image(tmp_path: Path) -> None:
    """CodeTool should detect PNG files saved by the executed code."""
    tool = CodeTool()
    result = tool.call(
        {
            "action": "execute",
            "code": (
                "import matplotlib\n"
                "matplotlib.use('Agg')\n"
                "from matplotlib import pyplot as plt\n"
                "plt.plot([0, 1], [0, 1])\n"
                "plt.savefig('plot.png')\n"
            ),
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    assert "plot.png" in result.data["output_files"]
    assert Path(result.data["output_files"]["plot.png"]).exists()


def test_code_tool_input_schema() -> None:
    """CodeToolInput should validate parameters."""
    inp = CodeToolInput(
        action="execute",
        code="print('hello')",
        timeout=30.0,
    )
    assert inp.language == "python"
    assert inp.timeout == 30.0


@pytest.mark.skipif(_skip_ci, reason="CI sandbox tmp_path not found")
def test_code_tool_reports_timeout_as_timeout(tmp_path: Path) -> None:
    """超时被杀必须报 timeout, 不能被下游渲染成 'sandbox/process crash'.

    ws11 复盘: 8 次 60s 超时全被 adapter 译成 "silent failure — likely a
    sandbox/process crash, not a code bug", 导致 agent 换方法而不是砍算力.
    """
    tool = CodeTool()
    result = tool.call(
        {
            "action": "execute",
            "code": "x = 0\nfor _ in range(2000000000):\n    x += 1",
            "timeout": 1.0,
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is False
    assert result.error_kind == ErrorKind.TIMEOUT
    assert result.data["timed_out"] is True
    assert result.data["timeout"] == 1.0
    assert result.error is not None
    assert "timed out" in result.error.lower()


def test_timeout_hint_never_says_crash() -> None:
    """adapter 的超时提示必须指向"砍算力", 不能指向"崩溃/换工具"."""
    hint = _timeout_failure_hint(60.0)
    low = hint.lower()
    assert "timed out" in low
    assert "> 60.0s" in hint
    assert "crash" in low  # 明确否认 crash
    assert "reduce" in low
