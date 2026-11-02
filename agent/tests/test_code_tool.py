"""Tests for the generic code execution tool."""

import os
from pathlib import Path

import pytest

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


@pytest.mark.parametrize(
    "mangled",
    [
        "C﹕＼Users＼wanzh＼Desktop＼matsci-agent",  # 全角冒号 + 反斜杠
        "CUserswanzhDesktopmatsci-agentagent",  # 分隔符被吃掉
    ],
)
def test_code_tool_rejects_working_dir_that_is_not_a_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mangled: str
) -> None:
    """畸形 working_dir 必须报错, 不能在 cwd 下建出同名目录."""
    monkeypatch.chdir(tmp_path)
    result = CodeTool().call(
        {
            "action": "execute",
            "code": "print('hello')",
            "working_dir": mangled,
        }
    )
    assert result.success is False
    assert "working_dir" in result.error
    assert not (tmp_path / mangled).exists()


def test_code_tool_input_schema() -> None:
    """CodeToolInput should validate parameters."""
    inp = CodeToolInput(
        action="execute",
        code="print('hello')",
        timeout=30.0,
    )
    assert inp.language == "python"
    assert inp.timeout == 30.0
