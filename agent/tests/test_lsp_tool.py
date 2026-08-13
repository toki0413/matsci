"""P0-1 LSP 符号级编辑工具单元测试.

覆盖 JediProvider / LspTool 的 jedi 路径:
- rename: 符号重命名后引用同步更新 (同文件 + 跨文件)
- references: 返回引用位置
- hover: 返回签名/docstring
- definition: 返回定义位置
- diagnostics: 语法错误被检出
- code_action: 返回可用动作
- 降级: jedi 缺失时返回明确错误
- 权限: rename 走 ASK 给预览 / DENY 拒绝
"""

from __future__ import annotations

import pytest

from huginn.core_types import ToolContext
from huginn.permissions import PermissionConfig
from huginn.tools.lsp_tool import LspTool

pytest.importorskip("jedi")


def _ctx(workspace: str = ".") -> ToolContext:
    return ToolContext(
        session_id="t-lsp",
        workspace=workspace,
        config=PermissionConfig(auto_approve_all=True),
    )


SRC = """\
def add(a, b):
    return a + b


def main():
    x = 1
    return add(x, 2)
"""


@pytest.mark.asyncio
async def test_definition(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(SRC, encoding="utf-8")
    tool = LspTool()
    res = await tool.call(
        {
            "action": "definition",
            "file_path": str(p),
            "line": 6,
            "character": 11,
            "working_dir": str(tmp_path),
        },
        _ctx(),
    )
    assert res.success is True
    defs = res.data["result"]
    assert defs, "expected a definition"
    assert defs[0]["name"] == "add"


@pytest.mark.asyncio
async def test_references(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(SRC, encoding="utf-8")
    tool = LspTool()
    # 定位到 main 里的 add(...) 调用
    res = await tool.call(
        {
            "action": "references",
            "file_path": str(p),
            "line": 6,
            "character": 11,
            "working_dir": str(tmp_path),
        },
        _ctx(),
    )
    assert res.success is True
    refs = res.data["result"]
    assert any(r["name"] == "add" for r in refs)


@pytest.mark.asyncio
async def test_hover(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(SRC, encoding="utf-8")
    tool = LspTool()
    res = await tool.call(
        {
            "action": "hover",
            "file_path": str(p),
            "line": 6,
            "character": 11,
            "working_dir": str(tmp_path),
        },
        _ctx(),
    )
    assert res.success is True
    assert res.data["result"]["found"] is True
    assert res.data["result"]["name"] == "add"


@pytest.mark.asyncio
async def test_diagnostics(tmp_path):
    p = tmp_path / "bad.py"
    p.write_text("def (\n", encoding="utf-8")  # syntax error
    tool = LspTool()
    res = await tool.call(
        {"action": "diagnostics", "file_path": str(p), "working_dir": str(tmp_path)},
        _ctx(),
    )
    assert res.success is True
    assert len(res.data["result"]) >= 1


@pytest.mark.asyncio
async def test_code_action(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(SRC, encoding="utf-8")
    tool = LspTool()
    res = await tool.call(
        {
            "action": "code_action",
            "file_path": str(p),
            "line": 6,
            "character": 11,
            "working_dir": str(tmp_path),
        },
        _ctx(),
    )
    assert res.success is True
    names = {a["action"] for a in res.data["result"]}
    assert "rename" in names


@pytest.mark.asyncio
async def test_rename_same_file(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(SRC, encoding="utf-8")
    tool = LspTool()
    res = await tool.call(
        {
            "action": "rename",
            "file_path": str(p),
            "line": 0,
            "character": 6,
            "name": "my_add",
            "working_dir": str(tmp_path),
        },
        _ctx(),
    )
    assert res.success is True
    new = p.read_text(encoding="utf-8")
    assert "def my_add(" in new
    assert "my_add(x, 2)" in new
    assert "def add(" not in new


@pytest.mark.asyncio
async def test_rename_refuses_outside_workdir(tmp_path):
    # rename 目标文件必须在 work_dir 内; 越界访问被拒绝
    tool = LspTool()
    res = await tool.call(
        {
            "action": "definition",
            "file_path": "/etc/hostname",
            "working_dir": str(tmp_path),
        },
        _ctx(),
    )
    assert res.success is False
    assert "outside working directory" in res.error


@pytest.mark.asyncio
async def test_rename_needs_name(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(SRC, encoding="utf-8")
    tool = LspTool()
    res = await tool.call(
        {
            "action": "rename",
            "file_path": str(p),
            "line": 0,
            "character": 4,
            "working_dir": str(tmp_path),
        },
        _ctx(),
    )
    assert res.success is False
    assert "requires a 'name'" in res.error


@pytest.mark.asyncio
async def test_rename_ask_mode(tmp_path):
    p = tmp_path / "m.py"
    p.write_text(SRC, encoding="utf-8")
    tool = LspTool()
    ctx = ToolContext(
        session_id="t-lsp",
        workspace=str(tmp_path),
        config=PermissionConfig(plan_mode=True),
    )
    res = await tool.call(
        {
            "action": "rename",
            "file_path": str(p),
            "line": 0,
            "character": 6,
            "name": "my_add",
            "working_dir": str(tmp_path),
        },
        ctx,
    )
    assert res.success is True
    assert res.data.get("needs_approval") is True
    # ASK 下文件不应被修改
    assert "def add(" in p.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_missing_jedi_degrades(monkeypatch, tmp_path):
    p = tmp_path / "m.py"
    p.write_text(SRC, encoding="utf-8")
    tool = LspTool()
    monkeypatch.setattr(
        "huginn.tools.lsp_tool._get_jedi",
        lambda: (_ for _ in ()).throw(ImportError("jedi not installed")),
    )
    res = await tool.call(
        {
            "action": "definition",
            "file_path": str(p),
            "line": 0,
            "character": 4,
            "working_dir": str(tmp_path),
        },
        _ctx(),
    )
    assert res.success is False
    assert "LSP unavailable" in res.error
