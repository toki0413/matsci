"""P0 接线验证 — 验证补强在 agent 工具路径 (ToolAdapter) 上真正生效.

覆盖:
- Hashline 自动闭环: agent 读文件 → 外部修改 → agent 编辑(不带 hash) 被 adapter
  自动注入 expected_hash 并拒绝, 不覆盖外部改动.
- Hashline 正常路径: agent 读 → 编辑(无并发) 成功.
- LSP 经由 adapter 可用: 符号重命名同步更新所有引用.
"""

from __future__ import annotations

import pytest

from huginn.permissions import PermissionConfig
from huginn.tools.adapter import _HASHLINE_CACHE, ToolAdapter
from huginn.tools.file_edit_tool import FileEditTool
from huginn.tools.file_read_tool import FileReadTool

pytestmark = pytest.mark.asyncio


def _perm() -> PermissionConfig:
    return PermissionConfig(auto_approve_all=True)


@pytest.fixture(autouse=True)
def _clear_hashline_cache():
    _HASHLINE_CACHE.clear()
    yield
    _HASHLINE_CACHE.clear()


async def test_hashline_adapter_blocks_external_edit(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world\n", encoding="utf-8")

    adapter = ToolAdapter()
    # agent 读文件 -> adapter 记录 "agent 所见版本" 的 hash
    read = adapter.adapt(FileReadTool(), permission_config=_perm())
    await read.ainvoke({"file_path": str(p), "working_dir": str(tmp_path)})
    assert _HASHLINE_CACHE.get(str(p.resolve())) is not None

    # 外部进程改了文件
    p.write_text("code changed by someone else\n", encoding="utf-8")

    # agent 编辑 (调用方不带 expected_hash) -> adapter 自动注入旧 hash -> 拒绝
    edit = adapter.adapt(FileEditTool(), permission_config=_perm())
    r = await edit.ainvoke(
        {
            "file_path": str(p),
            "old_string": "someone",
            "new_string": "agent",
            "working_dir": str(tmp_path),
        }
    )
    assert "changed since snapshot" in r.get("error", "")
    # 外部修改被保留, 未被覆盖
    assert p.read_text(encoding="utf-8") == "code changed by someone else\n"


async def test_hashline_adapter_normal_edit_succeeds(tmp_path):
    p = tmp_path / "a.txt"
    p.write_text("hello world\n", encoding="utf-8")

    adapter = ToolAdapter()
    read = adapter.adapt(FileReadTool(), permission_config=_perm())
    await read.ainvoke({"file_path": str(p), "working_dir": str(tmp_path)})

    edit = adapter.adapt(FileEditTool(), permission_config=_perm())
    r = await edit.ainvoke(
        {
            "file_path": str(p),
            "old_string": "world",
            "new_string": "huginn",
            "working_dir": str(tmp_path),
        }
    )
    assert "error" not in r, r
    assert p.read_text(encoding="utf-8") == "hello huginn\n"


async def test_lsp_rename_via_adapter(tmp_path):
    pytest.importorskip("jedi")
    from huginn.tools.lsp_tool import LspTool

    p = tmp_path / "m.py"
    p.write_text("def add(a, b):\n    return a + b\nx = add(1, 2)\n", encoding="utf-8")

    t = ToolAdapter().adapt(LspTool(), permission_config=_perm())
    r = await t.ainvoke(
        {
            "action": "rename",
            "file_path": str(p),
            "line": 0,
            "character": 4,
            "name": "my_add",
            "working_dir": str(tmp_path),
        }
    )
    assert "error" not in r, r
    new = p.read_text(encoding="utf-8")
    assert "def my_add(" in new
    assert "my_add(1, 2)" in new
    assert "def add(" not in new


async def test_hashline_cache_primed_by_edit_too(tmp_path):
    """编辑成功本身也会更新缓存, 后续基于新版本的编辑正常."""
    p = tmp_path / "a.txt"
    p.write_text("hello world\n", encoding="utf-8")

    adapter = ToolAdapter()
    read = adapter.adapt(FileReadTool(), permission_config=_perm())
    await read.ainvoke({"file_path": str(p), "working_dir": str(tmp_path)})

    edit = adapter.adapt(FileEditTool(), permission_config=_perm())
    r1 = await edit.ainvoke(
        {
            "file_path": str(p),
            "old_string": "world",
            "new_string": "huginn",
            "working_dir": str(tmp_path),
        }
    )
    assert "error" not in r1, r1
    # 第一次编辑后缓存应指向新内容
    assert _HASHLINE_CACHE[str(p.resolve())] is not None

    # 第二次编辑基于新版本 -> 成功
    r2 = await edit.ainvoke(
        {
            "file_path": str(p),
            "old_string": "huginn",
            "new_string": "huginn-agent",
            "working_dir": str(tmp_path),
        }
    )
    assert "error" not in r2, r2
    assert p.read_text(encoding="utf-8") == "hello huginn-agent\n"
