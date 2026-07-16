"""TodoWrite / TodoRead / NotebookEdit 工具自检.

只测纯逻辑 (无 LLM / 无外部服务), 真实集成靠 register_all_tools + 端到端.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ── TodoWrite / TodoRead ───────────────────────────────────────


def _run(coro):
    # asyncio.get_event_loop() 在 Python 3.12+ 无 running loop 时会报错,
    # 直接用 asyncio.run() 每次新建并关闭 loop, 更干净
    return asyncio.run(coro)


def test_todo_write_replaces_list():
    """整列表替换式 — 传入 2 个 todo 覆盖空状态."""
    from huginn.tools.todo_tool import TodoWriteTool, _TODO_STORE
    from huginn.tools.todo_tool import TodoWriteInput

    _TODO_STORE.clear()
    tool = TodoWriteTool()
    ctx = MagicMock()
    ctx.session_id = "test-session-1"
    args = TodoWriteInput(todos=[
        {"content": "fix bug", "status": "in_progress", "priority": "high"},
        {"content": "add test", "status": "pending", "priority": "medium"},
    ])
    res = _run(tool.call(args, ctx))
    assert res.success is True
    assert res.data["total"] == 2
    assert res.data["completed"] == 0
    # 进程级存储更新
    assert len(_TODO_STORE["test-session-1"]) == 2


def test_todo_write_empty_clears():
    """空列表清空."""
    from huginn.tools.todo_tool import TodoWriteTool, _TODO_STORE
    from huginn.tools.todo_tool import TodoWriteInput

    _TODO_STORE.clear()
    _TODO_STORE["s1"] = [{"content": "x", "status": "pending"}]
    tool = TodoWriteTool()
    ctx = MagicMock()
    ctx.session_id = "s1"
    res = _run(tool.call(TodoWriteInput(todos=[]), ctx))
    assert res.success is True
    assert res.data["total"] == 0
    assert _TODO_STORE["s1"] == []


def test_todo_read_returns_current():
    """TodoRead 读当前会话 todo."""
    from huginn.tools.todo_tool import TodoReadTool, _TODO_STORE
    from huginn.tools.todo_tool import TodoReadInput

    _TODO_STORE.clear()
    _TODO_STORE["s2"] = [
        {"content": "a", "status": "completed"},
        {"content": "b", "status": "pending"},
    ]
    tool = TodoReadTool()
    ctx = MagicMock()
    ctx.session_id = "s2"
    res = _run(tool.call(TodoReadInput(), ctx))
    assert res.success is True
    assert res.data["total"] == 2
    assert res.data["completed"] == 1


def test_todo_read_empty_session():
    """空会话返回空列表."""
    from huginn.tools.todo_tool import TodoReadTool, _TODO_STORE
    from huginn.tools.todo_tool import TodoReadInput

    _TODO_STORE.clear()
    tool = TodoReadTool()
    ctx = MagicMock()
    ctx.session_id = "never-existed"
    res = _run(tool.call(TodoReadInput(), ctx))
    assert res.success is True
    assert res.data["total"] == 0
    assert res.data["todos"] == []


def test_todo_session_isolation():
    """不同 session_id 互不干扰."""
    from huginn.tools.todo_tool import TodoWriteTool, TodoReadTool, _TODO_STORE
    from huginn.tools.todo_tool import TodoWriteInput, TodoReadInput

    _TODO_STORE.clear()
    write = TodoWriteTool()
    read = TodoReadTool()

    ctx_a = MagicMock(); ctx_a.session_id = "a"
    ctx_b = MagicMock(); ctx_b.session_id = "b"

    _run(write.call(TodoWriteInput(todos=[{"content": "A", "status": "pending"}]), ctx_a))
    _run(write.call(TodoWriteInput(todos=[{"content": "B", "status": "completed"}]), ctx_b))

    res_a = _run(read.call(TodoReadInput(), ctx_a))
    res_b = _run(read.call(TodoReadInput(), ctx_b))
    assert res_a.data["todos"][0]["content"] == "A"
    assert res_b.data["todos"][0]["content"] == "B"


# ── NotebookEdit ───────────────────────────────────────────────


def _make_notebook(path: Path, n_cells: int = 2):
    """造一个最小 .ipynb 用于测试."""
    import nbformat
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_code_cell(f"# cell {i}\nprint({i})") for i in range(n_cells)]
    nbformat.write(nb, str(path))


def test_notebook_edit_missing_file():
    """文件不存在 -> 报错."""
    from huginn.tools.notebook_tool import NotebookEditTool, NotebookEditInput
    tool = NotebookEditTool()
    args = NotebookEditInput(
        notebook_path="nonexistent.ipynb",
        edit_mode="replace", cell_index=0, source="x",
    )
    res = _run(tool.call(args, None))
    assert res.success is False
    assert "not found" in res.error


def test_notebook_edit_not_ipynb():
    """非 .ipynb -> 报错."""
    from huginn.tools.notebook_tool import NotebookEditTool, NotebookEditInput
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as f:
        f.write(b"print('hi')")
        path = f.name
    try:
        tool = NotebookEditTool()
        args = NotebookEditInput(
            notebook_path=path, edit_mode="replace",
            cell_index=0, source="x",
        )
        res = _run(tool.call(args, None))
        assert res.success is False
        assert "not a .ipynb" in res.error
    finally:
        os.unlink(path)


def test_notebook_edit_replace_cell():
    """replace 第 0 个 cell."""
    from huginn.tools.notebook_tool import NotebookEditTool, NotebookEditInput
    import nbformat

    with tempfile.TemporaryDirectory() as tmp:
        nb_path = Path(tmp) / "test.ipynb"
        _make_notebook(nb_path, n_cells=2)
        tool = NotebookEditTool()
        args = NotebookEditInput(
            notebook_path=str(nb_path),
            edit_mode="replace", cell_index=0,
            cell_type="markdown", source="# replaced",
        )
        res = _run(tool.call(args, None))
        assert res.success is True
        assert res.data["total_cells"] == 2
        # 读回验证
        nb = nbformat.read(str(nb_path), as_version=4)
        assert nb.cells[0].cell_type == "markdown"
        assert nb.cells[0].source == "# replaced"


def test_notebook_edit_insert_cell():
    """insert 新 cell 到 index 1."""
    from huginn.tools.notebook_tool import NotebookEditTool, NotebookEditInput
    import nbformat

    with tempfile.TemporaryDirectory() as tmp:
        nb_path = Path(tmp) / "test.ipynb"
        _make_notebook(nb_path, n_cells=2)
        tool = NotebookEditTool()
        args = NotebookEditInput(
            notebook_path=str(nb_path),
            edit_mode="insert", cell_index=1,
            cell_type="code", source="print('new')",
        )
        res = _run(tool.call(args, None))
        assert res.success is True
        assert res.data["total_cells"] == 3
        nb = nbformat.read(str(nb_path), as_version=4)
        assert nb.cells[1].source == "print('new')"


def test_notebook_edit_delete_cell():
    """delete 第 0 个 cell."""
    from huginn.tools.notebook_tool import NotebookEditTool, NotebookEditInput
    import nbformat

    with tempfile.TemporaryDirectory() as tmp:
        nb_path = Path(tmp) / "test.ipynb"
        _make_notebook(nb_path, n_cells=3)
        tool = NotebookEditTool()
        args = NotebookEditInput(
            notebook_path=str(nb_path),
            edit_mode="delete", cell_index=0,
        )
        res = _run(tool.call(args, None))
        assert res.success is True
        assert res.data["total_cells"] == 2


def test_notebook_edit_index_out_of_range():
    """cell_index 越界 -> 报错."""
    from huginn.tools.notebook_tool import NotebookEditTool, NotebookEditInput

    with tempfile.TemporaryDirectory() as tmp:
        nb_path = Path(tmp) / "test.ipynb"
        _make_notebook(nb_path, n_cells=2)
        tool = NotebookEditTool()
        args = NotebookEditInput(
            notebook_path=str(nb_path),
            edit_mode="replace", cell_index=10, source="x",
        )
        res = _run(tool.call(args, None))
        assert res.success is False
        assert "out of range" in res.error


def test_notebook_edit_invalid_mode_validation():
    """edit_mode 非法 -> Pydantic 校验失败."""
    from huginn.tools.notebook_tool import NotebookEditInput
    with pytest.raises(Exception):
        NotebookEditInput(
            notebook_path="x.ipynb",
            edit_mode="invalid", cell_index=0, source="x",
        )
