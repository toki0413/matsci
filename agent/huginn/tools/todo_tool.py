"""轻量 coding todo 工具 — 会话级任务清单 CRUD.

plan_store_tool 偏研究计划 (持久化 + 确认门 + 状态机), 对纯 coding todo 太重.
这里走按 session_id 分桶的整列表替换式 (跟 Claude Code 的 TodoWrite 一致),
落盘到 ``$HUGINN_CACHE_DIR/todos.json`` 防进程重启丢失, 不设确认门.

agent 侧用 todo_write_tool / todo_read_tool 维护清单, 前端面板经
``routes/todos.py`` 读写同一份 store (get_todos / set_todos), 两端看到同一份.

ponytail: 进程内单锁 + 整文件原子写, 不搞逐条目并发合并.
ceiling: 多进程同时写同一文件会互相覆盖, 没有跨进程锁.
升级路径: 换 sqlite 或引入文件锁协议.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from pydantic import BaseModel, Field

from huginn.core_types import ToolResult
from huginn.tools.base import HuginnTool
from huginn.utils.common import atomic_write_json
from huginn.utils.runtime import HUGINN_DIR_NAME

# 进程内缓存: {session_id: [todo_dict, ...]}. 空 session_id 用一个固定桶.
_TODO_STORE: dict[str, list[dict]] = {}
_store_path: Path | None = None
_lock = threading.Lock()


def _default_store_path() -> Path:
    base = os.environ.get("HUGINN_CACHE_DIR")
    if base:
        return Path(base) / "todos.json"
    return Path(HUGINN_DIR_NAME) / "todos.json"


def _load() -> None:
    """从磁盘读回 store. 损坏文件直接重来 (进程内数据不丢)."""
    global _store_path
    path = _store_path or _default_store_path()
    _store_path = path
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        for sid, items in (data.get("todos") or {}).items():
            _TODO_STORE[sid] = [t for t in items if isinstance(t, dict)]
    except (json.JSONDecodeError, TypeError, OSError):
        _TODO_STORE.clear()


def _save() -> None:
    global _store_path
    path = _store_path or _default_store_path()
    _store_path = path
    atomic_write_json(path, {"todos": _TODO_STORE}, indent=2)


def get_todos(session_id: str = "") -> list[dict]:
    with _lock:
        if _store_path is None:
            _load()
        return list(_TODO_STORE.get(session_id or "", []))


def set_todos(session_id: str, todos: list[dict]) -> None:
    with _lock:
        if _store_path is None:
            _load()
        _TODO_STORE[session_id or ""] = [t for t in todos if isinstance(t, dict)]
        _save()


class TodoItem(BaseModel):
    content: str = Field(description="任务描述.")
    status: str = Field(
        default="pending",
        description="pending | in_progress | completed",
    )
    priority: str = Field(
        default="medium",
        description="high | medium | low",
    )


class TodoWriteInput(BaseModel):
    todos: list[TodoItem] = Field(
        description="整个 todo 列表 (替换式, 不是增量). 空列表清空.",
    )


class TodoWriteOutput(BaseModel):
    todos: list[dict]
    total: int
    completed: int


class TodoWriteTool(HuginnTool[TodoWriteInput, TodoWriteOutput]):
    name = "todo_write_tool"
    category = "meta"
    description = (
        "管理 coding 任务清单 (会话级, 持久化到 todos.json). 整列表替换式: "
        "传入完整 todos 数组覆盖当前状态. 用于多步 coding 任务的进度跟踪. "
        "研究计划用 plan_store_tool, 这个只管轻量 coding todo."
    )
    destructive = False
    read_only = False
    input_schema = TodoWriteInput
    output_schema = TodoWriteOutput

    async def call(self, args: TodoWriteInput, context) -> ToolResult:
        session_id = ""
        if context is not None:
            session_id = getattr(context, "session_id", "") or ""
        bucket = [t.model_dump() for t in args.todos]
        set_todos(session_id, bucket)
        todos = get_todos(session_id)
        completed = sum(1 for t in todos if t.get("status") == "completed")
        out = TodoWriteOutput(
            todos=todos, total=len(todos), completed=completed,
        )
        return ToolResult(
            data=out.model_dump(),
            success=True,
            side_effects=[f"todos updated: {len(todos)} items, {completed} done"],
        )


class TodoReadInput(BaseModel):
    pass


class TodoReadOutput(BaseModel):
    todos: list[dict]
    total: int
    completed: int


class TodoReadTool(HuginnTool[TodoReadInput, TodoReadOutput]):
    name = "todo_read_tool"
    category = "meta"
    description = "读取当前会话的 coding todo 列表."
    destructive = False
    read_only = True
    input_schema = TodoReadInput
    output_schema = TodoReadOutput

    async def call(self, args: TodoReadInput, context) -> ToolResult:
        session_id = ""
        if context is not None:
            session_id = getattr(context, "session_id", "") or ""
        todos = get_todos(session_id)
        completed = sum(1 for t in todos if t.get("status") == "completed")
        out = TodoReadOutput(
            todos=todos, total=len(todos), completed=completed,
        )
        return ToolResult(data=out.model_dump(), success=True)
