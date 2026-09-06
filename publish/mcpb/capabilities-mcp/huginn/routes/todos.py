"""Todo 清单端点 — 前端面板读/写 agent 的 coding todo.

与 ``huginn.tools.todo_tool`` 共用同一进程内 store (持久化到 todos.json).
agent 用 todo_write_tool 维护清单, 前端从这里拉取/勾选, 两端看到同一份.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from huginn.security.auth import require_api_key
from huginn.tools.todo_tool import get_todos, set_todos

router = APIRouter(tags=["todos"])


class TodoPayload(BaseModel):
    todos: list[dict[str, Any]] = Field(
        description="整个 todo 列表 (替换式). 每个元素含 content/status/priority."
    )


@router.get("/todos", dependencies=[Depends(require_api_key)])
async def list_todos(session_id: str = Query(default="", description="会话 ID, 空取默认桶")) -> dict[str, Any]:
    """读取指定会话的 todo 清单."""
    todos = get_todos(session_id)
    completed = sum(1 for t in todos if t.get("status") == "completed")
    return {"todos": todos, "total": len(todos), "completed": completed}


@router.put("/todos", dependencies=[Depends(require_api_key)])
async def replace_todos(
    payload: TodoPayload,
    session_id: str = Query(default="", description="会话 ID, 空取默认桶"),
) -> dict[str, Any]:
    """整列表替换指定会话的 todo (供前端勾选/清空)."""
    set_todos(session_id, payload.todos)
    todos = get_todos(session_id)
    completed = sum(1 for t in todos if t.get("status") == "completed")
    return {"todos": todos, "total": len(todos), "completed": completed}
