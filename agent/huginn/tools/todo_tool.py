"""轻量 coding todo 工具 — 会话级任务清单 CRUD.

plan_store_tool 偏研究计划 (持久化 + 确认门 + 状态机), 对纯 coding todo 太重.
这个工具走进程级单例 + 按 session_id 分桶, 整列表替换式 (跟 Claude Code 的
TodoWrite 一致), 不持久化, 不走确认门.

ponytail: 不引入新依赖, 不改 ToolContext. 进程级 dict 存.
ceiling: 进程重启丢, 跨进程不可见. 升级路径: 接 session_state 持久化.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolResult


# 进程级单例: {session_id: [todo_dict, ...]}
# 跨工具调用共享, 但不跨进程. session_id 为空时用一个固定桶.
_TODO_STORE: dict[str, list[dict]] = {}


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
        "管理 coding 任务清单 (会话级, 不持久化). 整列表替换式: "
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
        _TODO_STORE[session_id] = bucket
        completed = sum(1 for t in bucket if t.get("status") == "completed")
        out = TodoWriteOutput(
            todos=bucket, total=len(bucket), completed=completed,
        )
        return ToolResult(
            data=out.model_dump(),
            success=True,
            side_effects=[f"todos updated: {len(bucket)} items, {completed} done"],
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
        bucket = _TODO_STORE.get(session_id, [])
        completed = sum(1 for t in bucket if t.get("status") == "completed")
        out = TodoReadOutput(
            todos=bucket, total=len(bucket), completed=completed,
        )
        return ToolResult(data=out.model_dump(), success=True)
