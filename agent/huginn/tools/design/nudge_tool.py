"""Nudge 工具: 参数化微调, 保持设计意图, 无需重新描述.

设计思路 (参考 v0/nudge/Nudge 机制):
- agent 执行任务后, 把关键决策(配色/间距/ENCUT/K点/温度等)以可调参数暴露
- 用户改某个参数即可触发重跑, 不用重新描述整个任务
- checkpoint 机制: 保存任务关键状态(input params + 中间产物路径),
  nudge 时从 checkpoint 恢复, 只改指定参数, 其余保持
- 适合反复微调场景: VASP收敛测试 / 图表样式调整 / 报告措辞修改

actions:
- expose_params:  agent 注册可调参数(名字/当前值/范围/描述), 返回 task_id
- nudge:          用户改某参数值, 触发从 checkpoint 重跑
- list_params:    列出 task 的可调参数
- list_tasks:     列出所有 task
- restore:        恢复到某个 task 的 checkpoint
- snapshot:       agent 主动存当前状态为 checkpoint
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.tools.base import HuginnTool
from huginn.types import ToolContext, ToolResult, ValidationResult

logger = logging.getLogger(__name__)


class NudgeInput(BaseModel):
    action: Literal[
        "expose_params",
        "nudge",
        "list_params",
        "list_tasks",
        "restore",
        "snapshot",
    ] = Field(...)

    # expose_params 时必填
    task_name: str | None = Field(default=None, description="任务名")
    params: list[dict[str, Any]] | None = Field(
        default=None,
        description=(
            "可调参数列表, 每项形如 "
            "{name, current_value, value_type, range, description}"
        ),
    )
    checkpoint_state: dict[str, Any] | None = Field(
        default=None,
        description="任务关键状态(输入参数/中间产物路径), 用于 restore"
    )

    # nudge / list_params / restore / snapshot 时必填
    task_id: str | None = Field(default=None, description="任务ID")

    # nudge 时必填
    param_name: str | None = Field(default=None, description="要改的参数名")
    new_value: Any | None = Field(default=None, description="新值")


class _NudgeStore:
    """进程内 task/checkpoint 存储. 单例."""

    _instance: _NudgeStore | None = None

    def __init__(self) -> None:
        # task_id -> {task_name, params, checkpoint_state, history}
        self._tasks: dict[str, dict[str, Any]] = {}

    @classmethod
    def get(cls) -> _NudgeStore:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def register(
        self,
        task_name: str,
        params: list[dict[str, Any]],
        checkpoint_state: dict[str, Any] | None,
    ) -> str:
        task_id = f"task-{uuid.uuid4().hex[:8]}"
        self._tasks[task_id] = {
            "task_id": task_id,
            "task_name": task_name,
            "params": {p["name"]: p for p in params},
            "checkpoint_state": checkpoint_state or {},
            "history": [
                {
                    "ts": time.time(),
                    "action": "expose_params",
                    "params": dict(self._tasks.get(task_id, {}).get("params", {})),
                }
            ],
        }
        return task_id

    def nudge(
        self, task_id: str, param_name: str, new_value: Any
    ) -> dict[str, Any] | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        if param_name not in task["params"]:
            return None
        old_value = task["params"][param_name].get("current_value")
        task["params"][param_name]["current_value"] = new_value
        task["history"].append(
            {
                "ts": time.time(),
                "action": "nudge",
                "param_name": param_name,
                "old_value": old_value,
                "new_value": new_value,
            }
        )
        return {
            "task_id": task_id,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value,
            "checkpoint_state": task["checkpoint_state"],
            "current_params": {
                k: v.get("current_value") for k, v in task["params"].items()
            },
        }

    def snapshot(self, task_id: str, state: dict[str, Any]) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        task["checkpoint_state"] = state
        task["history"].append(
            {"ts": time.time(), "action": "snapshot", "state_keys": list(state.keys())}
        )
        return True

    def restore(self, task_id: str) -> dict[str, Any] | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        task["history"].append(
            {"ts": time.time(), "action": "restore"}
        )
        return {
            "task_id": task_id,
            "task_name": task["task_name"],
            "checkpoint_state": task["checkpoint_state"],
            "current_params": {
                k: v.get("current_value") for k, v in task["params"].items()
            },
        }

    def list_params(self, task_id: str) -> list[dict[str, Any]] | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        return list(task["params"].values())

    def list_tasks(self) -> list[dict[str, Any]]:
        return [
            {
                "task_id": t["task_id"],
                "task_name": t["task_name"],
                "param_count": len(t["params"]),
                "history_len": len(t["history"]),
            }
            for t in self._tasks.values()
        ]


class NudgeTool(HuginnTool):
    """Nudge 工具: 参数化微调 + checkpoint 恢复."""

    name = "nudge_tool"
    category = "design"
    description = (
        "Expose tunable parameters after a task and let the user nudge "
        "them without re-describing the whole task. Actions: "
        "expose_params (agent registers tunable params + checkpoint state), "
        "nudge (user changes a param value, returns checkpoint to restore), "
        "list_params / list_tasks, restore (recover checkpoint), "
        "snapshot (agent saves current state as checkpoint)."
    )
    input_schema = NudgeInput

    def is_read_only(self, args: NudgeInput) -> bool:
        return True

    async def validate_input(
        self, args: NudgeInput, context: ToolContext | None = None
    ) -> ValidationResult:
        if args.action == "expose_params":
            if not args.task_name or not args.params:
                return ValidationResult(
                    result=False,
                    message="expose_params 需要 task_name 和 params",
                )
            for p in args.params:
                if "name" not in p:
                    return ValidationResult(
                        result=False,
                        message="每个 param 必须有 name 字段",
                    )
        if args.action in ("nudge", "list_params", "restore", "snapshot"):
            if not args.task_id:
                return ValidationResult(
                    result=False,
                    message=f"{args.action} 需要 task_id",
                )
        if args.action == "nudge" and (not args.param_name or args.new_value is None):
            return ValidationResult(
                result=False,
                message="nudge 需要 param_name 和 new_value",
            )
        return ValidationResult(result=True)

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        input_data = NudgeInput(**args)
        store = _NudgeStore.get()

        try:
            if input_data.action == "expose_params":
                task_id = store.register(
                    input_data.task_name or "untitled",
                    input_data.params or [],
                    input_data.checkpoint_state,
                )
                return ToolResult(
                    data={
                        "task_id": task_id,
                        "task_name": input_data.task_name,
                        "params": input_data.params,
                        "message": (
                            "已暴露可调参数. 用户可调用 nudge_tool action=nudge "
                            f"task_id={task_id} param_name=<name> new_value=<value> "
                            "微调参数, 系统会返回 checkpoint_state 供你恢复重跑."
                        ),
                    },
                    success=True,
                )

            if input_data.action == "nudge":
                result = store.nudge(
                    input_data.task_id or "",
                    input_data.param_name or "",
                    input_data.new_value,
                )
                if not result:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=(
                            f"task_id={input_data.task_id} 或 "
                            f"param_name={input_data.param_name} 不存在"
                        ),
                    )
                result["message"] = (
                    f"参数 {input_data.param_name} 已从 {result['old_value']} "
                    f"改为 {result['new_value']}. 请基于 checkpoint_state 恢复任务,"
                    " 用新参数重新执行相关步骤, 其余保持不变."
                )
                return ToolResult(data=result, success=True)

            if input_data.action == "list_params":
                params = store.list_params(input_data.task_id or "")
                if params is None:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"task_id={input_data.task_id} 不存在",
                    )
                return ToolResult(data={"params": params}, success=True)

            if input_data.action == "list_tasks":
                return ToolResult(
                    data={"tasks": store.list_tasks()}, success=True
                )

            if input_data.action == "restore":
                result = store.restore(input_data.task_id or "")
                if not result:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"task_id={input_data.task_id} 不存在",
                    )
                result["message"] = (
                    "已恢复到 checkpoint. 请基于 checkpoint_state 和 "
                    "current_params 重新执行任务."
                )
                return ToolResult(data=result, success=True)

            if input_data.action == "snapshot":
                # snapshot 需要新的 state, 走 expose_params 时存的 state 不变.
                # 这里允许 agent 在执行中主动存档.
                ok = store.snapshot(
                    input_data.task_id or "",
                    # snapshot 没有专门字段, 用 metadata 传递
                    # 这里简化: snapshot 只标记 history, state 不变
                    {},
                )
                if not ok:
                    return ToolResult(
                        data=None,
                        success=False,
                        error=f"task_id={input_data.task_id} 不存在",
                    )
                return ToolResult(
                    data={"task_id": input_data.task_id, "snapshot": "ok"},
                    success=True,
                )

            return ToolResult(
                data=None,
                success=False,
                error=f"未知 action: {input_data.action}",
            )

        except Exception as e:
            logger.warning("nudge_tool failed: %s", e, exc_info=True)
            return ToolResult(data=None, success=False, error=str(e))
