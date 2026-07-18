"""Prospective memory tools — 让 LLM 自己注册"未来要做的事".

3 个工具: schedule_intention / list_pending_intentions / cancel_intention.
落 ProspectiveMemory (workspace/.huginn/prospective.jsonl). 失败返回错误字符串,
不抛异常, 不阻塞 agent 主流程.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from huginn.memory.prospective import (
    ProspectiveIntention,
    ProspectiveMemory,
    _new_intention_id,
)
from huginn.tools.base import HuginnTool
from huginn.types import ToolResult


def _pm_from_context(context: Any) -> ProspectiveMemory | None:
    """从 ToolContext 拿 workspace, 起一个 ProspectiveMemory. 拿不到就返回 None."""
    workspace = getattr(context, "workspace", None) if context else None
    if not workspace:
        return None
    try:
        return ProspectiveMemory(Path(workspace))
    except Exception:
        return None


# ponytail: source_step 拿不到当前 step, 写 0. 升级: 从 ToolContext 注入 current_step.
def _make_intention(description: str, trigger_type: str, trigger_payload: dict, priority: int) -> ProspectiveIntention:
    return ProspectiveIntention(
        intention_id=_new_intention_id(),
        description=description,
        trigger_type=trigger_type,
        trigger_payload=trigger_payload,
        priority=priority,
        created_at=time.time(),
        source_step=0,
    )


# ── Tool 1: schedule_intention ──────────────────────────────────────────────

class ScheduleIntentionInput(BaseModel):
    description: str = Field(description="要记住的未来行动, 如 '复现 Figure 3 的 bootstrap CI'")
    trigger_type: str = Field(
        description=(
            "触发类型, 4 选 1: "
            "time (到点触发) / event (事件触发) / "
            "dependency (依赖 step 触发) / condition (条件表达式触发)"
        )
    )
    trigger_payload: dict = Field(
        description=(
            "触发参数, 按类型填. 示例:\n"
            '  time:        {"when": "2026-07-19T10:00"}\n'
            '  event:       {"event": "data_ready"}\n'
            '  dependency:  {"depends_on_step": 42}\n'
            '  condition:   {"expr": "memory_recall > 0.7"}'
        )
    )
    priority: int = Field(default=5, ge=0, le=9, description="优先级 0-9, 9 最高")


class ScheduleIntentionTool(HuginnTool[ScheduleIntentionInput, None]):
    name = "schedule_intention"
    category = "meta"
    description = (
        "注册一个前瞻意图 (prospective intention) — 记住'未来某个时间/事件/依赖/条件满足时要做 X'. "
        "支持 4 类触发: "
        "time (到点: {\"when\": \"2026-07-19T10:00\"}), "
        "event (事件: {\"event\": \"data_ready\"}), "
        "dependency (依赖 step: {\"depends_on_step\": 42}), "
        "condition (条件表达式: {\"expr\": \"memory_recall > 0.7\"}). "
        "每轮 agent loop 开始时由 scan_and_fire 检查并触发满足条件者."
    )
    destructive = False
    read_only = False
    input_schema = ScheduleIntentionInput

    async def call(self, args: ScheduleIntentionInput, context) -> ToolResult:
        try:
            pm = _pm_from_context(context)
            if pm is None:
                return ToolResult(data=None, success=False, error="workspace 不可用, 无法注册前瞻意图")
            intention = _make_intention(
                args.description, args.trigger_type, args.trigger_payload, args.priority
            )
            iid = pm.store(intention)
            return ToolResult(
                data={"intention_id": iid},
                success=True,
                side_effects=[f"scheduled intention {iid} ({args.trigger_type})"],
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"schedule_intention failed: {exc}")


# ── Tool 2: list_pending_intentions ─────────────────────────────────────────

class ListPendingIntentionsInput(BaseModel):
    pass


class ListPendingIntentionsTool(HuginnTool[ListPendingIntentionsInput, None]):
    name = "list_pending_intentions"
    category = "meta"
    description = (
        "列出当前所有 pending 状态的 prospective intentions, 按 priority 降序. "
        "已 fired / cancelled 的不返回. 用于查看自己之前计划了哪些待执行的事."
    )
    destructive = False
    read_only = True
    input_schema = ListPendingIntentionsInput

    async def call(self, args: ListPendingIntentionsInput, context) -> ToolResult:
        try:
            pm = _pm_from_context(context)
            if pm is None:
                return ToolResult(data=None, success=False, error="workspace 不可用, 无法列出意图")
            pending = pm.list_pending()
            items = [
                {
                    "intention_id": it.intention_id,
                    "description": it.description,
                    "trigger_type": it.trigger_type,
                    "trigger_payload": it.trigger_payload,
                    "priority": it.priority,
                    "source_step": it.source_step,
                    "created_at": it.created_at,
                }
                for it in pending
            ]
            return ToolResult(data={"pending": items, "count": len(items)}, success=True)
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"list_pending_intentions failed: {exc}")


# ── Tool 3: cancel_intention ────────────────────────────────────────────────

class CancelIntentionInput(BaseModel):
    intention_id: str = Field(description="要取消的 intention id (pim_xxx)")


class CancelIntentionTool(HuginnTool[CancelIntentionInput, None]):
    name = "cancel_intention"
    category = "meta"
    description = "取消一个 pending 状态的 prospective intention. 已 fired / cancelled 的取消失败."
    destructive = False
    read_only = False
    input_schema = CancelIntentionInput

    async def call(self, args: CancelIntentionInput, context) -> ToolResult:
        try:
            pm = _pm_from_context(context)
            if pm is None:
                return ToolResult(data=None, success=False, error="workspace 不可用, 无法取消意图")
            ok = pm.cancel(args.intention_id)
            if not ok:
                return ToolResult(
                    data=None,
                    success=False,
                    error=f"intention {args.intention_id} 不存在或非 pending 状态",
                )
            return ToolResult(
                data={"intention_id": args.intention_id, "cancelled": True},
                success=True,
                side_effects=[f"cancelled intention {args.intention_id}"],
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"cancel_intention failed: {exc}")


# ── self-check ─────────────────────────────────────────────────────────────
# 跟 self_observe_tool 同样的 inline self-check 模式: 造临时 workspace, 跑
# schedule → list → cancel 往返, 验证三个工具的端到端 wiring 没断. 不覆盖
# ProspectiveMemory 内部逻辑 (prospective.py 自己有完整 self-check).
if __name__ == "__main__" and "--self-check" in __import__("sys").argv:
    import asyncio
    import shutil
    import sys as _sys
    import tempfile
    from huginn.types import ToolContext

    _tmp = Path(tempfile.mkdtemp(prefix="pim_tool_selfcheck_"))
    try:
        _ctx = ToolContext(session_id="selfcheck", workspace=str(_tmp))

        async def _in(tool, d):
            return tool.input_schema(**d)

        async def _run():
            # 1. schedule 一个 time intention (远未来, 不触发)
            s = ScheduleIntentionTool()
            r1 = await s.call(
                await _in(s, {
                    "description": "selfcheck intention",
                    "trigger_type": "time",
                    "trigger_payload": {"when": "2099-01-01T00:00"},
                    "priority": 7,
                }),
                _ctx,
            )
            assert r1.success and r1.data["intention_id"].startswith("pim_"), r1
            iid = r1.data["intention_id"]

            # 2. list_pending 应该看到它
            l = ListPendingIntentionsTool()
            r2 = await l.call(await _in(l, {}), _ctx)
            assert r2.success and r2.data["count"] == 1, r2
            assert r2.data["pending"][0]["intention_id"] == iid, r2
            assert r2.data["pending"][0]["priority"] == 7, r2

            # 3. cancel 后 list_pending 应该为空
            c = CancelIntentionTool()
            r3 = await c.call(await _in(c, {"intention_id": iid}), _ctx)
            assert r3.success and r3.data["cancelled"], r3
            r4 = await l.call(await _in(l, {}), _ctx)
            assert r4.data["count"] == 0, r4

            # 4. cancel 不存在的 id 应失败但不抛异常
            r5 = await c.call(await _in(c, {"intention_id": "pim_nonexistent"}), _ctx)
            assert not r5.success, r5

            # 5. workspace 缺失时优雅返回错误
            _ctx_no_ws = ToolContext(session_id="x", workspace="")
            r6 = await s.call(
                await _in(s, {
                    "description": "no ws",
                    "trigger_type": "event",
                    "trigger_payload": {"event": "never"},
                    "priority": 1,
                }),
                _ctx_no_ws,
            )
            assert not r6.success and r6.error, r6

            print("prospective_tool self-check PASS")

        asyncio.run(_run())
    finally:
        shutil.rmtree(_tmp, ignore_errors=True)
    _sys.exit(0)
