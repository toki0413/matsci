"""phase_tool — 让 agent 主动查/补阶段门证据, 或强制放行.

autoloop engine 在阶段转移点插了 PhaseGateHook, 证据不足时阻断.
agent 通过这个工具:
- 查当前门状态 (哪个转移在等证据 / 上一条门决策)
- 补证据 (累积进共享状态, 等下一轮评估)
- 主动请求评审 (拿累积证据跑一次 evaluate)
- override 强制放行某个转移 (需人工确认, 走 ASK)

共享状态走 get_shared_phase_gate_state(), 与 engine.py 同一实例,
所以 tool 写进去的 evidence / override engine 下一轮立刻能看到.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from huginn.autoloop.phase_gate import (
    PhaseGate,
    PhaseGateConfig,
    PhaseGateHook,
    get_shared_phase_gate_state,
)
from huginn.permissions import PermissionConfig
from huginn.phases import ResearchPhase
from huginn.tools.base import HuginnTool
from huginn.tools.profile import ToolProfile
from huginn.types import ToolContext, ToolResult


class PhaseToolInput(BaseModel):
    action: Literal[
        "get_current_gate", "submit_evidence", "request_review", "override"
    ] = Field(
        description=(
            "get_current_gate: 查当前 pending 转移 + 上一条门决策; "
            "submit_evidence: 累积证据进共享状态; "
            "request_review: 用累积证据跑一次门评估; "
            "override: 强制放行某转移 (需人工确认)"
        )
    )
    from_phase: str | None = Field(
        default=None,
        description="阶段名 (perceive/hypothesize/plan/execute/validate/learn/report). request_review / override 必填.",
    )
    to_phase: str | None = Field(
        default=None,
        description="目标阶段名. request_review / override 必填.",
    )
    evidence: dict[str, Any] | None = Field(
        default=None,
        description="submit_evidence / request_review 用: 要补/评估的证据 dict.",
    )


class PhaseTool(HuginnTool):
    """阶段门查询 / 补证据 / 评审 / 强制放行."""

    name = "phase_tool"
    category = "meta"
    description = (
        "查询 autoloop 阶段门状态, 补证据, 请求评审, 或强制放行. "
        "门阻断时 agent 用它补齐证据或请求 override."
    )
    input_schema = PhaseToolInput
    profile = ToolProfile(
        cost_tier="light",
        phases=frozenset({ResearchPhase.OPEN}),
    )

    def __init__(self) -> None:
        # tool 自己持一个 hook 实例, 默认证据清单. reviewer_fn 不传,
        # 只做硬性证据检查; engine 那边的 hook 同样默认无 reviewer.
        self._hook = PhaseGateHook(config=PhaseGateConfig())

    async def call(
        self, args: dict[str, Any], context: ToolContext | None = None
    ) -> ToolResult:
        # schema 校验在 try 外, ValidationError 传播, 不吞进 ToolResult
        input_data = PhaseToolInput(**args)
        try:
            if input_data.action == "get_current_gate":
                return self._get_current_gate()
            if input_data.action == "submit_evidence":
                return self._submit_evidence(input_data)
            if input_data.action == "request_review":
                return self._request_review(input_data)
            if input_data.action == "override":
                return await self._override(input_data, context)
            return ToolResult(
                data=None, success=False, error=f"未知 action: {input_data.action}"
            )
        except Exception as exc:
            return ToolResult(data=None, success=False, error=f"Phase tool failed: {exc}")

    # ── action 实现 ──────────────────────────────────────────────

    def _get_current_gate(self) -> ToolResult:
        state = get_shared_phase_gate_state()
        last = state.last_gate()
        return ToolResult(
            data={
                "pending_transition": (
                    list(state.pending_transition)
                    if state.pending_transition
                    else None
                ),
                "last_gate": last.to_dict() if last else None,
                "submitted_evidence_keys": list(state.submitted_evidence.keys()),
                "overrides": [list(t) for t in state.overrides] if state.overrides else [],
            },
            success=True,
        )

    def _submit_evidence(self, input_data: PhaseToolInput) -> ToolResult:
        if not input_data.evidence:
            return ToolResult(
                data=None,
                success=False,
                error="submit_evidence 需要 evidence (dict)",
            )
        state = get_shared_phase_gate_state()
        # 合并进累积证据, 后提交的同名 key 覆盖前者
        state.submitted_evidence.update(input_data.evidence)
        return ToolResult(
            data={
                "accumulated_keys": list(state.submitted_evidence.keys()),
                "message": "证据已累积. 用 request_review 跑门评估.",
            },
            success=True,
        )

    def _request_review(self, input_data: PhaseToolInput) -> ToolResult:
        if not input_data.from_phase or not input_data.to_phase:
            return ToolResult(
                data=None,
                success=False,
                error="request_review 需要 from_phase 和 to_phase",
            )
        state = get_shared_phase_gate_state()
        # 累积证据 + 本次传入的 evidence 合并, 本次传入优先
        merged: dict[str, Any] = dict(state.submitted_evidence)
        if input_data.evidence:
            merged.update(input_data.evidence)

        gate = self._hook.evaluate(
            input_data.from_phase, input_data.to_phase, merged
        )
        state.history.append(gate)
        state.pending_transition = (
            input_data.from_phase,
            input_data.to_phase,
        )
        return ToolResult(data={"gate": gate.to_dict()}, success=True)

    async def _override(
        self, input_data: PhaseToolInput, context: ToolContext | None
    ) -> ToolResult:
        if not input_data.from_phase or not input_data.to_phase:
            return ToolResult(
                data=None,
                success=False,
                error="override 需要 from_phase 和 to_phase",
            )
        # override 是强制放行, 视为写动作, 走 ASK. auto_approve_all 才直接执行.
        if _needs_approval(context):
            return ToolResult(
                data={
                    "dry_run": True,
                    "needs_approval": True,
                    "from_phase": input_data.from_phase,
                    "to_phase": input_data.to_phase,
                    "message": (
                        "override 强制放行需要人工确认. "
                        "确认后再次调用 (auto_approve) 即生效."
                    ),
                },
                success=True,
            )
        state = get_shared_phase_gate_state()
        key = (input_data.from_phase, input_data.to_phase)
        state.overrides.add(key)
        return ToolResult(
            data={
                "overridden": True,
                "from_phase": input_data.from_phase,
                "to_phase": input_data.to_phase,
                "message": "已放行. engine 下次到这个转移会直接推进.",
            },
            success=True,
        )


def _needs_approval(context: ToolContext | None) -> bool:
    """override 是否需要人工确认. plan_mode 或非 auto_approve_all 都要确认."""
    if context is None:
        return True
    cfg = getattr(context, "config", None)
    if isinstance(cfg, PermissionConfig):
        if cfg.plan_mode:
            return True
        return not cfg.auto_approve_all
    return True
