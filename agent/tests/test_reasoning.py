"""Tests for structured external-thinking deepening (推理协议 + 自校验 + 蒸馏)."""

from __future__ import annotations

import asyncio

from huginn.core_types import ToolResult
from huginn.memory.manager import MemoryManager
from huginn.memory.reasoning import (
    ReasoningOutcome,
    ReasoningPhase,
    ReasoningRecord,
    ReasoningTrace,
)
from huginn.memory.session import ToolCallRecord
from huginn.tools.deep_think_tool import DeepThinkTool


def _run(coro):
    return asyncio.run(coro)


class TestReasoningRecord:
    def test_defaults(self):
        r = ReasoningRecord()
        assert r.phase is ReasoningPhase.THINK
        assert r.outcome is ReasoningOutcome.PENDING
        assert r.is_pending is True
        assert r.is_distillable is False

    def test_distillable_requires_confirmed_claim_and_estimate(self):
        r = ReasoningRecord(claim="x", estimate="1.0 eV")
        assert r.is_distillable is False  # 未确认
        r.outcome = ReasoningOutcome.CONFIRMED
        assert r.is_distillable is True
        r.estimate = ""
        assert r.is_distillable is False


class TestReasoningTrace:
    def test_append_recent_and_cap(self):
        t = ReasoningTrace(max_records=3)
        for i in range(5):
            t.append(ReasoningRecord(claim=f"c{i}"))
        assert len(t) == 3
        assert [r.claim for r in t.recent()] == ["c2", "c3", "c4"]

    def test_last_pending_prefers_pre_action_plan(self):
        t = ReasoningTrace()
        t.append(ReasoningRecord(phase=ReasoningPhase.THINK, claim="hyp"))
        t.append(ReasoningRecord(phase=ReasoningPhase.PRE_ACTION, claim="pred"))
        assert t.last_pending().claim == "pred"

    def test_mark_outcome_and_distillable(self):
        t = ReasoningTrace()
        r = ReasoningRecord(phase=ReasoningPhase.PRE_ACTION, claim="p", estimate="3")
        t.append(r)
        t.mark_outcome(r, ReasoningOutcome.CONFIRMED, verified_by="lammps_tool")
        assert r.outcome is ReasoningOutcome.CONFIRMED
        assert r.verified_by == "lammps_tool"
        assert [x.claim for x in t.distillable()] == ["p"]


class TestDeepThinkStructured:
    def test_structured_record_written_to_session(self):
        mm = MemoryManager()
        tool = DeepThinkTool()
        ctx = _ctx(mm)
        _run(
            tool.call(
                {
                    "analysis": "预判势能面为双阱",
                    "phase": "pre_action",
                    "hypothesis": "势能面在此构型为双阱",
                    "evidence": "对称性 + 近邻排斥",
                    "estimate": "能垒 ~0.3 eV",
                    "uncertainty": "温度效应未计入",
                    "plan": "跑过渡态搜索",
                },
                ctx,
            )
        )
        # 扁平通道仍写 (向后兼容)
        assert mm.session.reasoning_trace == ["预判势能面为双阱"]
        # 结构化侧信道
        assert len(mm.session.reasoning_records) == 1
        rec = mm.session.reasoning_records[0]
        assert rec.phase is ReasoningPhase.PRE_ACTION
        assert rec.claim == "势能面在此构型为双阱"
        assert rec.estimate == "能垒 ~0.3 eV"


def _ctx(mm):
    from huginn.core_types import ToolContext

    return ToolContext(session_id="s1", workspace="/tmp", memory_manager=mm)


class TestSelfVerifyAndDistill:
    def test_success_confirms_and_distills(self):
        mm = MemoryManager()
        mm.add_reasoning_record(
            ReasoningRecord(
                phase=ReasoningPhase.PRE_ACTION,
                claim="能垒 ~0.3 eV",
                estimate="0.3 eV",
                evidence="NEB 收敛",
            )
        )
        record = ToolCallRecord(
            tool_name="sim_lammps_tool",
            input_args={},
            result=ToolResult(data={"relax": 1}, success=True),
        )
        mm._promote_tool_result(record)
        rec = mm.session.reasoning_records[0]
        assert rec.outcome is ReasoningOutcome.CONFIRMED
        assert rec.verified_by == "sim_lammps_tool"
        hits = mm.longterm.retrieve(
            query="能垒", category="distilled_reasoning", top_k=5
        )
        assert any("0.3 eV" in str(h.get("content", h)) for h in hits)

    def test_failure_refutes_no_distill(self):
        mm = MemoryManager()
        mm.add_reasoning_record(
            ReasoningRecord(
                phase=ReasoningPhase.PRE_ACTION,
                claim="扩散势垒 0.1 eV",
                estimate="0.1 eV",
            )
        )
        record = ToolCallRecord(
            tool_name="sim_lammps_tool",
            input_args={},
            result=ToolResult(data=None, success=False, error="NEB diverged"),
        )
        mm._promote_tool_result(record)
        rec = mm.session.reasoning_records[0]
        assert rec.outcome is ReasoningOutcome.REFUTED

    def test_no_pending_record_is_noop(self):
        mm = MemoryManager()
        mm.add_reasoning_record(
            ReasoningRecord(phase=ReasoningPhase.THINK, claim="普通假设")
        )
        record = ToolCallRecord(
            tool_name="sim_lammps_tool",
            input_args={},
            result=ToolResult(data={"x": 1}, success=True),
        )
        mm._promote_tool_result(record)
        # THINK 阶段不是 self-verify 目标, 不回填
        assert mm.session.reasoning_records[0].outcome is ReasoningOutcome.PENDING
