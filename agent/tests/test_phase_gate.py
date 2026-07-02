"""Tests for phase-gate — hook 评估 / 共享状态 / phase_tool / engine 接线.

覆盖 R2 (Phase-gate 评审门) 的四层:
  * PhaseGateHook.evaluate — 硬性证据检查 / reviewer 拒绝 / 降级 / 裸值归一化
  * PhaseGateState — history / reset / last_gate / overrides
  * PhaseTool — get_current_gate / submit_evidence / request_review / override (ASK)
  * AutoloopEngine 接线 — plan→execute / execute→validate / validate→learn 门控,
    blocked 时 execute 不跑、feedback 进 _speculator_hint、override 放行
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from huginn.autoloop.engine import AutoloopEngine, _extract_tests_passed
from huginn.autoloop.phase_gate import (
    PhaseGate,
    PhaseGateConfig,
    PhaseGateHook,
    PhaseGateState,
    get_shared_phase_gate_state,
    set_shared_phase_gate_state,
)
from huginn.permissions import PermissionConfig
from huginn.tools.phase_tool import PhaseTool
from huginn.types import ToolContext


# ── 共享状态隔离: 每个 test 跑前重置共享单例 ─────────────────────


@pytest.fixture(autouse=True)
def _isolate_phase_gate_state():
    """每个 test 拿干净共享状态, 避免跨 test 污染."""
    set_shared_phase_gate_state(PhaseGateState())
    yield
    set_shared_phase_gate_state(None)


# ════════════════════════════════════════════════════════════════════
# PhaseGateHook
# ════════════════════════════════════════════════════════════════════


class TestPhaseGateHook:
    def test_approved_when_all_evidence_present(self):
        hook = PhaseGateHook()
        gate = hook.evaluate("plan", "execute", {"mode": "coder", "description": "do x"})
        assert gate.status == "approved"
        assert gate.is_blocked is False
        assert gate.missing_evidence == []

    def test_blocked_when_evidence_missing(self):
        hook = PhaseGateHook()
        gate = hook.evaluate("plan", "execute", {"mode": "coder"})
        # description 缺失
        assert gate.status == "blocked"
        assert gate.is_blocked is True
        assert "description" in gate.missing_evidence
        assert gate.feedback  # 有反馈文本

    def test_blocked_when_evidence_empty_string(self):
        """空串视同缺失."""
        hook = PhaseGateHook()
        gate = hook.evaluate(
            "plan", "execute", {"mode": "", "description": ""}
        )
        assert gate.status == "blocked"
        assert set(gate.missing_evidence) == {"mode", "description"}

    def test_reviewer_reject_returns_rejected(self):
        def reviewer(from_p, to_p, evidence):
            return False, "design has confound"
        hook = PhaseGateHook(reviewer_fn=reviewer)
        gate = hook.evaluate(
            "plan", "execute", {"mode": "coder", "description": "do x"}
        )
        assert gate.status == "rejected"
        assert gate.is_blocked is True
        assert "confound" in gate.feedback

    def test_reviewer_exception_degrades_to_approved(self):
        """reviewer 抛异常不阻断, 降级放行."""
        def reviewer(from_p, to_p, evidence):
            raise RuntimeError("LLM 挂了")
        hook = PhaseGateHook(reviewer_fn=reviewer)
        gate = hook.evaluate(
            "plan", "execute", {"mode": "coder", "description": "do x"}
        )
        assert gate.status == "approved"

    def test_non_dict_evidence_normalized(self):
        """裸值 evidence 归一成 {"value": ...}, 再查 required.
        plan→execute 要 mode+description, 裸值一定缺, 应 blocked."""
        hook = PhaseGateHook()
        gate = hook.evaluate("plan", "execute", "just a hypothesis string")
        assert gate.status == "blocked"

    def test_no_requirement_always_approved(self):
        """learn→report 无证据要求, 总放行."""
        hook = PhaseGateHook()
        gate = hook.evaluate("learn", "report", {})
        assert gate.status == "approved"
        assert gate.missing_evidence == []

    def test_custom_config_add_requirement(self):
        cfg = PhaseGateConfig()
        cfg.add_requirement("hypothesize", "plan", ["hypothesis", "rationale"])
        hook = PhaseGateHook(config=cfg)
        gate = hook.evaluate(
            "hypothesize", "plan", {"hypothesis": "h"}
        )
        assert gate.status == "blocked"
        assert "rationale" in gate.missing_evidence


# ════════════════════════════════════════════════════════════════════
# PhaseGateState
# ════════════════════════════════════════════════════════════════════


class TestPhaseGateState:
    def test_fresh_state_empty(self):
        s = PhaseGateState()
        assert s.history == []
        assert s.pending_transition is None
        assert s.submitted_evidence == {}
        assert s.overrides == set()
        assert s.last_gate() is None

    def test_last_gate_returns_most_recent(self):
        s = PhaseGateState()
        g1 = PhaseGate(from_phase="a", to_phase="b", status="approved")
        g2 = PhaseGate(from_phase="b", to_phase="c", status="blocked")
        s.history.append(g1)
        s.history.append(g2)
        assert s.last_gate() is g2

    def test_reset_clears_everything(self):
        s = PhaseGateState()
        s.history.append(PhaseGate(from_phase="a", to_phase="b", status="approved"))
        s.pending_transition = ("a", "b")
        s.submitted_evidence["mode"] = "coder"
        s.overrides.add(("a", "b"))
        s.reset()
        assert s.history == []
        assert s.pending_transition is None
        assert s.submitted_evidence == {}
        assert s.overrides == set()

    def test_shared_singleton_roundtrip(self):
        """get_shared 返回同一实例; set_shared 注入后 get 拿到新的."""
        set_shared_phase_gate_state(None)
        a = get_shared_phase_gate_state()
        b = get_shared_phase_gate_state()
        assert a is b
        fresh = PhaseGateState()
        set_shared_phase_gate_state(fresh)
        assert get_shared_phase_gate_state() is fresh


# ════════════════════════════════════════════════════════════════════
# PhaseTool
# ════════════════════════════════════════════════════════════════════


@pytest.fixture
def tool():
    return PhaseTool()


@pytest.fixture
def ctx(tmp_path):
    """auto_approve_all=True: override 直接生效."""
    return ToolContext(
        session_id="test",
        workspace=str(tmp_path),
        config=PermissionConfig(auto_approve_all=True),
    )


@pytest.fixture
def ctx_ask(tmp_path):
    """默认 ASK: override 返回 dry_run."""
    return ToolContext(
        session_id="test",
        workspace=str(tmp_path),
        config=PermissionConfig(auto_approve_all=False),
    )


def _call(tool, args, ctx):
    return asyncio.run(tool.call(args, ctx))


class TestPhaseToolGet:
    def test_get_current_gate_empty_state(self, tool, ctx):
        r = _call(tool, {"action": "get_current_gate"}, ctx)
        assert r.success is True
        assert r.data["pending_transition"] is None
        assert r.data["last_gate"] is None
        assert r.data["submitted_evidence_keys"] == []
        assert r.data["overrides"] == []

    def test_get_current_gate_reflects_state(self, tool, ctx):
        state = get_shared_phase_gate_state()
        state.pending_transition = ("plan", "execute")
        state.submitted_evidence["mode"] = "coder"
        state.overrides.add(("execute", "validate"))
        r = _call(tool, {"action": "get_current_gate"}, ctx)
        assert r.data["pending_transition"] == ["plan", "execute"]
        assert r.data["submitted_evidence_keys"] == ["mode"]
        assert ["execute", "validate"] in r.data["overrides"]


class TestPhaseToolSubmitEvidence:
    def test_submit_evidence_accumulates(self, tool, ctx):
        r1 = _call(
            tool,
            {"action": "submit_evidence", "evidence": {"mode": "coder"}},
            ctx,
        )
        assert r1.success is True
        assert r1.data["accumulated_keys"] == ["mode"]
        # 第二次提交不同 key, 累积
        r2 = _call(
            tool,
            {"action": "submit_evidence", "evidence": {"description": "do x"}},
            ctx,
        )
        assert set(r2.data["accumulated_keys"]) == {"mode", "description"}

    def test_submit_evidence_no_evidence_errors(self, tool, ctx):
        r = _call(tool, {"action": "submit_evidence"}, ctx)
        assert r.success is False
        assert "evidence" in r.error

    def test_submit_evidence_overwrites_same_key(self, tool, ctx):
        _call(tool, {"action": "submit_evidence", "evidence": {"mode": "a"}}, ctx)
        _call(tool, {"action": "submit_evidence", "evidence": {"mode": "b"}}, ctx)
        state = get_shared_phase_gate_state()
        assert state.submitted_evidence["mode"] == "b"


class TestPhaseToolRequestReview:
    def test_review_approved_with_full_evidence(self, tool, ctx):
        r = _call(
            tool,
            {
                "action": "request_review",
                "from_phase": "plan",
                "to_phase": "execute",
                "evidence": {"mode": "coder", "description": "do x"},
            },
            ctx,
        )
        assert r.success is True
        assert r.data["gate"]["status"] == "approved"
        # state 记录了这条决策
        state = get_shared_phase_gate_state()
        assert state.last_gate().status == "approved"
        assert state.pending_transition == ("plan", "execute")

    def test_review_blocked_missing_evidence(self, tool, ctx):
        r = _call(
            tool,
            {
                "action": "request_review",
                "from_phase": "plan",
                "to_phase": "execute",
                "evidence": {"mode": "coder"},
            },
            ctx,
        )
        assert r.success is True  # tool 调用成功, 只是门 blocked
        assert r.data["gate"]["status"] == "blocked"
        assert "description" in r.data["gate"]["missing_evidence"]

    def test_review_uses_accumulated_evidence(self, tool, ctx):
        """先 submit 一部分, 再 request_review 时合并."""
        _call(
            tool,
            {"action": "submit_evidence", "evidence": {"mode": "coder"}},
            ctx,
        )
        r = _call(
            tool,
            {
                "action": "request_review",
                "from_phase": "plan",
                "to_phase": "execute",
                "evidence": {"description": "do x"},
            },
            ctx,
        )
        assert r.data["gate"]["status"] == "approved"

    def test_review_missing_from_phase_errors(self, tool, ctx):
        r = _call(
            tool,
            {"action": "request_review", "to_phase": "execute"},
            ctx,
        )
        assert r.success is False
        assert "from_phase" in r.error


class TestPhaseToolOverride:
    def test_override_ask_mode_returns_dry_run(self, tool, ctx_ask):
        r = _call(
            tool,
            {
                "action": "override",
                "from_phase": "plan",
                "to_phase": "execute",
            },
            ctx_ask,
        )
        assert r.success is True
        assert r.data["dry_run"] is True
        assert r.data["needs_approval"] is True
        # 未生效: overrides 没加进去
        state = get_shared_phase_gate_state()
        assert ("plan", "execute") not in state.overrides

    def test_override_auto_mode_adds_to_overrides(self, tool, ctx):
        r = _call(
            tool,
            {
                "action": "override",
                "from_phase": "plan",
                "to_phase": "execute",
            },
            ctx,
        )
        assert r.success is True
        assert r.data["overridden"] is True
        state = get_shared_phase_gate_state()
        assert ("plan", "execute") in state.overrides

    def test_override_plan_mode_forces_approval(self, tool, tmp_path):
        """plan_mode 下即使 auto_approve_all 也要确认."""
        ctx_plan = ToolContext(
            session_id="test",
            workspace=str(tmp_path),
            config=PermissionConfig(auto_approve_all=True, plan_mode=True),
        )
        r = _call(
            tool,
            {"action": "override", "from_phase": "plan", "to_phase": "execute"},
            ctx_plan,
        )
        assert r.data["dry_run"] is True

    def test_override_missing_from_phase_errors(self, tool, ctx):
        r = _call(
            tool, {"action": "override", "to_phase": "execute"}, ctx
        )
        assert r.success is False
        assert "from_phase" in r.error


class TestPhaseToolSchema:
    def test_unknown_action_rejected_by_schema(self, tool, ctx):
        """Pydantic Literal 校验在 try 外, ValidationError 传播不进 ToolResult."""
        with pytest.raises(ValidationError):
            _call(tool, {"action": "bogus"}, ctx)


# ════════════════════════════════════════════════════════════════════
# _extract_tests_passed (engine helper)
# ════════════════════════════════════════════════════════════════════


class TestExtractTestsPassed:
    def test_dict_tests_passed_true(self):
        assert _extract_tests_passed({"tests_passed": True}) is True

    def test_dict_tests_passed_false(self):
        assert _extract_tests_passed({"tests_passed": False}) is False

    def test_dict_passed_key(self):
        assert _extract_tests_passed({"passed": False}) is False

    def test_dict_no_known_key_defaults_true(self):
        assert _extract_tests_passed({"random": "value"}) is True

    def test_string_with_fail(self):
        assert _extract_tests_passed("tests failed") is False

    def test_string_pass(self):
        assert _extract_tests_passed("all passed") is True

    def test_none_defaults_true(self):
        assert _extract_tests_passed(None) is True


# ════════════════════════════════════════════════════════════════════
# AutoloopEngine 接线
# ════════════════════════════════════════════════════════════════════


class _DummyTracker:
    def start_task(self, *a, **kw): ...
    def update(self, *a, **kw): ...
    def complete(self, *a, **kw): ...
    def fail(self, *a, **kw): ...


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """Engine with heavy sub-components stubbed, phase methods patched per-test."""
    monkeypatch.setattr(
        "huginn.autoloop.engine.get_model", lambda settings: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.MemoryManager", lambda: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda: MagicMock())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    eng = AutoloopEngine(workspace=tmp_path)
    eng.progress_tracker = _DummyTracker()
    return eng


def _patch_phases(engine, plan=None, validation=None):
    """Patch all phase methods with canned returns."""
    engine._perceive = lambda: {"changed_files": ["x.py"], "timestamp": "t"}  # type: ignore[assignment]
    engine._hypothesize = AsyncMock(return_value="test hypothesis")  # type: ignore[assignment]
    engine._plan = AsyncMock(return_value=plan or {"mode": "coder", "description": "do x"})  # type: ignore[assignment]
    engine._execute = AsyncMock(return_value={"mode": "coder", "status": "ok"})  # type: ignore[assignment]
    engine._validate = AsyncMock(return_value=validation or {"tests_passed": True})  # type: ignore[assignment]
    engine._learn = AsyncMock(return_value=None)  # type: ignore[assignment]
    engine._report = AsyncMock(return_value=str(engine.workspace / "report.md"))  # type: ignore[assignment]


class TestEngineGateIntegration:
    def test_happy_path_all_gates_pass(self, engine):
        """完整证据时三个门都放行, 6 phase + report 跑完."""
        _patch_phases(engine)
        result = asyncio.run(engine.run(objective="o", max_iterations=1))
        names = [p.name for p in result.phases]
        assert names == [
            "perceive", "hypothesize", "plan",
            "execute", "validate", "learn", "report",
        ]
        assert result.success is True

    def test_plan_to_execute_blocked_when_mode_empty(self, engine):
        """plan 缺 mode/description → 门阻断, execute 不跑."""
        _patch_phases(
            engine, plan={"mode": "", "description": ""}
        )
        asyncio.run(engine.run(objective="o", max_iterations=1))
        engine._execute.assert_not_called()
        # feedback 进了 _speculator_hint
        assert "plan" in engine._speculator_hint
        assert "execute" in engine._speculator_hint

    def test_plan_to_execute_blocked_feedback_records_missing(self, engine):
        _patch_phases(engine, plan={"mode": "", "description": ""})
        asyncio.run(engine.run(objective="o", max_iterations=1))
        state = get_shared_phase_gate_state()
        # history 里至少一条 blocked
        blocked = [g for g in state.history if g.is_blocked]
        assert blocked
        assert "mode" in blocked[0].missing_evidence
        assert "description" in blocked[0].missing_evidence

    def test_override_unblocks_plan_to_execute(self, engine):
        """override 后即使证据不足也放行, execute 跑."""
        _patch_phases(engine, plan={"mode": "", "description": ""})
        state = get_shared_phase_gate_state()
        state.overrides.add(("plan", "execute"))
        asyncio.run(engine.run(objective="o", max_iterations=1))
        engine._execute.assert_called_once()

    def test_validate_to_learn_blocked_when_tests_failed(self, engine):
        """validation tests_passed=False → learn 不跑."""
        _patch_phases(engine, validation={"tests_passed": False})
        asyncio.run(engine.run(objective="o", max_iterations=1))
        engine._learn.assert_not_called()

    def test_validate_to_learn_passes_when_tests_passed(self, engine):
        _patch_phases(engine, validation={"tests_passed": True})
        asyncio.run(engine.run(objective="o", max_iterations=1))
        engine._learn.assert_called_once()

    def test_gate_does_not_consume_extra_iterations(self, engine):
        """门阻断在 max_iter=1 时, 仍只跑 1 轮 + report, 不会多跑."""
        _patch_phases(engine, plan={"mode": "", "description": ""})
        result = asyncio.run(engine.run(objective="o", max_iterations=1))
        # perceive + hypothesize + plan (block 后 continue → 出循环) + report
        names = [p.name for p in result.phases]
        assert "execute" not in names
        assert "report" in names
