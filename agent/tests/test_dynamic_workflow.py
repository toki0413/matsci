"""Tests for dynamic workflow orchestration (W3 A5).

Locks the behaviour added in A5:
- WorkflowScript.from_dict: parse subtasks, defaults, clamping
- WorkflowOrchestrator.run: concurrent dispatch, semaphore cap, aggregation
- Partial failure: failed subtask doesn't crash siblings
- Tool-not-registered: marked failed, not crash
- WorkflowRegistry shared singleton: submit / get / cancel / list_active / clear
- WorkflowTool: submit_script / status / cancel / collect actions
- Engine integration: _execute_dynamic_workflow routes via orchestrator
- Budget: dynamic_workflow mode rejected in medium+light tiers (like workflow)

All tools are fakes registered into ToolRegistry; no real solver calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.autoloop.dynamic_workflow import (
    Subtask,
    SubtaskResult,
    WorkflowOrchestrator,
    WorkflowRegistry,
    WorkflowResult,
    WorkflowScript,
    get_shared_workflow_registry,
    set_shared_workflow_registry,
)
from huginn.tools.registry import ToolRegistry
from huginn.types import ToolContext, ToolResult


# ── fake tools ───────────────────────────────────────────────────────────────


class _FakeTool:
    """Minimal fake tool for orchestrator tests. callable via .call(args, ctx)."""

    def __init__(self, name: str, delay: float = 0.0, output: str = "ok", fail: bool = False):
        self.name = name
        self._delay = delay
        self._output = output
        self._fail = fail
        self.call_count = 0

    async def call(self, args: dict, context: ToolContext | None = None) -> ToolResult:
        self.call_count += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise RuntimeError(f"{self.name} forced failure")
        return ToolResult(data={"tool": self.name, "args": args, "result": self._output}, success=True)


def _register_fakes(*tools: _FakeTool) -> None:
    """Clear ToolRegistry and register the given fakes."""
    ToolRegistry.clear()
    for t in tools:
        ToolRegistry.register(t)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset ToolRegistry + shared workflow registry before and after each test."""
    ToolRegistry.clear()
    set_shared_workflow_registry(None)
    yield
    ToolRegistry.clear()
    set_shared_workflow_registry(None)


# ── WorkflowScript.from_dict ─────────────────────────────────────────────────


class TestWorkflowScriptParse:
    def test_basic_parse(self):
        script = WorkflowScript.from_dict({
            "objective": "test",
            "subtasks": [
                {"id": "s1", "tool": "fake_a", "args": {"x": 1}},
                {"id": "s2", "tool": "fake_b"},
            ],
        })
        assert script.objective == "test"
        assert len(script.subtasks) == 2
        assert script.subtasks[0].id == "s1"
        assert script.subtasks[0].tool_name == "fake_a"
        assert script.subtasks[0].args == {"x": 1}
        assert script.subtasks[1].args == {}
        assert script.max_concurrent == 8  # default

    def test_max_concurrent_clamped(self):
        script = WorkflowScript.from_dict({
            "objective": "t",
            "max_concurrent": 100,
            "subtasks": [{"tool": "a"}],
        })
        assert script.max_concurrent == 64  # clamped to 64

    def test_max_concurrent_floor(self):
        script = WorkflowScript.from_dict({
            "objective": "t",
            "max_concurrent": 0,
            "subtasks": [{"tool": "a"}],
        })
        assert script.max_concurrent == 1

    def test_auto_ids_for_missing(self):
        script = WorkflowScript.from_dict({
            "objective": "t",
            "subtasks": [{"tool": "a"}, {"tool": "b"}],
        })
        assert script.subtasks[0].id == "sub_1"
        assert script.subtasks[1].id == "sub_2"

    def test_skips_subtasks_without_tool(self):
        script = WorkflowScript.from_dict({
            "objective": "t",
            "subtasks": [
                {"tool": "a"},
                {"id": "s2"},  # no tool → skipped
                {"tool": ""},   # empty tool → skipped
                "not a dict",   # non-dict → skipped
            ],
        })
        assert len(script.subtasks) == 1
        assert script.subtasks[0].tool_name == "a"

    def test_tool_name_alias(self):
        """subtask can use 'tool_name' key as alias for 'tool'."""
        script = WorkflowScript.from_dict({
            "objective": "t",
            "subtasks": [{"tool_name": "alias_tool"}],
        })
        assert script.subtasks[0].tool_name == "alias_tool"

    def test_to_dict_roundtrip(self):
        script = WorkflowScript.from_dict({
            "objective": "rt",
            "max_concurrent": 4,
            "subtasks": [{"id": "s1", "tool": "a", "args": {"k": "v"}, "description": "d"}],
        })
        d = script.to_dict()
        assert d["objective"] == "rt"
        assert d["max_concurrent"] == 4
        assert d["n_subtasks"] == 1
        assert d["subtasks"][0]["tool"] == "a"


# ── WorkflowOrchestrator.run ─────────────────────────────────────────────────


class TestOrchestratorRun:
    def test_concurrent_dispatch_all_complete(self):
        a = _FakeTool("a", output="A")
        b = _FakeTool("b", output="B")
        _register_fakes(a, b)
        script = WorkflowScript(
            id="wf_test1", objective="o",
            subtasks=[Subtask(id="s1", tool_name="a"), Subtask(id="s2", tool_name="b")],
        )
        orch = WorkflowOrchestrator(max_concurrent=8)
        result = asyncio.run(orch.run(script))
        assert result.status == "completed"
        assert result.n_completed == 2
        assert result.n_failed == 0
        assert result.success is True
        assert a.call_count == 1
        assert b.call_count == 1

    def test_empty_subtasks_completes_immediately(self):
        script = WorkflowScript(id="wf_empty", objective="o", subtasks=[])
        orch = WorkflowOrchestrator()
        result = asyncio.run(orch.run(script))
        assert result.status == "completed"
        assert result.n_total == 0

    def test_partial_failure_doesnt_crash_siblings(self):
        a = _FakeTool("a", output="A")
        bad = _FakeTool("bad", fail=True)
        _register_fakes(a, bad)
        script = WorkflowScript(
            id="wf_partial", objective="o",
            subtasks=[
                Subtask(id="s1", tool_name="a"),
                Subtask(id="s2", tool_name="bad"),
                Subtask(id="s3", tool_name="a"),
            ],
        )
        orch = WorkflowOrchestrator(max_concurrent=8)
        result = asyncio.run(orch.run(script))
        assert result.status == "completed"  # workflow completes even with failures
        assert result.n_completed == 2
        assert result.n_failed == 1
        assert result.success is True  # at least one completed
        sr = result.subtask_results["s2"]
        assert sr.status == "failed"
        assert "RuntimeError" in sr.error

    def test_unregistered_tool_marked_failed(self):
        a = _FakeTool("a")
        _register_fakes(a)
        script = WorkflowScript(
            id="wf_missing", objective="o",
            subtasks=[
                Subtask(id="s1", tool_name="a"),
                Subtask(id="s2", tool_name="nonexistent"),
            ],
        )
        orch = WorkflowOrchestrator()
        result = asyncio.run(orch.run(script))
        assert result.n_completed == 1
        assert result.n_failed == 1
        sr = result.subtask_results["s2"]
        assert sr.status == "failed"
        assert "not registered" in sr.error

    def test_semaphore_caps_concurrency(self):
        """With max_concurrent=2 and 4 slow subtasks, at most 2 run at once."""
        import time

        concurrent = {"current": 0, "max": 0}

        class _Tracker(_FakeTool):
            async def call(self, args, context=None):
                concurrent["current"] += 1
                concurrent["max"] = max(concurrent["max"], concurrent["current"])
                await asyncio.sleep(0.05)
                concurrent["current"] -= 1
                return ToolResult(data="ok", success=True)

        tools = [_Tracker(f"t{i}") for i in range(4)]
        _register_fakes(*tools)
        script = WorkflowScript(
            id="wf_sem", objective="o",
            subtasks=[Subtask(id=f"s{i}", tool_name=f"t{i}") for i in range(4)],
            max_concurrent=2,
        )
        orch = WorkflowOrchestrator(max_concurrent=8)
        asyncio.run(orch.run(script))
        assert concurrent["max"] <= 2

    def test_result_to_dict_serializable(self):
        a = _FakeTool("a", output={"key": "value"})
        _register_fakes(a)
        script = WorkflowScript(
            id="wf_ser", objective="o",
            subtasks=[Subtask(id="s1", tool_name="a")],
        )
        orch = WorkflowOrchestrator()
        result = asyncio.run(orch.run(script))
        d = result.to_dict()
        assert d["status"] == "completed"
        assert d["n_total"] == 1
        assert "s1" in d["subtasks"]
        # output should be serialized (ToolResult.to_dict)
        assert d["subtasks"]["s1"]["status"] == "completed"


# ── WorkflowRegistry shared singleton ────────────────────────────────────────


class TestWorkflowRegistry:
    def test_get_set_singleton(self):
        set_shared_workflow_registry(None)
        r1 = get_shared_workflow_registry()
        r2 = get_shared_workflow_registry()
        assert r1 is r2

    def test_set_injects_custom(self):
        custom = WorkflowRegistry()
        set_shared_workflow_registry(custom)
        assert get_shared_workflow_registry() is custom

    def test_submit_returns_pending_result(self):
        a = _FakeTool("a")
        _register_fakes(a)
        reg = WorkflowRegistry()
        set_shared_workflow_registry(reg)
        script = WorkflowScript(
            id="wf_reg1", objective="o",
            subtasks=[Subtask(id="s1", tool_name="a")],
        )
        result = reg.submit(script)
        assert result.id == "wf_reg1"
        # submit 从同步上下文调 (没 running loop), 不开后台任务, 停在 pending
        assert result.status == "pending"
        # collect 同步跑一遍, 结果就地更新
        asyncio.run(reg.collect("wf_reg1"))
        assert result.status == "completed"
        assert result.n_completed == 1

    def test_get_returns_result(self):
        a = _FakeTool("a")
        _register_fakes(a)
        reg = WorkflowRegistry()
        set_shared_workflow_registry(reg)
        script = WorkflowScript(
            id="wf_reg2", objective="o",
            subtasks=[Subtask(id="s1", tool_name="a")],
        )
        reg.submit(script)
        assert reg.get("wf_reg2") is not None
        assert reg.get("nonexistent") is None

    def test_cancel_running(self):
        a = _FakeTool("a", delay=0.5)
        _register_fakes(a)
        reg = WorkflowRegistry()
        set_shared_workflow_registry(reg)
        script = WorkflowScript(
            id="wf_cancel", objective="o",
            subtasks=[Subtask(id="s1", tool_name="a")],
        )
        reg.submit(script)
        ok = reg.cancel("wf_cancel")
        assert ok is True
        assert reg.get("wf_cancel").status == "cancelled"

    def test_cancel_already_done_returns_false(self):
        a = _FakeTool("a")
        _register_fakes(a)
        reg = WorkflowRegistry()
        set_shared_workflow_registry(reg)
        script = WorkflowScript(
            id="wf_done", objective="o",
            subtasks=[Subtask(id="s1", tool_name="a")],
        )
        reg.submit(script)
        # 同步跑完, 否则 result 停在 pending, cancel 会返回 True
        asyncio.run(reg.collect("wf_done"))
        assert reg.get("wf_done").status == "completed"
        assert reg.cancel("wf_done") is False

    def test_cancel_nonexistent_returns_false(self):
        reg = WorkflowRegistry()
        assert reg.cancel("nope") is False

    def test_list_active(self):
        a = _FakeTool("a", delay=0.3)
        _register_fakes(a)
        reg = WorkflowRegistry()
        set_shared_workflow_registry(reg)
        reg.submit(WorkflowScript(
            id="wf_active", objective="o",
            subtasks=[Subtask(id="s1", tool_name="a")],
        ))
        assert "wf_active" in reg.list_active()

    def test_clear_removes_finished(self):
        a = _FakeTool("a")
        _register_fakes(a)
        reg = WorkflowRegistry()
        set_shared_workflow_registry(reg)
        reg.submit(WorkflowScript(
            id="wf_clr", objective="o",
            subtasks=[Subtask(id="s1", tool_name="a")],
        ))
        # 同步跑完, clear 才会把它清掉 (只清 completed/failed/cancelled)
        asyncio.run(reg.collect("wf_clr"))
        assert reg.get("wf_clr") is not None
        reg.clear()
        assert reg.get("wf_clr") is None


# ── WorkflowTool ─────────────────────────────────────────────────────────────


class TestWorkflowTool:
    @pytest.fixture
    def tool(self):
        from huginn.tools.workflow_tool import WorkflowTool
        return WorkflowTool()

    @pytest.fixture
    def ctx(self, tmp_path):
        return ToolContext(session_id="test", workspace=str(tmp_path))

    def _call(self, tool, args, ctx=None):
        return asyncio.run(tool.call(args, ctx))

    def test_submit_script_returns_id(self, tool, ctx):
        a = _FakeTool("a")
        _register_fakes(a)
        result = self._call(tool, {
            "action": "submit_script",
            "script": {
                "objective": "test",
                "subtasks": [{"id": "s1", "tool": "a"}],
            },
        }, ctx)
        assert result.success is True
        assert result.data["workflow_id"].startswith("wf_")
        assert result.data["n_subtasks"] == 1

    def test_submit_script_missing_script(self, tool, ctx):
        result = self._call(tool, {"action": "submit_script"}, ctx)
        assert result.success is False
        assert "script" in result.error

    def test_submit_script_no_subtasks(self, tool, ctx):
        result = self._call(tool, {
            "action": "submit_script",
            "script": {"objective": "o", "subtasks": []},
        }, ctx)
        assert result.success is False

    def test_status_returns_summary(self, tool, ctx):
        a = _FakeTool("a")
        _register_fakes(a)
        # submit first
        r = self._call(tool, {
            "action": "submit_script",
            "script": {"objective": "o", "subtasks": [{"id": "s1", "tool": "a"}]},
        }, ctx)
        wf_id = r.data["workflow_id"]
        # submit 在一个 asyncio.run 里建的 background task, 那个 loop 关了
        # task 就死了, result 停在 pending. collect 同步跑一遍补上结果.
        asyncio.run(get_shared_workflow_registry().collect(wf_id))
        # query status
        result = self._call(tool, {"action": "status", "workflow_id": wf_id})
        assert result.success is True
        assert result.data["status"] == "completed"
        assert result.data["n_total"] == 1

    def test_status_missing_id(self, tool, ctx):
        result = self._call(tool, {"action": "status"})
        assert result.success is False

    def test_status_not_found(self, tool, ctx):
        result = self._call(tool, {"action": "status", "workflow_id": "wf_nope"})
        assert result.success is False

    def test_collect_blocking_with_timeout(self, tool, ctx):
        a = _FakeTool("a")
        _register_fakes(a)
        r = self._call(tool, {
            "action": "submit_script",
            "script": {"objective": "o", "subtasks": [{"id": "s1", "tool": "a"}]},
        }, ctx)
        wf_id = r.data["workflow_id"]
        result = self._call(tool, {
            "action": "collect", "workflow_id": wf_id, "timeout": 2.0,
        })
        assert result.success is True
        assert result.data["status"] == "completed"
        assert "subtasks" in result.data

    def test_collect_non_blocking(self, tool, ctx):
        a = _FakeTool("a")
        _register_fakes(a)
        r = self._call(tool, {
            "action": "submit_script",
            "script": {"objective": "o", "subtasks": [{"id": "s1", "tool": "a"}]},
        }, ctx)
        wf_id = r.data["workflow_id"]
        asyncio.run(asyncio.sleep(0.1))
        result = self._call(tool, {
            "action": "collect", "workflow_id": wf_id,
        })
        assert result.success is True
        assert result.data["status"] == "completed"

    def test_cancel_active(self, tool, ctx):
        a = _FakeTool("a", delay=0.5)
        _register_fakes(a)
        r = self._call(tool, {
            "action": "submit_script",
            "script": {"objective": "o", "subtasks": [{"id": "s1", "tool": "a"}]},
        }, ctx)
        wf_id = r.data["workflow_id"]
        result = self._call(tool, {"action": "cancel", "workflow_id": wf_id})
        assert result.success is True
        assert result.data["cancelled"] is True

    def test_cancel_missing_id(self, tool, ctx):
        result = self._call(tool, {"action": "cancel"})
        assert result.success is False

    def test_unknown_action_rejected_by_schema(self, tool, ctx):
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            self._call(tool, {"action": "bogus"}, ctx)


# ── engine integration ──────────────────────────────────────────────────────


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from huginn.autoloop.engine import AutoloopEngine

    monkeypatch.setattr("huginn.autoloop.engine.get_model", lambda s: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.MemoryManager", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    eng = AutoloopEngine(workspace=tmp_path)

    class _DummyTracker:
        def start_task(self, *a, **kw): ...
        def update(self, *a, **kw): ...
        def complete(self, *a, **kw): ...
        def fail(self, *a, **kw): ...

    eng.progress_tracker = _DummyTracker()
    return eng


class TestEngineDynamicWorkflow:
    def test_execute_dynamic_workflow_dispatches(self, engine):
        """_execute with mode=dynamic_workflow routes to orchestrator."""
        a = _FakeTool("a", output="A")
        b = _FakeTool("b", output="B")
        _register_fakes(a, b)
        plan = {
            "mode": "dynamic_workflow",
            "description": "parallel test",
            "script": {
                "objective": "run 2 tools in parallel",
                "subtasks": [
                    {"id": "s1", "tool": "a"},
                    {"id": "s2", "tool": "b"},
                ],
            },
        }
        result = asyncio.run(engine._execute(plan, {}))
        assert result["mode"] == "dynamic_workflow"
        assert result["success"] is True
        assert result["n_total"] == 2
        assert result["n_completed"] == 2
        assert result["n_failed"] == 0

    def test_execute_dynamic_workflow_partial_failure(self, engine):
        a = _FakeTool("a")
        bad = _FakeTool("bad", fail=True)
        _register_fakes(a, bad)
        plan = {
            "mode": "dynamic_workflow",
            "description": "partial",
            "script": {
                "objective": "o",
                "subtasks": [
                    {"id": "s1", "tool": "a"},
                    {"id": "s2", "tool": "bad"},
                ],
            },
        }
        result = asyncio.run(engine._execute(plan, {}))
        assert result["success"] is True  # at least one completed
        assert result["n_completed"] == 1
        assert result["n_failed"] == 1

    def test_execute_dynamic_workflow_empty_script(self, engine):
        plan = {
            "mode": "dynamic_workflow",
            "description": "empty",
            "script": {"objective": "o", "subtasks": []},
        }
        result = asyncio.run(engine._execute(plan, {}))
        assert result["success"] is False
        assert "无有效 subtask" in result["error"]

    def test_execute_dynamic_workflow_json_string(self, engine):
        """plan['script'] can be a JSON string (agent may pass string)."""
        import json

        a = _FakeTool("a")
        _register_fakes(a)
        plan = {
            "mode": "dynamic_workflow",
            "description": "json string",
            "script": json.dumps({
                "objective": "o",
                "subtasks": [{"id": "s1", "tool": "a"}],
            }),
        }
        result = asyncio.run(engine._execute(plan, {}))
        assert result["success"] is True
        assert result["n_completed"] == 1

    def test_budget_rejects_dynamic_workflow_in_medium_tier(self, engine):
        """dynamic_workflow is a heavy mode — rejected in medium tier (like workflow)."""
        from huginn.autoloop.budget import ProgressiveBudget

        engine._budget = ProgressiveBudget.default()
        engine._budget_degraded = False
        engine._budget_rejects = {}
        engine._speculator_hint = ""
        # iter 15 = medium tier (open is 1-10, medium is 11-30)
        assert engine._check_budget(15, {"mode": "dynamic_workflow"}) is False
        assert "medium" in engine._speculator_hint

    def test_budget_rejects_dynamic_workflow_in_light_tier(self, engine):
        from huginn.autoloop.budget import ProgressiveBudget

        engine._budget = ProgressiveBudget.default()
        engine._budget_degraded = False
        engine._budget_rejects = {}
        engine._speculator_hint = ""
        # iter 35 = light tier (light is 31-50)
        assert engine._check_budget(35, {"mode": "dynamic_workflow"}) is False
        assert "light" in engine._speculator_hint

    def test_budget_allows_dynamic_workflow_in_open_tier(self, engine):
        from huginn.autoloop.budget import ProgressiveBudget

        engine._budget = ProgressiveBudget.default()
        engine._budget_degraded = False
        engine._budget_rejects = {}
        engine._speculator_hint = ""
        # iter 3 = open tier (no restriction)
        assert engine._check_budget(3, {"mode": "dynamic_workflow"}) is True
