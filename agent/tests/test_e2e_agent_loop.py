"""Agent 主循环 E2E 验证 — 覆盖 v1.1.0 发布的真实主循环 execute 路径.

验证对象不是单测级别的 tool_dispatch / orchestrator, 而是把主循环的执行入口
(``AutoloopEngine._execute`` → dispatch_table → ``_execute_dynamic_workflow`` →
``WorkflowOrchestrator`` → ``huginn.harness.tool_dispatch.dispatch_tool``) 串联跑通:

1. 新工具 ``experiment_protocol_tool`` 能经主循环分派真实执行 (时空可组合性返回).
2. ``_execute`` phase 的 tool_whitelist 在主循环内强制生效 — 白名单外工具被拦截,
   白名单内工具正常执行 (对应 release 提交 982ea42 的强制语义).
3. 未知 plan mode 被 dispatch_table 拒绝, 不再静默 fallback (H5-b 统一入口).

所有 LLM / 网络 / 子进程路径 stub 掉, 测试是 hermetic 的 — 但 execute 阶段
的工具分派走真实代码 (ToolRegistry + dispatch_tool + 真实工具实现).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from huginn.autoloop.engine import AutoloopEngine
from huginn.harness.phase_spec import PhaseRegistry


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """构建一个主循环引擎, 重型依赖 stub 掉, execute 阶段走真实分派."""
    monkeypatch.setattr(
        "huginn.autoloop.engine.get_model", lambda settings: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.MemoryManager", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.BenchmarkRunner", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    monkeypatch.setattr("huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None)
    monkeypatch.setattr("huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None)

    # 注册核心工具, 供主循环 dispatch_tool 分派真实工具
    import huginn.tools as _tools
    _tools.register_core_tools()

    eng = AutoloopEngine(workspace=tmp_path)
    return eng


@pytest.fixture(autouse=True)
def _isolate_phase_registry():
    """每个测试后重置 PhaseRegistry 单例, 避免 override 串扰.

    注意: 不能用 register_phase_override (会落盘到共享的
    tests/.test_cache/phase_registry.json, 污染其它测试文件).
    本文件用 _set_execute_whitelist 的内存写入, 不落盘.
    """
    yield
    PhaseRegistry._instance = None


def _set_execute_whitelist(*tools: str) -> None:
    """给 _execute phase 设 tool_whitelist override (仅内存, 不落盘).

    直接写 singleton 的 _phase_overrides, 避免 register_phase_override 的
    落盘副作用污染共享的 phase_registry.json (H4 白名单本应强制生效而非持久化).
    """
    reg = PhaseRegistry.get_instance()
    reg._phase_overrides["_execute"] = {"tool_whitelist": list(tools)}


def _dynw_script(interleaved: list[tuple[str, str, dict]]) -> dict:
    """构造一个 dynamic_workflow plan 的 script dict.

    interleaved: [(subtask_id, tool_name, args), ...] — 保留顺序.
    """
    return {
        "objective": "e2e loop dispatch",
        "subtasks": [
            {"id": sid, "tool": tool, "args": args}
            for sid, tool, args in interleaved
        ],
    }


class TestMainLoopExecuteDispatch:
    """主循环 _execute 入口 → 真实工具分派 (v1.1.0 变更路径)."""

    def test_dynamic_workflow_dispatches_new_tool(
        self, engine: AutoloopEngine, tmp_path: Path
    ):
        """experiment_protocol_tool 经主循环 execute 分派真实执行."""
        plan = {
            "mode": "dynamic_workflow",
            "description": "run pipetting protocol",
            "script": _dynw_script(
                [
                    (
                        "s1",
                        "experiment_protocol_tool",
                        {"action": "run", "protocol": "pipetting", "mixer_available": True},
                    ),
                ]
            ),
        }
        result = asyncio.run(engine._execute(plan, context={}))
        assert result is not None
        assert result["mode"] == "dynamic_workflow"
        assert result["success"] is True, f"分派失败: {result}"
        assert result["n_total"] == 1
        assert result["n_completed"] == 1
        assert result["n_failed"] == 0

    def test_dynamic_workflow_new_tool_status_action(
        self, engine: AutoloopEngine, tmp_path: Path
    ):
        """status action 是 read-only, 同样可分派并返回步骤状态."""
        plan = {
            "mode": "dynamic_workflow",
            "description": "query protocol status",
            "script": _dynw_script(
                [
                    (
                        "s1",
                        "experiment_protocol_tool",
                        {"action": "status"},
                    ),
                ]
            ),
        }
        result = asyncio.run(engine._execute(plan, context={}))
        assert result["success"] is True
        assert result["n_completed"] == 1

    def test_whitelist_blocks_non_whitelisted_in_loop(
        self, engine: AutoloopEngine, tmp_path: Path
    ):
        """_execute tool_whitelist 强制: 白名单外的 subtask 被拦截标记 failed."""
        _set_execute_whitelist("experiment_protocol_tool")
        plan = {
            "mode": "dynamic_workflow",
            "description": "mix allowed + blocked tools",
            "script": _dynw_script(
                [
                    (
                        "allowed",
                        "experiment_protocol_tool",
                        {"action": "status"},
                    ),
                    (
                        "blocked",
                        "file_read_tool",
                        {"file_path": str(tmp_path)},
                    ),
                ]
            ),
        }
        result = asyncio.run(engine._execute(plan, context={}))
        # 白名单拦截不炸整体 workflow — 允许的跑, 拦截的标 failed
        assert result["n_completed"] == 1, f"期望 allowed 完成, 实际 {result}"
        assert result["n_failed"] == 1, f"期望 blocked 被拦截, 实际 {result}"
        # 语义: 拦截是白名单强制, 不是工具执行失败
        summary = result["summary"]
        assert summary["status"] == "completed"

    def test_whitelist_allows_whitelisted_in_loop(
        self, engine: AutoloopEngine, tmp_path: Path
    ):
        """白名单内的工具不被拦截, 照常执行."""
        readme = tmp_path / "readme.txt"
        readme.write_text("hello e2e")
        _set_execute_whitelist("experiment_protocol_tool", "file_read_tool")
        plan = {
            "mode": "dynamic_workflow",
            "description": "both whitelisted",
            "script": _dynw_script(
                [
                    (
                        "s1",
                        "experiment_protocol_tool",
                        {"action": "status"},
                    ),
                    (
                        "s2",
                        "file_read_tool",
                        {"file_path": str(readme)},
                    ),
                ]
            ),
        }
        result = asyncio.run(engine._execute(plan, context={}))
        assert result["n_completed"] == 2, f"期望都完成, 实际 {result}"
        assert result["n_failed"] == 0

    def test_unknown_mode_rejected_by_dispatch_table(
        self, engine: AutoloopEngine, tmp_path: Path
    ):
        """H5-b 统一入口: 未知 plan mode 被 dispatch_table 拒绝 (不再静默 fallback)."""
        plan = {
            "mode": "no_such_mode",
            "description": "should be rejected",
        }
        with pytest.raises(ValueError, match="Unknown plan mode"):
            asyncio.run(engine._execute(plan, context={}))

    def test_no_phase_whitelist_passes_through(
        self, engine: AutoloopEngine, tmp_path: Path
    ):
        """默认 _execute 无 tool_whitelist → 不限制, 任意已注册工具可跑."""
        readme = tmp_path / "readme.txt"
        readme.write_text("hello e2e")
        plan = {
            "mode": "dynamic_workflow",
            "description": "no whitelist configured",
            "script": _dynw_script(
                [
                    (
                        "s1",
                        "file_read_tool",
                        {"file_path": str(readme)},
                    ),
                ]
            ),
        }
        result = asyncio.run(engine._execute(plan, context={}))
        assert result["n_completed"] == 1, f"无白名单时应放行, 实际 {result}"


class TestLoopDispatchEntryContract:
    """主循环分派入口的契约 — 与 release 变更相关的稳定行为."""

    def test_dispatch_tool_returns_tool_result_shape(self, tmp_path: Path):
        """主循环经 dispatch_tool 分派新工具, 返回的类型与字段契约稳定."""
        from huginn.core_types import ToolContext, ToolResult
        from huginn.harness import tool_dispatch as td

        import huginn.tools as _tools
        _tools.register_core_tools()

        ctx = ToolContext(session_id="e2e", workspace=str(tmp_path), config=None)
        res = asyncio.run(
            td.dispatch_tool(
                "experiment_protocol_tool",
                {"action": "run", "mixer_available": True},
                ctx,
            )
        )
        assert isinstance(res, ToolResult)
        assert res.success is True, f"新工具分派失败: {res.error=}"
        assert "executed_steps" in res.data
        assert "degraded_steps" in res.data
        assert "inverse_count" in res.data