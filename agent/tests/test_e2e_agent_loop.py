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


class TestFullCognitiveLoop:
    """完整认知循环 E2E — 真实 CognitiveLoop 编排 + 真实规则版 decider + 真实 execute.

    与 test_autoloop_engine (所有 phase 全 mock) 的区别: 这里保留真实
    CognitiveLoop (observe/decide/execute/reflect) + 真实规则版 decider +
    真实 ``_execute`` (dynamic_workflow → dispatch_tool). 只 mock 需要 LLM 的阶段
    (hypothesize/plan/validate/learn/report) 与阻塞型 inter-phase helper.

    验证: 整环能按 hypothesize→plan→execute→validate→learn 编排, execute 阶段
    真实分派新工具, 最终产出完整 AutoloopResult (success=True, 含 report).
    """

    def _drive(self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch) -> AutoloopResult:
        """用真实循环驱动 run_cognitive, 仅 mock LLM 阶段与阻塞 helper."""
        from unittest.mock import AsyncMock

        dync_script = _dynw_script(
            [
                (
                    "s1",
                    "experiment_protocol_tool",
                    {"action": "run", "mixer_available": True},
                ),
            ]
        )
        # 真实规则版 decider (不调 LLM 选 action)
        monkeypatch.setenv("HUGINN_COGNITIVE_LLM_DECIDER", "0")
        engine._perceive = lambda: {"changed_files": [], "timestamp": "t"}  # type: ignore[assignment]
        engine._hypothesize = AsyncMock(  # type: ignore[assignment]
            return_value="Hypothesis: pipetting protocol is composable"
        )
        engine._plan = AsyncMock(  # type: ignore[assignment]
            return_value={
                "mode": "dynamic_workflow",
                "description": "run pipetting",
                "script": dync_script,
            }
        )
        # `_execute` 保持真实
        engine._validate = AsyncMock(  # type: ignore[assignment]
            return_value={"tests_passed": True, "constraints_satisfied": True}
        )
        engine._learn = AsyncMock(return_value=None)  # type: ignore[assignment]
        engine._report = AsyncMock(  # type: ignore[assignment]
            return_value=str(engine.workspace / "report.md")
        )
        # 阻塞型 inter-phase helper 短路
        engine._maybe_clarify = AsyncMock(return_value=None)  # type: ignore[assignment]
        engine._blind_spot_pass = AsyncMock(return_value=[])  # type: ignore[assignment]
        engine._wait_if_checkpoint_pending = AsyncMock(return_value=None)  # type: ignore[assignment]
        engine._drain_side_questions = AsyncMock(return_value=0)  # type: ignore[assignment]

        from huginn.autoloop.types import AutoloopResult

        return asyncio.run(engine.run_cognitive(objective="e2e full loop", max_iterations=6))

    def test_full_loop_reaches_real_execute_and_succeeds(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ):
        """整环跑通, execute 真实分派新工具, result 契约完整."""
        from huginn.autoloop.types import AutoloopResult

        result = self._drive(engine, monkeypatch)
        assert isinstance(result, AutoloopResult)
        names = [p.name for p in result.phases]
        # 规则版 decider: hypothesize→plan→execute→validate→learn (+report)
        for expected in ("hypothesize", "plan", "execute", "validate", "learn", "report"):
            assert expected in names, f"缺 {expected} phase, 实际 {names}"
        # execute 阶段走了真实 dynamic_workflow 分派
        exec_phase = next(p for p in result.phases if p.name == "execute")
        assert exec_phase.status == "completed", f"execute 未完成: {exec_phase}"
        assert isinstance(exec_phase.result, dict)
        assert exec_phase.result.get("mode") == "dynamic_workflow"
        assert exec_phase.result.get("success") is True, f"execute 分派失败: {exec_phase.result}"
        assert exec_phase.result.get("n_completed") == 1
        # 完整 result 契约
        assert result.success is True
        assert result.report_path is not None

    def test_full_loop_execute_whitelist_blocks_in_loop(
        self, engine: AutoloopEngine, monkeypatch: pytest.MonkeyPatch
    ):
        """整环内 _execute 白名单强制: 白名单外工具在 execute 被拦截."""
        _set_execute_whitelist("experiment_protocol_tool")
        # 脚本含一个白名单外的 file_read_tool → 该 subtask 被拦截
        blocked_script = _dynw_script(
            [
                ("allowed", "experiment_protocol_tool", {"action": "status"}),
                ("blocked", "file_read_tool", {"file_path": str(engine.workspace)}),
            ]
        )
        from unittest.mock import AsyncMock

        monkeypatch.setenv("HUGINN_COGNITIVE_LLM_DECIDER", "0")
        engine._perceive = lambda: {"changed_files": [], "timestamp": "t"}  # type: ignore[assignment]
        engine._hypothesize = AsyncMock(return_value="h")  # type: ignore[assignment]
        engine._plan = AsyncMock(  # type: ignore[assignment]
            return_value={"mode": "dynamic_workflow", "description": "x", "script": blocked_script}
        )
        # `_execute` 保持真实
        engine._validate = AsyncMock(return_value={"tests_passed": True})  # type: ignore[assignment]
        engine._learn = AsyncMock(return_value=None)  # type: ignore[assignment]
        engine._report = AsyncMock(return_value=str(engine.workspace / "report.md"))  # type: ignore[assignment]
        engine._maybe_clarify = AsyncMock(return_value=None)  # type: ignore[assignment]
        engine._blind_spot_pass = AsyncMock(return_value=[])  # type: ignore[assignment]
        engine._wait_if_checkpoint_pending = AsyncMock(return_value=None)  # type: ignore[assignment]
        engine._drain_side_questions = AsyncMock(return_value=0)  # type: ignore[assignment]

        from huginn.autoloop.types import AutoloopResult

        result = asyncio.run(engine.run_cognitive(objective="e2e whitelist", max_iterations=6))
        assert isinstance(result, AutoloopResult)
        exec_phase = next(p for p in result.phases if p.name == "execute")
        # 白名单拦截不炸整环 — execute 标记完成, 但含 1 个被拦截的 subtask
        assert exec_phase.status == "completed"
        assert exec_phase.result["mode"] == "dynamic_workflow"
        assert exec_phase.result["n_completed"] == 1
        assert exec_phase.result["n_failed"] == 1


class _FakeSubagent:
    """最小 subagent 假实现 — 记录白名单应用 / 注册 / 任务, chat 产出最终 state."""

    def __init__(self, output: str = "subagent done") -> None:
        self.tool_filter: set[str] | None = None
        self.langchain_tools: list[Any] = []
        self._max_tool_calls: int | None = None
        self.registered_tool_names: set[str] | None = None
        self.task_seen: str | None = None
        self._output = output
        self.select_model_calls: list[str] = []

    def register_tools_from_registry(self) -> None:
        # 模拟只用 tool_filter 白名单里的工具注册
        self.registered_tool_names = set(self.tool_filter) if self.tool_filter else set()
        self.langchain_tools = list(self.registered_tool_names)

    def select_model(self, task: str) -> Any:
        self.select_model_calls.append(task)
        return _FakeSummarizeModel()

    async def chat(self, task: str, thread_id: str):
        self.task_seen = task
        yield {"messages": [_Msg(self._output)]}


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeSummarizeModel:
    """返回简短摘要的假模型."""

    def invoke(self, messages: list[Any]) -> Any:
        return _Msg("summarized by routed model")


class _FakeFactory:
    """最小 AgentFactory 假实现, 供 SubagentDispatch.dispatch 使用."""

    def __init__(self, agent: _FakeSubagent) -> None:
        self._agent = agent

    def get_profile(self, name: str) -> Any:
        return {"id": name}

    def list_profiles(self) -> list[Any]:
        return []

    def create(
        self,
        profile_id: str,
        thread_id: str,
        system_prompt_override: str | None = None,
        approval_callback: Any = None,
    ) -> _FakeSubagent:
        return self._agent


class TestSubagentDispatch:
    """subagent 派发 E2E — 覆盖 da791f7 改动的 subagent 路径.

    验证:
    1. 核心 dispatch: spec.allowed_tools 白名单应用为 agent.tool_filter,
       并按白名单重新注册工具 (H5-b / da791f7 相关).
    2. H5-a摘要模型路由: 长输出 (>2000) 时 select_model("summarize") 被调用,
       摘要走路由模型而非 factory 回退.
    3. 守卫: 未知 spec / 递归深度超限 → 干净失败.
    """

    def test_dispatch_applies_tool_whitelist(self):
        """explore spec 的 allowed_tools 白名单应用到子 agent 工具集."""
        from huginn.agents.subagent import SubagentDispatch

        agent = _FakeSubagent()
        factory = _FakeFactory(agent)
        dispatch = SubagentDispatch()
        result = asyncio.run(
            dispatch.dispatch("explore", "explore repo", context={"agent_factory": factory})
        )
        assert result.success is True, f"dispatch 失败: {result.error}"
        assert result.spec_name == "explore"
        assert "subagent done" in result.summary
        # 白名单已应用为 tool_filter
        expected = set(dispatch._specs["explore"].allowed_tools)
        assert agent.tool_filter == expected, f"白名单未应用: {agent.tool_filter}"
        # 按白名单重新注册 (只含白名单工具)
        assert agent.registered_tool_names == expected
        assert agent.task_seen == "explore repo"

    def test_dispatch_h5a_summarize_uses_routed_model(self):
        """长输出 (>2000) 触发摘要, 走 select_model('summarize') 路由模型."""
        from huginn.agents.subagent import SubagentDispatch

        long_output = "A" * 2500  # > _SUMMARIZE_THRESHOLD (2000)
        agent = _FakeSubagent(output=long_output)
        factory = _FakeFactory(agent)
        dispatch = SubagentDispatch()
        spec = dispatch._specs["explore"]
        # explore spec 默认 summarize_result=True; 长输出触发 _summarize
        result = asyncio.run(
            dispatch.dispatch("explore", "t", context={"agent_factory": factory})
        )
        assert result.success is True
        # H5-a: select_model("summarize") 被调用, 摘要来自路由模型
        assert "summarize" in agent.select_model_calls, (
            f"select_model('summarize') 未调用: {agent.select_model_calls}"
        )
        assert result.summary == "summarized by routed model"

    def test_dispatch_unknown_spec_fails(self):
        """未知 spec 干净失败."""
        from huginn.agents.subagent import SubagentDispatch

        agent = _FakeSubagent()
        factory = _FakeFactory(agent)
        dispatch = SubagentDispatch()
        result = asyncio.run(
            dispatch.dispatch("no_such_spec", "t", context={"agent_factory": factory})
        )
        assert result.success is False
        assert "Unknown subagent spec" in (result.error or "")

    def test_dispatch_depth_guard(self):
        """递归深度超限拒绝, 防 subagent 无限递归 (G1)."""
        from huginn.agents.subagent import SubagentDispatch

        agent = _FakeSubagent()
        factory = _FakeFactory(agent)
        dispatch = SubagentDispatch()
        # explore spec max_depth=1; _depth=1 已经超 (>= max_depth) → 拒绝
        result = asyncio.run(
            dispatch.dispatch(
                "explore", "t",
                context={"agent_factory": factory},
                _depth=1,
            )
        )
        assert result.success is False
        assert "recursion" in (result.error or "").lower() or "depth" in (result.error or "").lower()