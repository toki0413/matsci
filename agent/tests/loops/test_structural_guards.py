"""结构性守卫 TDD 测试: 确定性执行对象锚点 + 确定性验收锚点.

覆盖四条结构性修复:
a) plan DESCRIPTION 为空 → 重写一次 → fail-fast (不再用当前假设兜底)
b) objective 交付物清单确定性解析 + 验收 (agent 不可改写)
c) plan→execute 缺确定性证据 → 硬阻断 (原 advisory 放行)
d) 同因连续两次结构性失败 → 熔断
另加:
e) A2 — 核心循环钩子失败可见化 (LoopState.failures)
f) 第0批门禁 — autoloop except 处理器总数 / 静默宽捕获棘轮
"""

from __future__ import annotations

import ast
import asyncio
from pathlib import Path

# ===== Task 1: deliverables 模块 =====


def test_parse_manifest_from_explicit_section():
    from huginn.autoloop.deliverables import parse_manifest

    obj = "【研究任务】做实验\n【交付物】results.json, report.md, fig1.png"
    m = parse_manifest(obj)
    assert [d.path for d in m] == ["results.json", "report.md", "fig1.png"]


def test_parse_manifest_ignores_source_paths():
    from huginn.autoloop.deliverables import parse_manifest

    obj = "修改 agent/huginn/foo.py 并产出 out/report.md"
    m = parse_manifest(obj)
    assert [d.path for d in m] == ["out/report.md"]


def test_parse_manifest_ignores_absolute_paths():
    """绝对路径是引用材料, 不是交付物.

    ws31 实测: "/workspace/.../design_solid_h2.md" 的匹配跳过前导 "/" 从
    "workspace" 开始, 被当成 workspace 下的相对路径 → 必然判 missing,
    把一次真实跑通的 run 判成 success=False.
    """
    from huginn.autoloop.deliverables import parse_manifest

    obj = (
        "背景材料：/workspace/research/design_solid_h2.md\n\n"
        "交付物（全部写到当前 workspace 目录下）：\n"
        "- results.json：数值汇总\n- report.md：报告\n- loop_log.jsonl：台账\n"
    )
    m = parse_manifest(obj)
    assert [d.path for d in m] == ["results.json", "report.md", "loop_log.jsonl"]


def test_stop_gate_rejects_stop_when_deliverables_missing():
    """交付物未验收时, 模型单方面 stop 不被接受 (确定性门)."""
    from types import SimpleNamespace

    from huginn.autoloop.cognitive_loop import ActionDecision, CognitiveRunner

    hints: list[str] = []
    stub = SimpleNamespace(
        _deliverables_ok=lambda: {"passed": False, "missing": ["report.md"],
                                  "empty": [], "invalid": []},
        _append_hint=hints.append,
        _control_trace=lambda *a, **k: None,
    )
    state = SimpleNamespace(iteration=3)
    cog = {"plan": {"description": "x"}}
    out = CognitiveRunner._rc_gate_stop(stub, ActionDecision(action="stop"), state, cog)
    assert out.action == "execute"
    assert hints and "交付物硬门" in hints[0]


def test_stop_gate_allows_stop_when_deliverables_pass():
    from types import SimpleNamespace

    from huginn.autoloop.cognitive_loop import ActionDecision, CognitiveRunner

    stub = SimpleNamespace(
        _deliverables_ok=lambda: {"passed": True},
        _append_hint=lambda _h: None,
        _control_trace=lambda *a, **k: None,
    )
    out = CognitiveRunner._rc_gate_stop(
        stub, ActionDecision(action="stop"), SimpleNamespace(iteration=3), {"plan": None}
    )
    assert out.action == "stop"


def test_repetition_guard_forces_execute_when_deliverables_missing():
    """连续 skip 到 2 倍上限时, 交付物未验收 → 改判 execute, 不是强制停机.

    ws33 实测: 模型连发 6 次 skip, 引擎 forcing stop, iteration 11 收口,
    loop_log.jsonl / results.json / report.md 全缺 → success=False.
    """
    from types import SimpleNamespace

    from huginn.autoloop.cognitive_loop import ActionDecision, CognitiveLoop

    hints: list[str] = []
    stub = SimpleNamespace(
        _max_repeated_actions=3,
        _deliverables_ok=lambda: {"passed": False, "missing": ["loop_log.jsonl"],
                                  "empty": [], "invalid": []},
        _append_hint=hints.append,
        _control_trace=lambda *a, **k: None,
        _last_good_plan={"description": "x"},
    )
    state = SimpleNamespace(iteration=11, should_stop=False, should_redirect=False,
                            redirect_reason="")
    decision = ActionDecision(action="skip")
    quit_loop = CognitiveLoop._guard_repetition(stub, state, decision, 6)
    assert quit_loop is False
    assert state.should_stop is False
    assert decision.action == "execute"
    assert hints and "交付物硬门" in hints[0]


def test_repetition_guard_forces_plan_when_no_plan_yet():
    from types import SimpleNamespace

    from huginn.autoloop.cognitive_loop import ActionDecision, CognitiveLoop

    stub = SimpleNamespace(
        _max_repeated_actions=3,
        _deliverables_ok=lambda: {"passed": False, "missing": ["report.md"]},
        _append_hint=lambda _h: None,
        _control_trace=lambda *a, **k: None,
        _last_good_plan=None,
    )
    state = SimpleNamespace(iteration=5, should_stop=False, should_redirect=False,
                            redirect_reason="")
    decision = ActionDecision(action="skip")
    assert CognitiveLoop._guard_repetition(stub, state, decision, 6) is False
    assert decision.action == "plan"


def test_repetition_guard_still_stops_when_deliverables_pass():
    from types import SimpleNamespace

    from huginn.autoloop.cognitive_loop import ActionDecision, CognitiveLoop

    stub = SimpleNamespace(
        _max_repeated_actions=3,
        _deliverables_ok=lambda: {"passed": True},
        _append_hint=lambda _h: None,
        _control_trace=lambda *a, **k: None,
        _last_good_plan={"description": "x"},
    )
    state = SimpleNamespace(iteration=9, should_stop=False, should_redirect=False,
                            redirect_reason="")
    decision = ActionDecision(action="skip")
    assert CognitiveLoop._guard_repetition(stub, state, decision, 6) is True
    assert state.should_stop is True


def test_check_manifest_reports_missing_and_empty(tmp_path):
    from huginn.autoloop.deliverables import Deliverable, check_manifest

    (tmp_path / "ok.md").write_text("x" * 64)
    (tmp_path / "empty.json").write_text("")
    report = check_manifest(
        tmp_path,
        [Deliverable("ok.md"), Deliverable("empty.json"), Deliverable("missing.csv")],
    )
    assert report["passed"] is False
    assert report["missing"] == ["missing.csv"]
    assert report["empty"] == ["empty.json"]


def test_check_manifest_invalid_json(tmp_path):
    from huginn.autoloop.deliverables import Deliverable, check_manifest

    (tmp_path / "r.json").write_text("{not json")
    report = check_manifest(tmp_path, [Deliverable("r.json")])
    assert report["invalid"] == ["r.json"]
    assert report["passed"] is False


# ===== Task 2: plan→execute 确定性硬门 =====


def test_plan_execute_hard_blocks_empty_description():
    from huginn.autoloop.phase_gate import PhaseGateHook

    hook = PhaseGateHook()
    gate = hook.evaluate("plan", "execute", {"mode": "coder", "description": ""})
    assert gate.is_blocked


def test_plan_execute_still_advisory_for_non_hard_evidence():
    from huginn.autoloop.phase_gate import PhaseGateHook

    hook = PhaseGateHook()
    gate = hook.evaluate("hypothesize", "plan", {})
    assert not gate.is_blocked


# ===== Task 3: 去假设兜底 + plan 空描述 fail-fast =====


def test_execute_coder_empty_description_fails_without_calling_coder():
    from huginn.autoloop.engine_act import EngineAct

    class E:
        _needs_external_data_io = staticmethod(lambda t: False)
        _record_internal_tool_call = staticmethod(lambda *a, **k: None)

    class C:
        def run(self, task):
            raise AssertionError("coder 不应被空任务调用")

    e = E()
    e.coder = C()
    out = asyncio.run(EngineAct(e)._execute_coder("", {}))
    assert out["success"] is False
    assert "empty_description" in out["error"]


def test_plan_empty_description_retries_then_fails(monkeypatch):
    from huginn.autoloop.engine_act import EngineAct

    class E:
        _objective = "x"
        _current_hyp_id_for_plan = None
        _last_plan_failure = None
        _grill_active = False
        _grill_turns = 0

        def _build_plan_prompt(self, h, c):
            return "p"

        def _parse_plan(self, r):
            return {"mode": "coder", "description": ""}

        def _override_plan_mode(self, p):
            return p

        async def _plan_check_and_refine(self, p, h, c):
            return p

        def _plan_jepa_slots(self, p):
            return None

        def _get_plan_store(self):
            return None

        def _resolve_readonly_tools(self):
            return []

    async def _fake_chat(self, *a, **k):
        return "MODE: coder\nDESCRIPTION:"

    # EngineAct 自有 _llm_chat (不走引擎转发), 必须在类上替换.
    monkeypatch.setattr(EngineAct, "_llm_chat", _fake_chat)
    e = E()
    act = EngineAct(e)
    out = asyncio.run(act._plan("h", {}))
    assert out is None
    assert e._last_plan_failure["kind"] == "empty_description"
    assert e._last_plan_failure["attempts"] == 2


# ===== Task 4: plan 拒绝停机 + 交付物门决定 success =====


def test_finalize_success_requires_deliverables(tmp_path):
    from huginn.autoloop.cognitive_loop import CognitiveRunner
    from huginn.autoloop.deliverables import Deliverable

    class E:
        workspace = tmp_path
        _deliverable_manifest = [Deliverable("report.md")]

    runner = CognitiveRunner(E())
    assert runner._deliverables_ok()["passed"] is False
    (tmp_path / "report.md").write_text("x" * 100)
    assert runner._deliverables_ok()["passed"] is True


def test_exec_plan_rejection_stops_run():
    from huginn.autoloop.cognitive_loop import CognitiveRunner

    class E:
        _last_plan_failure = {"kind": "empty_description", "diagnostic": "x"}

        @staticmethod
        async def _plan(*a, **k):
            return None

    runner = CognitiveRunner(E())

    class State:
        should_stop = False
        iteration = 1

    class Phase:
        result = None
        status = "failed"

    async def fake_run_phase(*a, **k):
        return Phase()

    runner._run_phase_async = fake_run_phase
    runner._emit_campaign = lambda *a, **k: None
    cog = {
        "hypothesis": "h",
        "phases": [],
        "completed_steps": 0,
        "plan": None,
        "context": {},
    }
    state = State()
    asyncio.run(runner._exec_plan(state, cog, {}))
    assert state.should_stop is True


# ===== Task 5: 同因熔断 =====


def test_structural_failure_type_preserved():
    from huginn.autoloop.hypothesis_loop import HypothesisLoop

    out = HypothesisLoop._classify_failure(
        {"failure_kind": "empty_description", "errors": "", "result": ""}
    )
    assert out == "empty_description"


def test_same_structural_cause_twice_stops():
    from huginn.autoloop.cognitive_loop import CognitiveRunner, LoopState

    class E:
        _consecutive_failures = 0
        _consecutive_failures_by_type = {}
        _max_consecutive_failures = 20
        _max_failures_by_type = {}
        _validate_window = None
        _last_failure_diagnostic = {"kind": "empty_description", "diagnostic": "x"}

        @staticmethod
        def _control_trace(*a, **k):
            return None

        @staticmethod
        def _redteam_findings():
            return []

    runner = CognitiveRunner(E())
    val = {"tests_passed": False, "failure_kind": "empty_description"}
    assert runner._reflect_failure_budget(val, LoopState(max_iterations=5)) is None
    res = runner._reflect_failure_budget(val, LoopState(max_iterations=5))
    assert res is not None and res.should_stop is True
    assert "empty_description" in res.advice


# ===== Task 6: 否定句路由 =====


def test_negated_write_code_not_routed():
    from huginn.autoloop.plan_check import PlanCheck

    class E:
        _objective = "不要写代码，只做文献综述"

    pc = PlanCheck(E())
    assert pc._asks_to_write_and_run_code({"description": "综述"}) is False


def test_affirmative_write_code_still_routed():
    from huginn.autoloop.plan_check import PlanCheck

    class E:
        _objective = "写并运行脚本计算均方误差"

    pc = PlanCheck(E())
    assert pc._asks_to_write_and_run_code({"description": "跑实验"}) is True


# ===== Task 7 (A2): 核心循环钩子失败可见化 =====


def _run_one_iteration(**overrides):
    """跑一轮 CognitiveLoop, 允许按 stage 注入会抛异常的钩子."""
    from huginn.autoloop.cognitive_loop import (
        ActionDecision,
        CognitiveLoop,
        LoopState,
        ReflectionResult,
    )

    async def _observe(state):
        return {"ok": True}

    async def _decide(state, obs):
        return ActionDecision(action="stop")

    async def _execute(state, decision):
        return "result"

    async def _reflect(state, decision, result):
        return ReflectionResult()

    fns = {
        "observe_fn": _observe,
        "decide_fn": _decide,
        "execute_fn": _execute,
        "reflect_fn": _reflect,
    }
    fns.update(overrides)
    loop = CognitiveLoop(max_iterations=1, **fns)
    return asyncio.run(loop.run(LoopState(max_iterations=1)))


def test_observe_failure_is_recorded_not_silently_degraded():
    async def boom(state):
        raise RuntimeError("observe boom")

    state = _run_one_iteration(observe_fn=boom)
    assert state.failed is True
    assert state.failures[0].stage == "observe"
    assert state.failures[0].error == "RuntimeError: observe boom"


def test_decide_failure_is_recorded_and_stops():
    async def boom(state, obs):
        raise RuntimeError("decide boom")

    state = _run_one_iteration(decide_fn=boom)
    assert state.failed is True
    assert state.failures[0].stage == "decide"
    assert state.should_stop is True


def test_execute_failure_is_recorded_and_redirects():
    async def boom(state, decision):
        raise RuntimeError("execute boom")

    state = _run_one_iteration(execute_fn=boom)
    assert state.failed is True
    assert state.failures[0].stage == "execute"
    assert "action=stop" in state.failures[0].error
    assert state.should_redirect is False  # 非超时只在 reflect 里评估


def test_reflect_failure_is_recorded():
    async def boom(state, decision, result):
        raise RuntimeError("reflect boom")

    state = _run_one_iteration(reflect_fn=boom)
    assert state.failed is True
    assert state.failures[0].stage == "reflect"


def test_degradable_writer_failure_is_not_a_loop_failure():
    class _BoomWriter:
        def write_step(self, **kwargs):
            raise RuntimeError("write boom")

    state = _run_one_iteration(output_writer=_BoomWriter())
    assert state.failed is False
    assert state.failures == []


def test_step_failure_to_dict_is_serializable():
    from huginn.autoloop.cognitive_loop import StepFailure

    assert StepFailure("observe", "RuntimeError: x", 3).to_dict() == {
        "stage": "observe",
        "error": "RuntimeError: x",
        "iteration": 3,
    }


# ===== Task 8 (第0批门禁): autoloop except 卫生棘轮 =====
#
# 口径: AST 统计 huginn/autoloop 下所有 ExceptHandler.
#   broad  = type 为 Exception / BaseException 或裸 except;
#   silent = handler 内既没有 raise, 也没有 warning/error/critical/exception 级日志,
#            即失败在常规日志里不可见.
# 基线只减不增: 新增静默宽捕获 → 红. 收敛后请下调基线.

_AUTOLOOP_ROOT = Path(__file__).resolve().parents[2] / "huginn" / "autoloop"
_AUTOLOOP_EXCEPT_TOTAL_BASELINE = 575
_AUTOLOOP_BROAD_SILENT_BASELINE = 473
_LOUD_LOG_METHODS = frozenset({"warning", "error", "critical", "exception", "fatal"})


def _except_stats() -> tuple[int, int]:
    total = broad_silent = 0
    for py in sorted(_AUTOLOOP_ROOT.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            total += 1
            if not _is_broad_handler(node):
                continue
            has_raise = any(isinstance(n, ast.Raise) for n in ast.walk(node))
            has_loud_log = any(
                isinstance(n, ast.Call)
                and isinstance(n.func, ast.Attribute)
                and n.func.attr in _LOUD_LOG_METHODS
                for n in ast.walk(node)
            )
            if not has_raise and not has_loud_log:
                broad_silent += 1
    return total, broad_silent


def _is_broad_handler(node: ast.ExceptHandler) -> bool:
    t = node.type
    if t is None:
        return True
    if isinstance(t, ast.Name):
        return t.id in ("Exception", "BaseException")
    if isinstance(t, ast.Tuple):
        return any(
            isinstance(e, ast.Name) and e.id in ("Exception", "BaseException")
            for e in t.elts
        )
    return False


def test_autoloop_except_handler_total_does_not_grow():
    total, _ = _except_stats()
    assert total <= _AUTOLOOP_EXCEPT_TOTAL_BASELINE, (
        f"autoloop except 处理器 {total} 个, 超过基线 {_AUTOLOOP_EXCEPT_TOTAL_BASELINE}. "
        "新增 except 前先问: 这个失败真能被吞吗? 能就显式降级并记录, 不能就别捕获."
    )


def test_autoloop_silent_broad_except_does_not_grow():
    _, broad_silent = _except_stats()
    assert broad_silent <= _AUTOLOOP_BROAD_SILENT_BASELINE, (
        f"autoloop 静默宽捕获 {broad_silent} 个, 超过基线 "
        f"{_AUTOLOOP_BROAD_SILENT_BASELINE}. 宽捕获里必须至少记 warning/error 级日志, "
        "或把失败上报到调用方 —— 只记 debug 等于把故障埋掉."
    )


