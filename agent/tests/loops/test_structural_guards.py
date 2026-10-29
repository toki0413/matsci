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
        _deliverables_fn=lambda: {"passed": False, "missing": ["loop_log.jsonl"],
                                  "empty": [], "invalid": []},
        _append_hint_fn=hints.append,
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
        _deliverables_fn=lambda: {"passed": False, "missing": ["report.md"]},
        _append_hint_fn=lambda _h: None,
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
        _deliverables_fn=lambda: {"passed": True},
        _append_hint_fn=lambda _h: None,
        _control_trace=lambda *a, **k: None,
        _last_good_plan={"description": "x"},
    )
    state = SimpleNamespace(iteration=9, should_stop=False, should_redirect=False,
                            redirect_reason="")
    decision = ActionDecision(action="skip")
    assert CognitiveLoop._guard_repetition(stub, state, decision, 6) is True
    assert state.should_stop is True


def test_repetition_guard_runs_on_real_kernel_instance():
    """真实例回归: 桩对象自带 _append_hint / _control_trace, 掩盖了内核没有这两个方法.

    ws35 实跑在此崩: AttributeError: 'CognitiveLoop' object has no attribute
    '_append_hint' —— 交付物硬门本该改判 execute, 结果整跑退出码 1.
    """
    from huginn.autoloop.cognitive_loop import ActionDecision, CognitiveLoop, LoopState

    hints: list[str] = []
    traces: list[tuple[str, str, dict]] = []

    async def _observe(_s):
        return {}

    async def _decide(_s, _o):
        return ActionDecision(action="skip")

    async def _exec(_s, _d):
        return None

    async def _reflect(_s, _d, _r):
        return None

    loop = CognitiveLoop(
        observe_fn=_observe,
        decide_fn=_decide,
        execute_fn=_exec,
        reflect_fn=_reflect,
        deliverables_fn=lambda: {"passed": False, "missing": ["results.json"],
                                 "empty": [], "invalid": []},
        append_hint_fn=hints.append,
        control_trace_fn=lambda name, ev, **kw: traces.append((name, ev, kw)),
    )
    state = LoopState(iteration=7)
    decision = ActionDecision(action="skip")
    assert loop._guard_repetition(state, decision, 6) is False
    assert state.should_stop is False
    assert decision.action == "plan"
    assert hints and "交付物硬门" in hints[0]
    assert traces and traces[0][0] == "stop_gate"


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


# ===== 存在性判据的负例探针 (缺陷类: 存在性判据只能判「在」, 无法判「真」) =====
#
# 每个负例都是历史上真实出现过的假绿形态: 空壳 results.json、错名脚本、
# 同名目录。判据必须在**这些输入下变红**, 否则「门存在」= 假绿。


def test_check_manifest_empty_shell_json_is_red(tmp_path):
    """空壳 JSON 能解析、也凑够字节数, 但没有任何可核对数值 → 必须判红."""
    from huginn.autoloop.deliverables import Deliverable, check_manifest

    (tmp_path / "results.json").write_text(
        '{"results": [], "notes": {}}\n\n\n\n\n\n\n\n\n\n'  # 人为凑到 >32 字节
    )
    report = check_manifest(tmp_path, [Deliverable("results.json")])
    assert report["passed"] is False
    assert report["empty"] == ["results.json"]


def test_check_manifest_wrong_named_script_is_red(tmp_path):
    """ws35 声明 analysis.py; agent 若写成 meta_analysis.py, 必须判红 (缺精确名)."""
    from huginn.autoloop.deliverables import Deliverable, check_manifest

    (tmp_path / "meta_analysis.py").write_text("print('hi')\n" * 8)
    report = check_manifest(tmp_path, [Deliverable("analysis.py")])
    assert report["passed"] is False
    assert report["missing"] == ["analysis.py"]


def test_check_manifest_directory_at_declared_path_is_red(tmp_path):
    """同名目录不是交付文件 (is_file() 判据), 不能被当成「在」."""
    from huginn.autoloop.deliverables import Deliverable, check_manifest

    (tmp_path / "results.json").mkdir()
    (tmp_path / "results.json" / "inner.json").write_text('{"q1": 1.0}')
    report = check_manifest(tmp_path, [Deliverable("results.json")])
    assert report["passed"] is False
    assert report["missing"] == ["results.json"]


def test_parse_manifest_keeps_script_declared_in_section():
    """显式交付物段里的 .py 是验收物: ws35/ws34 的 analysis.py 曾被正则丢掉, 只查 3 个文件."""
    from huginn.autoloop.deliverables import parse_manifest

    obj = (
        "【研究任务】做元分析\n\n【交付物】\n- analysis.py：可重复运行，seed=20261029\n"
        "- results.json：全部数值\n- loop_log.jsonl：50 行\n- report.md：六节\n"
    )
    paths = [d.path for d in parse_manifest(obj)]
    assert paths == ["analysis.py", "results.json", "loop_log.jsonl", "report.md"]


def test_parse_manifest_ignores_prose_script_without_section():
    """无交付物段时, 散文里的示例命令 `["python3","run.py"]` 不是交付物.

    否则 run.py 永久判 missing, 把成功跑判成失败 (ws31 同源误判).
    """
    from huginn.autoloop.deliverables import parse_manifest

    obj = '用 bash_tool 执行数组 ["python3","run.py"]，产出 data/out.csv'
    paths = [d.path for d in parse_manifest(obj)]
    assert paths == ["data/out.csv"]


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


# ===== 缺陷台账: 按 class 记账, 不按实例记账 =====
#
# 规则 (用户定): 同一类缺陷第 2 次复发起, **禁止再提交实例补丁**, 必须先有结构性修复
# 与可执行探针. 台账把这条规则变成门: 删掉探针 / 复发了却只标 instance 补丁 → 红.
#
# 字段:
#   class_id    缺陷类 (稳定标识, 全表唯一)
#   symptom     这类缺陷的可观察症状 (一句话)
#   repeat_count 已复发次数 (>=2 即"结构性")
#   fix_scope   "structural" (改判据/改契约) | "instance" (only 一次性的实例补丁)
#   probes      本文件里证明该类缺陷已被拦住的可执行探针 (test_* 函数名)

_DEFECT_LEDGER: tuple[dict, ...] = (
    {
        "class_id": "A-spec-undecidable",
        "symptom": "objective 里的交付物/键名只是散文, 引擎无法判定是否满足",
        "repeat_count": 3,
        "fix_scope": "structural",
        "probes": (
            "test_parse_manifest_from_explicit_section",
            "test_parse_manifest_keeps_script_declared_in_section",
            "test_parse_manifest_ignores_prose_script_without_section",
        ),
    },
    {
        "class_id": "B-acceptance-authority-misplaced",
        "symptom": "完成与否由同一模型自评; 模型说 stop 就收工",
        "repeat_count": 2,
        "fix_scope": "structural",
        "probes": (
            "test_stop_gate_rejects_stop_when_deliverables_missing",
            "test_finalize_success_requires_deliverables",
            "test_repetition_guard_forces_execute_when_deliverables_missing",
        ),
    },
    {
        "class_id": "C-gate-strength-asymmetry",
        "symptom": "门写着 advisory, 实际不阻断; progress 只数动作不判质量",
        "repeat_count": 2,
        "fix_scope": "structural",
        "probes": (
            "test_plan_execute_hard_blocks_empty_description",
            "test_plan_execute_still_advisory_for_non_hard_evidence",
            "test_same_structural_cause_twice_stops",
        ),
    },
    {
        "class_id": "D-probe-fidelity-gap",
        "symptom": "测试桩自带真实内核没有的方法, 桩绿而真跑崩",
        "repeat_count": 2,
        "fix_scope": "structural",
        "probes": ("test_repetition_guard_runs_on_real_kernel_instance",),
    },
    {
        "class_id": "E-failure-unobservable",
        "symptom": "LLM 超时后静默重试, 无日志; 运行挂了只看到退出码",
        "repeat_count": 2,
        "fix_scope": "structural",
        "probes": (
            "test_observe_failure_is_recorded_not_silently_degraded",
            "test_decide_failure_is_recorded_and_stops",
            "test_execute_failure_is_recorded_and_redirects",
            "test_reflect_failure_is_recorded",
        ),
    },
    {
        "class_id": "F-existence-gate-fake-green",
        "symptom": "存在性判据把「文件在」当「结果真」: 空壳 JSON / 错名文件 / 同名目录都判绿",
        "repeat_count": 3,
        "fix_scope": "structural",
        "probes": (
            "test_check_manifest_empty_shell_json_is_red",
            "test_check_manifest_wrong_named_script_is_red",
            "test_check_manifest_directory_at_declared_path_is_red",
            "test_check_manifest_invalid_json",
        ),
    },
)


def test_defect_ledger_class_ids_are_unique():
    ids = [e["class_id"] for e in _DEFECT_LEDGER]
    assert len(ids) == len(set(ids)), f"缺陷台账存在重复 class_id: {ids}"


def test_repeated_defect_class_requires_structural_fix_and_probe():
    """同 class 复发 >= 2 次时, 只许结构性修复, 且必须挂可执行探针.

    "instance" 补丁只能用于 repeat_count == 1 的首次出现; 复发仍打实例补丁 → 红.
    """
    for e in _DEFECT_LEDGER:
        assert e["repeat_count"] >= 1, f"{e['class_id']} repeat_count 非法"
        if e["repeat_count"] >= 2:
            assert e["fix_scope"] == "structural", (
                f"{e['class_id']} 已复发 {e['repeat_count']} 次, 仍是实例补丁 —— "
                "复发类缺陷必须先改判据/契约再提交."
            )
        assert e["probes"], f"{e['class_id']} 无探针: 修复未被任何可执行断言拦住"


def test_defect_probes_exist_and_are_collected():
    """台账里每个探针都真实存在且会被 pytest 收集 (删探针 = 门静默失效)."""
    missing = [
        p
        for e in _DEFECT_LEDGER
        for p in e["probes"]
        if not (p.startswith("test_") and callable(globals().get(p)))
    ]
    assert missing == [], f"缺陷台账引用了不存在/未收集的探针: {missing}"



