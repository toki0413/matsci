"""结构性守卫 TDD 测试: 确定性执行对象锚点 + 确定性验收锚点.

覆盖四条结构性修复:
a) plan DESCRIPTION 为空 → 重写一次 → fail-fast (不再用当前假设兜底)
b) objective 交付物清单确定性解析 + 验收 (agent 不可改写)
c) plan→execute 缺确定性证据 → 硬阻断 (原 advisory 放行)
d) 同因连续两次结构性失败 → 熔断
"""

from __future__ import annotations

import asyncio


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
