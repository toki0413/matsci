"""run_cognitive 自检 — 用 mock engine 验证 4 钩子接线, 不依赖 LLM/workspace.

跑法: python -m huginn.autoloop.verify_run_cognitive

4 个场景:
  1. Happy path: hyp→plan→exec→validate→learn→stop, finalize 拿到 phases
  2. Hypothesize 反复失败: 没 hyp 可 pivot → 直接停 (不死循环)
  2b. Pivot 死循环防护: hyp 成功 + plan 反复失败 → 3 pivots 后硬停
  3. Validate 反复失败: consecutive_failures 上限触发 stop
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from huginn.autoloop.engine import AutoloopEngine
from huginn.autoloop.types import LoopPhase


def _make_mock_engine() -> AutoloopEngine:
    """bypass __init__, 手动设 run_cognitive() 涉及的属性."""
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng.workspace = None
    eng._objective = ""
    eng._iteration = 0
    eng._should_stop = False
    eng._consecutive_failures = 0
    eng._max_consecutive_failures = 3
    eng._refine_count = 0
    eng._max_refines = 8
    eng._pivot_count = 0
    eng._max_pivots = 10
    eng._next_phase_hint = None
    eng._refined_hypothesis = None
    eng._speculator_hint = ""
    eng._current_prediction = ""
    # Step C: selfcheck 走规则版, 关掉 LLM decider (mock engine 不调 LLM)
    eng._use_llm_decider = False
    eng._current_hyp_id_for_plan = None
    eng._run_id = "test"
    eng._parent_run_id = None
    eng._progress_task_id = "test"

    class _HG:
        def add_hypothesis(self, statement, rationale=""):
            return f"hyp_{int(time.time()*1000) % 10000}"
        def pivot(self, cur, evidence, model, objective):
            return "new_hyp"
    eng.hypothesis_graph = _HG()
    return eng


def _install_stubs(eng: AutoloopEngine, scenario: str) -> list[str]:
    """装 stub phase 方法 + tracker. 返回 execute_fn 跑过的 phase 名列表."""
    actions: list[str] = []

    class _GateState:
        def reset_runtime(self): pass
    from huginn.autoloop import engine as eng_mod
    eng_mod.get_shared_phase_gate_state = lambda: _GateState()

    # _perceive 是 sync (real method 也是 sync, asyncio.to_thread 调)
    def _perceive():
        return {"summary": "mock context"} if scenario != "empty_perceive" else None

    async def _hypothesize(ctx):
        if scenario == "hyp_always_fail":
            return None
        return "mock hypothesis"
    async def _plan(hyp, ctx):
        if scenario == "plan_fail":
            return None
        return {"mode": "test", "description": "mock plan", "expected_prediction": ""}
    async def _execute(plan, ctx):
        if scenario == "exec_fail":
            return None
        return {"result": "ok"}
    async def _validate(exec_result):
        if scenario == "validate_always_fail":
            return {"tests_passed": False}
        return {"tests_passed": True}
    async def _learn(hyp, plan, val):
        return None
    async def _report():
        return "mock report"

    eng._perceive = _perceive
    eng._hypothesize = _hypothesize
    eng._plan = _plan
    eng._execute = _execute
    eng._validate = _validate
    eng._learn = _learn
    eng._report = _report

    def _prepare_run(obj, pb, goal):
        eng._objective = obj
        return ("test_run", None, None)
    eng._prepare_run = _prepare_run

    async def _finalize_run(obj, phases, run_id, provenance, collector,
                            tracker, progress_task_id, completed_steps):
        return {"status": "ok",
                "phases_count": len(phases),
                "completed_steps": completed_steps}
    eng._finalize_run = _finalize_run

    def _check_gate(frm, to, evidence):
        return True
    eng._check_gate = _check_gate

    async def _run_phase_async(name, fn, *args):
        phase = LoopPhase(name=name)
        phase.status = "running"
        phase.start_time = time.time()
        try:
            r = fn(*args)
            if asyncio.iscoroutine(r):
                r = await r
            phase.result = r
            phase.status = "completed"
        except Exception as e:
            phase.status = "failed"
            phase.error = str(e)
        phase.end_time = time.time()
        actions.append(name)
        return phase
    eng._run_phase_async = _run_phase_async

    def _git_commit(plan, it):
        pass
    eng._git_commit_after_execute = _git_commit
    eng._get_refine_model = lambda: None

    class _PT:
        def start_task(self, **kw): pass
        def is_expired(self, _tid): return False
        def update(self, *a, **kw): pass
        def complete(self, *a, **kw): pass
        def fail(self, *a, **kw): pass
    eng_mod.get_progress_tracker = lambda: _PT()

    return actions


async def _selfcheck():
    # 1. Happy path — 6 phase (observe 在 observe_fn 不进 actions) 都跑通
    eng1 = _make_mock_engine()
    actions1 = _install_stubs(eng1, "happy")
    result1 = await eng1.run_cognitive(
        objective="test", max_iterations=10, progressive_budget=False,
    )
    assert result1["status"] == "ok", f"happy: status wrong: {result1}"
    for needed in ("hypothesize", "plan", "execute", "validate", "learn"):
        assert needed in actions1, f"happy: missing {needed} in {actions1}"
    assert eng1._consecutive_failures == 0, f"happy: should have 0 failures, got {eng1._consecutive_failures}"
    assert eng1._pivot_count == 0, f"happy: should have 0 pivots, got {eng1._pivot_count}"
    print(f"1. Happy path OK — phases: {actions1}")

    # 2. Hypothesize 反复失败 → 没 hyp 可 pivot, 直接停 (不死循环)
    eng2 = _make_mock_engine()
    actions2 = _install_stubs(eng2, "hyp_always_fail")
    await eng2.run_cognitive(objective="test", max_iterations=20, progressive_budget=False)
    # decide 看到 redirect + 没 hyp → 直接 stop, 不浪费 pivot
    assert eng2._pivot_count == 0, (
        f"hyp_fail: should not pivot (no hyp to pivot from), got {eng2._pivot_count}"
    )
    # 应该 2 轮内停: iter1 hyp(fail)→redirect, iter2 decide→stop
    assert len(actions2) <= 3, (
        f"hyp_fail: should stop in <=3 actions, got {actions2}"
    )
    print(f"2. Hyp-fail early-stop OK — actions: {actions2}, pivots: {eng2._pivot_count}")

    # 2b. Pivot 死循环防护 — hyp 成功, plan 反复失败 → pivot 3 次后硬停
    eng2b = _make_mock_engine()
    actions2b = _install_stubs(eng2b, "plan_fail")
    # plan_fail: hyp 成功, plan 返回 None → reflect redirect → decide pivot
    # 第 1 次 hyp 成功, 后续 fail → pivot → fail → pivot → fail → pivot → STOP
    await eng2b.run_cognitive(objective="test", max_iterations=30, progressive_budget=False)
    # 死循环防护: pivot_count >= 3 时硬停 (reflect 的 pivot guard)
    assert eng2b._pivot_count >= 3, (
        f"plan_fail: should pivot >=3 times, got {eng2b._pivot_count}"
    )
    assert eng2b._pivot_count < eng2b._max_pivots, (
        f"plan_fail: pivot_count {eng2b._pivot_count} reached max, "
        f"death-loop guard didn't fire early"
    )
    print(f"2b. Pivot death-loop guard OK — pivots: {eng2b._pivot_count}, actions: {actions2b}")

    # 3. Validate 反复失败 → consecutive_failures 上限 (3) 触发 stop
    eng3 = _make_mock_engine()
    actions3 = _install_stubs(eng3, "validate_always_fail")
    eng3._max_consecutive_failures = 3
    await eng3.run_cognitive(objective="test", max_iterations=20, progressive_budget=False)
    assert eng3._consecutive_failures >= 3, (
        f"validate_fail: should have >=3 failures, got {eng3._consecutive_failures}"
    )
    val_count = actions3.count("validate")
    assert val_count >= 3, f"validate_fail: expected >=3 validates, got {val_count}"
    # 不应该跑满 20 iter
    assert len(actions3) < 20 * 2, f"validate_fail: ran too long: {actions3}"
    print(f"3. Validate-fail consecutive limit OK — validates: {val_count}, failures: {eng3._consecutive_failures}")

    # 4. LLM decider 采纳 — mock _llm_chat 返回 "stop", 验证 LLM 决策被采纳
    eng4 = _make_mock_engine()
    eng4._use_llm_decider = True  # 开启 LLM decider
    actions4 = _install_stubs(eng4, "happy")
    # mock _llm_chat: 第 1 次调 (decider) 返回 stop — 首轮不调 LLM, 第 2 轮才调
    _llm_calls = []
    async def _mock_llm_chat(prompt, **kw):
        _llm_calls.append(prompt)
        # 返回 stop, 让 LLM 决策直接停 loop
        return '{"action": "stop", "rationale": "llm decided stop", "expected_outcome": "test"}'
    eng4._llm_chat = _mock_llm_chat
    await eng4.run_cognitive(objective="test", max_iterations=10, progressive_budget=False)
    # 首轮规则版 hypothesize (last_action="" 不调 LLM), 第 2 轮 LLM 返回 stop
    assert "hypothesize" in actions4, f"llm_decider: missing hypothesize in {actions4}"
    assert len(_llm_calls) >= 1, "llm_decider: should have called LLM at least once"
    assert actions4[-1] == "stop" or len(actions4) <= 3, (
        f"llm_decider: LLM stop should end loop quickly, got {actions4}"
    )
    print(f"4. LLM decider adopted OK — actions: {actions4}, llm_calls: {len(_llm_calls)}")

    # 4b. LLM decider 失败 fallback — mock _llm_chat 抛异常, 验证 fallback 到规则版
    eng4b = _make_mock_engine()
    eng4b._use_llm_decider = True
    actions4b = _install_stubs(eng4b, "happy")
    async def _mock_llm_chat_fail(prompt, **kw):
        raise RuntimeError("LLM unavailable")
    eng4b._llm_chat = _mock_llm_chat_fail
    await eng4b.run_cognitive(objective="test", max_iterations=6, progressive_budget=False)
    # fallback 到规则版, 应该走完 happy path (hyp→plan→exec→validate→learn)
    for needed in ("hypothesize", "plan", "execute", "validate", "learn"):
        assert needed in actions4b, f"llm_fallback: missing {needed} in {actions4b}"
    print(f"4b. LLM decider fallback OK — actions: {actions4b}")

    print("run_cognitive selfcheck OK (6/6)")


if __name__ == "__main__":
    asyncio.run(_selfcheck())
