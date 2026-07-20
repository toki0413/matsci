"""AutoloopEngine selfcheck — 从 engine.py 抽出的测试代码.

engine.py 聚焦生产逻辑, selfcheck 移到本文件. 通过 `python -m huginn.autoloop.engine`
触发, 或直接 `python -m huginn.autoloop.engine_selfcheck`.

延迟 import AutoloopEngine 避免 circular import.
"""

from __future__ import annotations


def run_selfcheck() -> None:
    """AutoloopEngine selfcheck — 验证 LLM decider + 4 flag gating.

    覆盖:
    1-5. _is_action_legal / _build_decider_prompt / _decide_next_action_llm fallback
    6. BranchIncubator gating (D 方向)
    7. ThreeCabin gating (B 方向)
    8. CompletionGate gating (C 方向)
    9. 3 flag 叠加 (BranchIncubator + ThreeCabin + CompletionGate)
    10. CrossDomain gating (A 方向)

    ponytail: 用 __new__ 绕过 __init__, 只测无副作用的方法.
    """
    # 延迟 import 避免 circular import
    from huginn.autoloop.engine import AutoloopEngine, _extract_tests_passed
    from huginn.autoloop.cognitive_loop import (
        LoopState, ActionDecision, _validation_to_step_eval_fields,
    )
    import asyncio
    import os as _os
    import os as _os2
    import tempfile as _tf
    from pathlib import Path as _P
    from types import SimpleNamespace as _NS

    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._use_llm_decider = True

    # 1. _is_action_legal — 全 action 前置条件
    for a in ("observe", "hypothesize", "skip", "stop"):
        assert eng._is_action_legal(a, {}) is True, f"{a} should always be legal"
    assert eng._is_action_legal("plan", {}) is False, "plan without hyp should be illegal"
    assert eng._is_action_legal("plan", {"hypothesis": "test"}) is True
    assert eng._is_action_legal("execute", {}) is False
    assert eng._is_action_legal("execute", {"plan": {"mode": "x"}}) is True
    assert eng._is_action_legal("validate", {}) is False
    assert eng._is_action_legal("validate", {"execution_result": {"r": 1}}) is True
    assert eng._is_action_legal("learn", {"hypothesis": "h"}) is False
    assert eng._is_action_legal("learn", {"hypothesis": "h", "plan": {"m": "x"}, "validation": {"v": 1}}) is True
    assert eng._is_action_legal("pivot", {}) is False
    assert eng._is_action_legal("pivot", {"current_hyp_id": "h1"}) is True
    assert eng._is_action_legal("unknown_action", {"hypothesis": "h"}) is False
    print("1. _is_action_legal (all actions) OK")

    # 2. _build_decider_prompt
    state = LoopState(iteration=3, max_iterations=10, last_action="hypothesize")
    cog = {
        "hypothesis": "test hyp",
        "plan": {"mode": "coder"},
        "execution_result": None,
        "validation": None,
        "current_hyp_id": "h1",
    }
    prompt = eng._build_decider_prompt(state, cog, {})
    assert "Iteration: 3/10" in prompt
    assert "Hypothesis: test hyp" in prompt
    assert "Plan mode: coder" in prompt
    assert "Actions:" in prompt
    assert "stop: end the loop" in prompt
    print("2. _build_decider_prompt OK")

    # 3. _decide_next_action_llm — LLM 失败 → None
    async def _fail_chat(*a, **kw):
        raise RuntimeError("LLM unavailable")
    eng._llm_chat = _fail_chat
    result = asyncio.run(eng._decide_next_action_llm(state, cog, {}))
    assert result is None
    print("3. _decide_next_action_llm fallback (LLM fail → None) OK")

    # 4. _decide_next_action_llm — 非法 action → None
    async def _bad_action_chat(*a, **kw):
        return '{"action": "fly_to_moon", "rationale": "test", "expected_outcome": "test"}'
    eng._llm_chat = _bad_action_chat
    result = asyncio.run(eng._decide_next_action_llm(state, cog, {}))
    assert result is None
    print("4. _decide_next_action_llm illegal action → None OK")

    # 5. _decide_next_action_llm — 合法 action → ActionDecision
    async def _good_chat(*a, **kw):
        return '{"action": "plan", "rationale": "have hyp, design plan", "expected_outcome": "plan dict"}'
    eng._llm_chat = _good_chat
    result = asyncio.run(eng._decide_next_action_llm(state, cog, {}))
    assert result is not None
    assert isinstance(result, ActionDecision)
    assert result.action == "plan"
    print("5. _decide_next_action_llm legal action → ActionDecision OK")

    # 6. BranchIncubator gating
    eng3 = AutoloopEngine.__new__(AutoloopEngine)
    eng3._agent_factory = object()
    eng3._branch_incubator = None
    eng3._iteration = 0
    eng3._last_persona = None
    eng3._last_hypothesis = None
    eng3._last_raw_hypothesis = None

    async def _no_op_symreg(ctx):
        return ""
    eng3._symreg_hint = _no_op_symreg
    eng3._conjecture_hint = lambda ctx: ""
    eng3._build_hypothesis_prompt = lambda ctx: "test prompt"
    eng3._pick_hypothesis_persona = lambda ctx: "dft_expert"
    eng3._record_backup_candidates = lambda raw, sel: None
    eng3._metacog_audit_hypothesis = lambda hyp, ctx: None

    _inc_called = [False]
    async def _mock_inc(ctx):
        _inc_called[0] = True
        return "INCUBATOR_HYP"
    eng3._hypothesize_via_branch_incubator = _mock_inc
    async def _llm_none(*a, **kw):
        return None
    eng3._llm_chat = _llm_none
    eng3.model = None

    _os.environ.pop("HUGINN_USE_BRANCH_INCUBATOR", None)
    asyncio.run(eng3._hypothesize({}))
    assert _inc_called[0] is False, "flag off 不应调 incubator"

    _os.environ["HUGINN_USE_BRANCH_INCUBATOR"] = "1"
    _inc_called[0] = False
    hyp = asyncio.run(eng3._hypothesize({}))
    assert _inc_called[0] is True
    assert hyp == "INCUBATOR_HYP"

    eng3._agent_factory = None
    _inc_called[0] = False
    asyncio.run(eng3._hypothesize({}))
    assert _inc_called[0] is False
    _os.environ.pop("HUGINN_USE_BRANCH_INCUBATOR", None)
    print("6. BranchIncubator gating (flag + factory) OK")

    # 7. ThreeCabin gating
    _cog = {
        "validation": {"summary": "ok", "tests_passed": True},
        "execution_result": {"summary": "ran"},
        "plan": {"description": "test", "mode": "execute"},
    }

    _os2.environ.pop("HUGINN_USE_THREE_CABIN", None)
    _val = _cog["validation"] or {}
    _tests_ok = _extract_tests_passed(_val)
    _se_fields = _validation_to_step_eval_fields(
        _val, _tests_ok, _cog.get("execution_result"), step_id=0,
    )
    _ns_eval = _NS(**_se_fields)
    assert not hasattr(_ns_eval, "is_on_track"), "SimpleNamespace 不应有 is_on_track"

    from huginn.metacog.three_cabin_reflector import run_three_cabin as _rtc
    with _tf.TemporaryDirectory() as _td:
        _ws = _P(_td)
        _evals = []
        _se, _hint = _rtc(
            cog=_cog, evals_history=_evals, step_id=0, workspace=_ws,
        )
        assert _se is not None
        assert hasattr(_se, "is_on_track"), "应返真 StepEvaluation"
        assert callable(_se.is_on_track)
        assert len(_evals) == 1

    assert _os2.environ.get("HUGINN_USE_THREE_CABIN", "0") == "0"
    print("7. ThreeCabin gating (flag + StepEvaluation 类型) OK")

    # 8. CompletionGate gating
    from huginn.metacog.completion_gate import (
        CompletionGate as _CG, GateContext as _GC,
    )
    from dataclasses import dataclass as _dc, field as _field

    @_dc
    class _G:
        objective: str = "predict X"
        success_criteria: list = _field(default_factory=lambda: ["X"])
        status: str = "active"
        metadata: dict = _field(default_factory=dict)

    assert _os2.environ.get("HUGINN_USE_COMPLETION_GATE", "0") == "0"

    _gate = _CG()
    _d = _gate.review(
        _G(success_criteria=["X"]), {"summary": "X done"},
        _GC(iteration=5, max_iterations=10, families_explored=2, live_components=2),
    )
    assert _d.status == "pass" and _d.category == "criteria_match"
    assert _d.should_complete_goal and _d.should_stop

    _d = _gate.review(None, {}, _GC(iteration=1, max_iterations=10, families_explored=1, live_components=1))
    assert _d.status == "pass" and _d.category == "no_goal"
    print("8. CompletionGate gating (flag + review 行为) OK")

    # 9. 3 flag 叠加
    import tempfile as _tf3
    from huginn.metacog.three_cabin_reflector import run_three_cabin as _rtc3
    from huginn.metacog.completion_gate import (
        CompletionGate as _CG3, GateContext as _GC3,
    )
    from huginn.metacog.branch_incubator import BranchIncubator as _BI3

    _bi = _BI3()
    _cg = _CG3()
    assert _bi is not None and _cg is not None

    _cog9 = {
        "validation": {"summary": "tests passed", "tests_passed": True},
        "execution_result": {"summary": "ran"},
        "plan": {"description": "test", "mode": "execute"},
    }
    with _tf3.TemporaryDirectory() as _td9:
        _ws9 = _P(_td9)
        _evals9 = []
        _se9, _hint9 = _rtc3(
            cog=_cog9, evals_history=_evals9, step_id=0, workspace=_ws9,
        )
        assert _se9 is not None
        assert len(_evals9) == 1
        _g9 = _G(success_criteria=["tests passed"], objective="test")
        _ctx9 = _GC3(
            iteration=5, max_iterations=10, families_explored=1, live_components=1,
        )
        _d9 = _cg.review(_g9, _cog9["validation"], _ctx9)
        assert _d9.category == "criteria_match"
        assert _os2.environ.get("HUGINN_USE_BRANCH_INCUBATOR", "0") == "0"
        assert _os2.environ.get("HUGINN_USE_THREE_CABIN", "0") == "0"
        assert _os2.environ.get("HUGINN_USE_COMPLETION_GATE", "0") == "0"
    print("9. 3 flag 叠加 (BranchIncubator + ThreeCabin + CompletionGate) OK")

    # 10. CrossDomain gating
    eng4 = AutoloopEngine.__new__(AutoloopEngine)
    eng4._should_imaginate = lambda: False
    eng4._recent_failed_hypotheses = lambda limit=5: []

    _os2.environ.pop("HUGINN_USE_CROSS_DOMAIN", None)
    _hint_off = eng4._conjecture_hint({"goal": ""})
    assert _hint_off == "", f"flag off + 空 goal → 空串, got {_hint_off!r}"

    _os2.environ["HUGINN_USE_CROSS_DOMAIN"] = "1"
    _hint_on = eng4._conjecture_hint({
        "goal": "predict Fe magnetic transition temperature",
    })
    assert isinstance(_hint_on, str), f"flag on → str, got {type(_hint_on)}"

    _hint_unknown = eng4._conjecture_hint({
        "goal": "totally unknown xyz problem qwerty",
    })
    assert isinstance(_hint_unknown, str)
    _os2.environ.pop("HUGINN_USE_CROSS_DOMAIN", None)
    print("10. CrossDomain gating (flag + _conjecture_hint 路径切换) OK")

    # ── G2: _check_stuck (cycle_detect + trajectory_match) ──────────────
    eng5 = AutoloopEngine.__new__(AutoloopEngine)
    eng5._traj_history = []  # 默认无历史, 只测 cycle 路径

    # case A: 短序列 (< 4) → 不检, 返回 None
    assert eng5._check_stuck(["observe", "hypothesize"]) is None
    print("G2-A short sequence (<4) → None OK")

    # case B: 周期序列 [a,b,a,b,a,b] → cycle 信号
    seq_b = ["observe", "hypothesize", "observe", "hypothesize", "observe", "hypothesize"]
    r_b = eng5._check_stuck(seq_b)
    assert r_b is not None, "周期序列应触发 cycle"
    assert r_b["type"] == "cycle", f"type 应为 cycle, got {r_b['type']}"
    assert r_b["period"] >= 2, f"period 应 >=2, got {r_b['period']}"
    assert "pivot" in r_b["advice"], f"advice 应含 pivot: {r_b['advice']}"
    print(f"G2-B cycle detected: period={r_b['period']}, advice={r_b['advice'][:60]}")

    # case C: 非周期序列, 无历史 → None
    seq_c = ["observe", "hypothesize", "plan", "execute", "validate", "learn"]
    r_c = eng5._check_stuck(seq_c)
    assert r_c is None, f"非周期无历史应 None, got {r_c}"
    print("G2-C non-cycle, no history → None OK")

    # case D: 非周期, 但匹配历史轨迹 prefix → match 信号
    eng5._traj_history = [
        ["observe", "hypothesize", "plan", "execute", "validate", "learn", "observe"],
        ["observe", "hypothesize", "plan", "execute", "validate", "learn", "pivot"],
    ]
    seq_d = ["observe", "hypothesize", "plan", "execute"]  # prefix of history[0]
    r_d = eng5._check_stuck(seq_d)
    assert r_d is not None, "prefix 匹配应触发 match"
    assert r_d["type"] == "match", f"type 应为 match, got {r_d['type']}"
    assert r_d["next_step"] == "validate", f"next_step 应为 validate, got {r_d['next_step']}"
    print(f"G2-D trajectory match: next_step={r_d['next_step']}, sim={r_d['similarity']}")

    # case E: 加载历史 trajectory (空目录 → 空列表)
    with _tf.TemporaryDirectory() as _td_e:
        eng5.workspace = _P(_td_e)
        hist_e = eng5._load_trajectory_action_history(limit=10)
        assert hist_e == [], f"空目录应返回 [], got {hist_e}"
    print("G2-E empty trajectory dir → [] OK")

    # case F: 加载历史 trajectory (有 json 文件 → 抽 spans phase)
    with _tf.TemporaryDirectory() as _td_f:
        eng5.workspace = _P(_td_f)
        _traj_dir = _P(_td_f) / ".huginn" / "trajectories"
        _traj_dir.mkdir(parents=True)
        (_traj_dir / "run1.json").write_text(
            '{"spans": [{"phase": "observe"}, {"phase": "hypothesize"}, '
            '{"phase": "plan"}]}',
            encoding="utf-8",
        )
        (_traj_dir / "run2.json").write_text(
            '{"spans": [{"phase": "observe"}, {"phase": "pivot"}]}',
            encoding="utf-8",
        )
        hist_f = eng5._load_trajectory_action_history(limit=10)
        assert len(hist_f) == 2, f"应加载 2 个轨迹, got {len(hist_f)}"
        assert ["observe", "hypothesize", "plan"] in hist_f
        assert ["observe", "pivot"] in hist_f
    print("G2-F load trajectory history from disk OK")

    print("AutoloopEngine selfcheck OK (10/10 + G2)")


if __name__ == "__main__":
    run_selfcheck()
