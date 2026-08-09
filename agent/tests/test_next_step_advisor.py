"""Next-step advisor 自检.

覆盖:
- _has_post_task_signal 三个触发条件 + 无信号静默
- _NEXT_STEP_ADVISOR_PROMPT 模板填充
- _advisor_post_task_recommend 调用链 (mock LLM + mock memory)
- HUMAN_PAUSE=1 走 pause_for_decision 路径

不引入 pytest — assert-based, `python -m tests.test_next_step_advisor` 可跑.
"""

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

_AGENT_DIR = Path(__file__).resolve().parent.parent
if str(_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(_AGENT_DIR))


def _make_state(hist=None, ts_list=None):
    """构造 mock LoopState — 只含 iteration_history + _physical_timeseries."""
    state = MagicMock()
    state.iteration_history = hist or []
    # _physical_timeseries 挂在 engine 上, 不挂 state
    return state


def _check_signal_no_anomaly():
    """无信号场景: 历史 passed + 无 timeseries + outcome completed → 不推."""
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._physical_timeseries = []

    state = _make_state(hist=[
        {"iter": 1, "val_status": "passed", "plan_mode": "lammps"},
        {"iter": 2, "val_status": "passed", "plan_mode": "lammps"},
    ])
    has, reason = eng._has_post_task_signal(state, prev_outcome="completed")
    assert not has, f"无信号不应触发, reason={reason}"
    assert reason == ""
    print("[ok] 无信号场景静默")


def _check_signal_iteration_repeat_failure():
    """触发1: iteration_history 最近 5 轮 failed 同一 mode >=2 次."""
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._physical_timeseries = []

    state = _make_state(hist=[
        {"iter": 1, "val_status": "passed", "plan_mode": "lammps"},
        {"iter": 2, "val_status": "failed", "plan_mode": "vasp", "advice": "SCF 不收敛"},
        {"iter": 3, "val_status": "failed", "plan_mode": "vasp", "advice": "ENCUT 太低"},
        {"iter": 4, "val_status": "passed", "plan_mode": "lammps"},
    ])
    has, reason = eng._has_post_task_signal(state, prev_outcome="completed")
    assert has, "2 次 vasp failed 应触发"
    assert "iteration_history" in reason or "failed" in reason, f"reason 应含来源: {reason}"
    print(f"[ok] 触发1 iteration_history 重复失败: {reason}")


def _check_signal_timeseries_anomaly():
    """触发2: _physical_timeseries 有 peak_v decay > 0.01."""
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._physical_timeseries = [{
        "name": "van_hove_G_s",
        "spatial": True,
        "data": [
            (0, 1.0, 0.8), (0, 2.0, 0.5),  # t=0 peak=0.8
            (100, 1.0, 0.3), (100, 2.0, 0.2),  # t=100 peak=0.3, decay=0.5
        ],
    }]

    state = _make_state(hist=[
        {"iter": 1, "val_status": "passed", "plan_mode": "lammps"},
    ])
    has, reason = eng._has_post_task_signal(state, prev_outcome="completed")
    assert has, "timeseries peak_v decay 应触发"
    assert "timeseries" in reason or "decay" in reason, f"reason 应含来源: {reason}"
    print(f"[ok] 触发2 timeseries 反常: {reason}")


def _check_signal_prev_inconclusive():
    """触发3: prev_run_context.outcome == inconclusive."""
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._physical_timeseries = []

    state = _make_state(hist=[
        {"iter": 1, "val_status": "passed", "plan_mode": "lammps"},
    ])
    has, reason = eng._has_post_task_signal(state, prev_outcome="inconclusive")
    assert has, "prev inconclusive 应触发"
    assert "inconclusive" in reason, f"reason 应含来源: {reason}"
    print(f"[ok] 触发3 prev_run_context inconclusive: {reason}")


def _check_prompt_template_filling():
    """prompt 模板所有占位符能被 .format 填充."""
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)

    prompt = eng._NEXT_STEP_ADVISOR_PROMPT.format(
        hypothesis="Fe 扩散系数随温度升高",
        outcome="inconclusive",
        anomalies="iter3: SCF 不收敛; iter4: ENCUT 太低",
        ts_anomaly="VACF (lammps): 3 pts, trend=decaying",
        prev_tail="hypothesis: Fe diffuses | outcome: inconclusive",
    )
    # 占位符都填上了
    assert "{hypothesis}" not in prompt
    assert "{outcome}" not in prompt
    assert "{anomalies}" not in prompt
    assert "{ts_anomaly}" not in prompt
    assert "{prev_tail}" not in prompt
    # 内容注入
    assert "Fe 扩散" in prompt
    assert "inconclusive" in prompt
    assert "SCF 不收敛" in prompt
    # 约束文本保留
    assert "A/B/C 不能来自同一信号源" in prompt
    assert "D 永远是用户自由出口" in prompt
    print("[ok] prompt 模板填充")


def _check_advisor_no_signal_skips():
    """无信号时 _advisor_post_task_recommend 立即 return, 不调 LLM."""
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._physical_timeseries = []
    eng._prev_run_context = ""
    eng._llm_chat = AsyncMock(return_value="should not be called")
    eng.memory = MagicMock()

    state = _make_state(hist=[{"iter": 1, "val_status": "passed"}])
    import asyncio
    asyncio.run(eng._advisor_post_task_recommend(
        "run01", "test obj", {"hypothesis": "test"}, state,
        prev_outcome="completed",
    ))
    assert not eng._llm_chat.called, "无信号不应调 LLM"
    assert not eng.memory.remember.called, "无信号不应写 memory"
    print("[ok] 无信号时 advisor 立即 return")


def _check_advisor_with_signal_calls_llm_and_memory():
    """有信号时调 LLM + 写 memory (category=next_step_hint)."""
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._physical_timeseries = [{
        "name": "van_hove_G_s", "spatial": True,
        "data": [(0, 1.0, 0.8), (100, 1.0, 0.2)],
    }]
    eng._prev_run_context = ""
    eng._llm_chat = AsyncMock(return_value="## 本轮看到\n- inconclusive\n\n## 可能的下一步\n**A. ...")
    eng.memory = MagicMock()
    eng._format_timeseries_context = lambda: "van_hove_G_s (lammps): peak_drift=+0.5"
    # 避免真实 HUMAN_PAUSE 触发
    os.environ.pop("HUGINN_AUTOLOOP_HUMAN_PAUSE", None)

    state = _make_state(hist=[
        {"iter": 1, "val_status": "passed", "plan_mode": "lammps"},
    ])
    import asyncio
    asyncio.run(eng._advisor_post_task_recommend(
        "run01", "test obj",
        {"hypothesis": "Fe diffuses", "validation": {"tests_passed": False}},
        state, prev_outcome="inconclusive",
    ))

    # LLM 被调
    assert eng._llm_chat.called, "有信号应调 LLM"
    _call = eng._llm_chat.call_args
    assert _call.kwargs.get("persona_name") == "reviewer"
    assert _call.kwargs.get("task") == "reasoning"

    # memory 写入
    assert eng.memory.remember.called, "应写 memory"
    _mem_call = eng.memory.remember.call_args
    assert _mem_call.kwargs.get("category") == "next_step_hint"
    assert "inconclusive" in _mem_call.kwargs.get("content", "")
    assert _mem_call.kwargs.get("metadata", {}).get("run_id") == "run01"
    print("[ok] 有信号时 advisor 调 LLM + 写 memory")


def _check_advisor_human_pause_triggers_decision():
    """HUMAN_PAUSE=1 时走 _await_human_decision_via_inbox."""
    from huginn.autoloop.engine import AutoloopEngine
    eng = AutoloopEngine.__new__(AutoloopEngine)
    eng._physical_timeseries = []
    eng._prev_run_context = ""
    eng._llm_chat = AsyncMock(return_value="## 推荐\n**A. 深化**")
    eng.memory = MagicMock()
    eng._format_timeseries_context = lambda: ""
    eng._await_human_decision_via_inbox = AsyncMock(return_value="A")
    eng._iteration = 5
    os.environ["HUGINN_AUTOLOOP_HUMAN_PAUSE"] = "1"
    try:
        state = _make_state(hist=[{"iter": 1, "val_status": "passed"}])
        import asyncio
        asyncio.run(eng._advisor_post_task_recommend(
            "run01", "test obj",
            {"hypothesis": "test", "validation": {"tests_passed": False}},
            state, prev_outcome="inconclusive",
        ))
        assert eng._await_human_decision_via_inbox.called, "HUMAN_PAUSE=1 应走 pause_for_decision"
        _call = eng._await_human_decision_via_inbox.call_args
        _opts = _call.args[1] if len(_call.args) > 1 else _call.kwargs.get("options")
        assert len(_opts) == 4, f"应 4 个选项 (A/B/C/D): {len(_opts)}"
        assert _opts[0]["id"] == "A"
        assert _opts[-1]["id"] == "D"
        print("[ok] HUMAN_PAUSE=1 走 pause_for_decision")
    finally:
        os.environ.pop("HUGINN_AUTOLOOP_HUMAN_PAUSE", None)


def main():
    _check_signal_no_anomaly()
    _check_signal_iteration_repeat_failure()
    _check_signal_timeseries_anomaly()
    _check_signal_prev_inconclusive()
    _check_prompt_template_filling()
    _check_advisor_no_signal_skips()
    _check_advisor_with_signal_calls_llm_and_memory()
    _check_advisor_human_pause_triggers_decision()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
