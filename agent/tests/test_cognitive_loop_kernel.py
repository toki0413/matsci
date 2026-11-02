"""CognitiveLoop 共享内核的行为测试 (E1: 从生产模块搬出的自检).

原 ``huginn/autoloop/cognitive_loop.py`` 末尾自带 assert 自检 + 测试替身
(``_sc_*`` / ``_MockWriter`` / ``_scr_*``), 只有 ``__main__`` 会调用它们:
生产模块背着 160 行测试代码, 而 CI 一条都没跑. 这里把它们搬进 pytest,
逐条保留原断言语义, 顺带让这些行为真正进入回归.
"""

from __future__ import annotations

from typing import Any

from huginn.autoloop.cognitive_checks import (
    classify_stall,
    darwin_ratchet_check,
    metacog_check_completion,
    metacog_check_selection_bias,
    metacog_check_topology_collapse,
)
from huginn.autoloop.cognitive_loop import (
    ActionDecision,
    CognitiveLoop,
    LoopState,
    OutputWriter,
    ReflectionResult,
)

# ── 控制流自检 (原 _sc_* 函数) ──────────────────────────────────────────


async def _observe(state: LoopState) -> dict[str, Any]:
    return {"step": state.iteration}


async def _decide(state: LoopState, obs: dict[str, Any]) -> ActionDecision:
    """第 3 轮后选 stop, 验证主循环能退出."""
    if state.iteration >= 3:
        return ActionDecision(action="stop", rationale="done")
    return ActionDecision(action="execute", rationale="working")


async def _execute(state: LoopState, decision: ActionDecision) -> Any:
    return f"result_iter_{state.iteration}"


async def _reflect(
    state: LoopState, decision: ActionDecision, result: Any
) -> ReflectionResult:
    return ReflectionResult(should_continue=True)


async def _stuck_decide(state: LoopState, obs: dict[str, Any]) -> ActionDecision:
    return ActionDecision(action="execute", rationale="stuck")


async def _bad_decide(state: LoopState, obs: dict[str, Any]) -> ActionDecision:
    return ActionDecision(action="invalid_action", rationale="bad")


class _MockWriter(OutputWriter):
    def __init__(self):
        self.steps: list[tuple[int, str]] = []

    def write_step(self, iteration, action, result, reflection):
        self.steps.append((iteration, action))


def _new_loop(writer: OutputWriter, decide, **kwargs) -> CognitiveLoop:
    return CognitiveLoop(
        observe_fn=_observe,
        decide_fn=decide,
        execute_fn=_execute,
        reflect_fn=_reflect,
        output_writer=writer,
        **kwargs,
    )


async def test_basic_flow_stops_at_third_iteration():
    writer = _MockWriter()
    state = await _new_loop(writer, _decide, max_iterations=10).run()
    assert state.iteration == 3, f"expected 3 iters, got {state.iteration}"
    assert state.should_stop, "should_stop should be True after 'stop' action"
    assert len(writer.steps) == 3, f"writer should have 3 steps, got {len(writer.steps)}"
    assert writer.steps[-1] == (3, "stop"), f"last step wrong: {writer.steps[-1]}"


async def test_stuck_loop_forces_stop_at_twice_repeat_limit():
    """死循环防护: decide 总返回相同 action."""
    state = await _new_loop(
        _MockWriter(), _stuck_decide, max_iterations=10, max_repeated_actions=3
    ).run()
    # 死循环防护: 6 次 (2x limit) 后强制 stop, should_redirect 在过程中被设过
    assert state.should_stop, "stuck loop should force stop at 2x limit"
    assert state.iteration == 6, (
        f"stuck loop should stop at 6 iters (2x limit=3), got {state.iteration}"
    )
    assert state.should_redirect, "should_redirect should be set before force stop"
    assert "死循环防护" in state.redirect_reason, (
        f"redirect_reason wrong: {state.redirect_reason}"
    )


async def test_invalid_action_is_replaced_by_skip():
    """invalid action 被替换为 'skip', 不崩溃."""
    state = await _new_loop(_MockWriter(), _bad_decide, max_iterations=2).run()
    assert state.iteration >= 1, "should run at least 1 iter"
    assert "skip" in state.action_history


# ── RCB 顶层认知反馈自检 (原 _scr_* 函数) ───────────────────────────────


def test_darwin_ratchet_four_paths():
    """darwin_ratchet_check: continue / pivot / counterexample / stop 四路."""
    act, st = darwin_ratchet_check(8.0, 0.6, 0, 3)
    assert act == "continue", f"expected continue, got {act}"
    act, st = darwin_ratchet_check(8.0, 0.6, 3, 3)
    assert act == "pivot", f"expected pivot, got {act}"
    assert st["stagnation_count"] == 0, "pivot 应重置 stagnation"
    act, st = darwin_ratchet_check(1.0, 0.1, 3, 3)
    assert act == "stop", f"expected stop, got {act}"
    act, st = darwin_ratchet_check(5.0, 0.1, 3, 3)
    assert act == "counterexample", f"expected counterexample, got {act}"
    assert st["stagnation_count"] == 0, "counterexample 应重置 stagnation"
    assert st["speculator_hint"] and "counterexample" in st["speculator_hint"].lower()


def test_classify_stall_history_paths():
    """classify_stall: history-based 四路."""
    assert classify_stall([], []) == "pivot"
    assert classify_stall([1.0, 1.0, 1.0], [0.5]) == "stop"
    assert classify_stall([5.0, 5.1, 4.9], [0.1]) == "counterexample"
    assert classify_stall([5.0, 5.1, 4.9], [0.6]) == "pivot"
    assert classify_stall([5.0, 5.1, 4.9], [0.7, 0.7, 0.5, 0.4]) == "counterexample"


def test_completion_check_four_layers():
    """metacog_check_completion: 4 层 + passed."""
    r = metacog_check_completion("no derivation here", "/nonexistent/path")
    assert r["passed"] is False
    assert "derivation_chain_coverage" in r
    assert r["repro_evidence"] < 0.5
    r2 = metacog_check_completion(
        "derivation: foo[1] ref: bar cite: baz see: qux [2] [3] [4] [5]",
        "/nonexistent/path",
        derivation_chain=["derivation: foo"],
    )
    assert r2["derivation_chain_coverage"] >= 0.5
    assert r2["evidence_strength"] >= 0.5


def test_topology_collapse_three_paths():
    """metacog_check_topology_collapse: 三路径."""
    r3 = metacog_check_topology_collapse()
    assert r3["collapsed"] is False and r3["diversity_score"] == 1.0
    r4 = metacog_check_topology_collapse(hypothesis_list=[])
    assert r4["collapsed"] is True  # 空 list 视为坍缩
    r5 = metacog_check_topology_collapse(hypothesis_list=[
        type("H", (), {"statement": "a"})(),
        type("H", (), {"statement": "a"})(),
        type("H", (), {"statement": "a"})(),
    ])
    assert r5["collapsed"] is True and r5["diversity_score"] < 0.5
    r6 = metacog_check_topology_collapse(hypothesis_list=[
        type("H", (), {"statement": "a"})(),
        type("H", (), {"statement": "b"})(),
        type("H", (), {"statement": "c"})(),
    ])
    assert r6["collapsed"] is False and r6["diversity_score"] == 1.0


def test_selection_bias_hit_miss_empty_and_hint_passthrough():
    """metacog_check_selection_bias: 命中 / 不命中 / 空 / hint 透传."""
    r7 = metacog_check_selection_bias()
    assert r7["biased"] is False and r7["n"] == 0
    r8 = metacog_check_selection_bias(
        [{"outcome": "survived"} for _ in range(5)]
    )
    assert r8["biased"] is True and r8["bias_type"] == "closed_outcome"
    assert r8["hint"] and "selection bias" in r8["hint"]
    r9 = metacog_check_selection_bias(
        [{"outcome": "a", "group": "z"}] * 2 + [{"outcome": "b", "group": "z"}] * 2
    )
    assert r9["biased"] is True and r9["bias_type"] == "single_group"
    r10 = metacog_check_selection_bias(
        [{"outcome": "a", "group": "z1"}] * 3 + [{"outcome": "b", "group": "z2"}] * 3
    )
    assert r10["biased"] is False
    r11 = metacog_check_selection_bias([{"outcome": "x"}], hint="custom hint")
    assert r11["hint"] == "custom hint"
