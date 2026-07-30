"""P4-3 自检: 验证 TurnEngine 异步 turn-based 循环的 turn 级超时 + 中断检查.

最小可运行检查 (ponytail): 造 fake 钩子, 验证:
1. turn_timeout 超时后跳过当前 turn, 设置 redirect_reason, 不阻塞后续
2. observe 后 should_stop=True 立即中断
3. decide 后 should_stop=True 立即中断
4. 正常完成所有 iterations 不超时
"""
from __future__ import annotations

import asyncio
import time


async def _test_turn_timeout_skips_and_redirects():
    """execute 钩子模拟卡住, turn_timeout 触发后跳过 + 设置 redirect."""
    from huginn.autoloop.cognitive_loop import (
        CognitiveLoop, LoopState, ActionDecision, ReflectionResult,
    )

    # execute 钩子 sleep 比超时长, 模拟卡住
    async def slow_execute(state, decision):
        await asyncio.sleep(10)
        return "should_not_reach"

    async def noop_observe(state):
        return {}

    async def noop_decide(state, obs):
        return ActionDecision(action="execute", rationale="test")

    async def noop_reflect(state, decision, result):
        return ReflectionResult()

    loop = CognitiveLoop(
        observe_fn=noop_observe,
        decide_fn=noop_decide,
        execute_fn=slow_execute,
        reflect_fn=noop_reflect,
        max_iterations=3,
        turn_timeout=0.05,  # 50ms 超时
    )
    start = time.time()
    state = await loop.run()
    elapsed = time.time() - start

    # 超时后应该跑完所有 3 个 turn, 不是 30 秒
    assert elapsed < 5.0, f"turn_timeout 没生效, elapsed={elapsed:.2f}s"
    # 最后一次 turn 超时时设置了 redirect_reason (含 timeout 字样)
    # 注意: should_redirect 会被后续 turn 的 reflect 覆盖, 不做断言
    assert state.redirect_reason is not None, "redirect_reason 未设置"
    assert "timeout" in state.redirect_reason, f"reason 不对: {state.redirect_reason}"
    # should_stop 没被设 — 超时不等于停止
    assert state.should_stop is False


async def _test_observe_interrupt():
    """observe 后 should_stop=True, 立即中断不调后续钩子."""
    from huginn.autoloop.cognitive_loop import (
        CognitiveLoop, LoopState, ActionDecision, ReflectionResult,
    )

    call_log: list[str] = []

    async def observe(state):
        call_log.append("observe")
        state.should_stop = True  # 模拟外部信号
        return {}

    async def decide(state, obs):
        call_log.append("decide")
        return ActionDecision(action="skip")

    async def execute(state, decision):
        call_log.append("execute")
        return None

    async def reflect(state, decision, result):
        call_log.append("reflect")
        return ReflectionResult()

    loop = CognitiveLoop(
        observe_fn=observe, decide_fn=decide,
        execute_fn=execute, reflect_fn=reflect,
        max_iterations=5,
    )
    state = await loop.run()
    assert state.should_stop is True
    # 只调了 observe, 没调 decide/execute/reflect
    assert call_log == ["observe"], f"不该继续调钩子: {call_log}"


async def _test_decide_interrupt():
    """decide 后 should_stop=True, 立即中断不调 execute."""
    from huginn.autoloop.cognitive_loop import (
        CognitiveLoop, ActionDecision, ReflectionResult,
    )

    call_log: list[str] = []

    async def observe(state):
        call_log.append("observe")
        return {}

    async def decide(state, obs):
        call_log.append("decide")
        state.should_stop = True
        return ActionDecision(action="skip")

    async def execute(state, decision):
        call_log.append("execute")
        return None

    async def reflect(state, decision, result):
        call_log.append("reflect")
        return ReflectionResult()

    loop = CognitiveLoop(
        observe_fn=observe, decide_fn=decide,
        execute_fn=execute, reflect_fn=reflect,
        max_iterations=5,
    )
    state = await loop.run()
    assert state.should_stop is True
    # 调了 observe + decide, 没调 execute/reflect
    assert call_log == ["observe", "decide"], f"不该调 execute: {call_log}"


async def _test_normal_completion():
    """无超时无中断, 正常跑完 max_iterations."""
    from huginn.autoloop.cognitive_loop import (
        CognitiveLoop, ActionDecision, ReflectionResult,
    )

    counter = {"n": 0}

    async def observe(state):
        return {}

    async def decide(state, obs):
        counter["n"] += 1
        return ActionDecision(action="skip", rationale=f"iter {counter['n']}")

    async def execute(state, decision):
        return f"result_{counter['n']}"

    async def reflect(state, decision, result):
        return ReflectionResult()

    loop = CognitiveLoop(
        observe_fn=observe, decide_fn=decide,
        execute_fn=execute, reflect_fn=reflect,
        max_iterations=4,
    )
    state = await loop.run()
    assert state.iteration == 4, f"应跑完 4 轮, 实际 {state.iteration}"
    assert state.should_stop is False
    # 无超时 → redirect_reason 应该是 None 或被 reflect 清掉
    assert state.last_action_result == "result_4"


async def _main():
    await _test_turn_timeout_skips_and_redirects()
    print("1. turn_timeout skips + redirects OK")
    await _test_observe_interrupt()
    print("2. observe interrupt OK")
    await _test_decide_interrupt()
    print("3. decide interrupt OK")
    await _test_normal_completion()
    print("4. normal completion OK")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    asyncio.run(_main())
