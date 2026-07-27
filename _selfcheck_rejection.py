"""自检: 连续 TASK COMPLETE 驳回计数逻辑.

模拟 Math_003 场景: agent iter 1 做完任务, iter 2-5 反复说 TASK COMPLETE.
验证: 连续 3 次驳回后, 第 4 次接受 (counter > MAX).

ponytail: 不 import rcb_runner (依赖太重), 复制核心计数逻辑做最小验证.
ceiling: 真实集成验证要跑 Math_003 extreme 20 iter.
"""
import os


def simulate_complete_loop(
    iter_responses: list[str],
    max_rejections: int = 3,
    effort_floor_always_rejects: bool = True,
) -> tuple[int, str]:
    """模拟 rcb_runner Step 2 主循环的 TASK COMPLETE 驳回逻辑.

    返回 (退出时 iter 序号, 退出原因).
    """
    _consecutive_complete_rejections = 0
    _MAX_COMPLETE_REJECTIONS = max_rejections

    for _iter_n, _ai_text in enumerate(iter_responses):
        # reset on non-TASK-COMPLETE
        if not (_ai_text and "TASK COMPLETE" in _ai_text.upper()):
            _consecutive_complete_rejections = 0
            continue

        # enter TASK COMPLETE block
        _consecutive_complete_rejections += 1
        _force_accept = _consecutive_complete_rejections > _MAX_COMPLETE_REJECTIONS
        if _force_accept:
            return _iter_n, f"force_accept (counter={_consecutive_complete_rejections})"

        # mock effort floor
        if effort_floor_always_rejects:
            # rejected, continue
            continue

        # passed
        return _iter_n, "passed"

    return len(iter_responses) - 1, "loop_exhausted"


def test_consecutive_rejections_force_accept():
    """Math_003 场景: iter 1 做事, iter 2-4 驳回, iter 5 强制接受."""
    responses = [
        "I made tool calls and wrote report.md",  # iter 0: work
        "TASK COMPLETE\nsummary...",               # iter 1: reject 1
        "TASK COMPLETE\nsummary...",               # iter 2: reject 2
        "TASK COMPLETE\nsummary...",               # iter 3: reject 3
        "TASK COMPLETE\nsummary...",               # iter 4: force accept (4 > 3)
    ]
    idx, reason = simulate_complete_loop(responses, max_rejections=3)
    assert idx == 4, f"force_accept 应在 iter 4, got iter {idx}"
    assert "force_accept" in reason, f"应 force_accept, got {reason}"
    print(f"PASS test_consecutive_rejections_force_accept: iter {idx}, {reason}")


def test_reset_on_work():
    """agent 在驳回间做了实际工作 → 计数清零, 不触发 force_accept."""
    responses = [
        "TASK COMPLETE\nsummary...",  # iter 0: reject 1, counter=1
        "I made more tool calls",     # iter 1: work, counter reset to 0
        "TASK COMPLETE\nsummary...",  # iter 2: reject 1, counter=1
        "I made more tool calls",     # iter 3: work, counter reset to 0
        "TASK COMPLETE\nsummary...",  # iter 4: reject 1, counter=1
    ]
    idx, reason = simulate_complete_loop(responses, max_rejections=3)
    # 5 个 iter 跑完, 没 force_accept (counter 始终 <= 1)
    assert reason == "loop_exhausted", f"agent 做了工作, 不应 force_accept, got {reason}"
    print(f"PASS test_reset_on_work: iter {idx}, {reason}")


def test_pass_when_effort_floor_accepts():
    """effort floor 通过 → 直接 break, 不需要 force_accept."""
    responses = [
        "TASK COMPLETE\nsummary...",  # iter 0: counter=1, effort floor 通过
    ]
    idx, reason = simulate_complete_loop(
        responses, max_rejections=3, effort_floor_always_rejects=False
    )
    assert idx == 0, f"应在 iter 0 通过, got iter {idx}"
    assert reason == "passed", f"应 passed, got {reason}"
    print(f"PASS test_pass_when_effort_floor_accepts: iter {idx}, {reason}")


def test_default_max_from_env():
    """env var HUGINN_RCB_MAX_COMPLETE_REJECTIONS 覆盖默认 3."""
    # 不改 env, 默认 3
    default_max = int(os.environ.get("HUGINN_RCB_MAX_COMPLETE_REJECTIONS", "3"))
    assert default_max == 3, f"默认应是 3, got {default_max}"
    print(f"PASS test_default_max_from_env: default_max={default_max}")


if __name__ == "__main__":
    test_consecutive_rejections_force_accept()
    test_reset_on_work()
    test_pass_when_effort_floor_accepts()
    test_default_max_from_env()
    print("\nAll selfcheck passed.")
