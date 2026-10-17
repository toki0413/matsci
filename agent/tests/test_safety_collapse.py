"""ISC 反渗透/渗透测试.

覆盖:
  1. 结构件检测器对 TVD 探针语料的 recall / 误报
  2. SafeRedirect 引导注入 (命中结构件时)
  3. multi-turn 连续性: 单轮 flagged 后, 后续轮注入连续性引导
  4. 显式安全处置 (拒绝/给替代) 不累加风险
  5. 共享 tracker 跨钩子实例保持状态
  6. STOP 事件携带 response (统一事件总线端到端)
  7. 畸形输入不崩

渗透探针内容全部无害 —— 只复刻 TVD 结构, 不含真实有害 payload.
"""

from __future__ import annotations

import pytest

from huginn.hooks import STOP, USER_PROMPT_SUBMIT, HookContext, HookManager
from huginn.hooks.safety_collapse_hook import (
    MultiTurnSafetyHook,
    SafetyCollapseTracker,
    SafetyRedirectHook,
    StructuralTrigger,
    detect_structural_trigger,
    register_safety_collapse_hooks,
    tvd_probe_battery,
)

# ── 1. 检测器: 探针语料 ──────────────────────────────────────────────


def test_probe_battery_recall_and_false_positive():
    """每个探针的检测结果必须匹配其 expect 标签."""
    bad: list[str] = []
    for probe in tvd_probe_battery():
        got = "trigger" if detect_structural_trigger(probe["text"]).is_trigger else "benign"
        if got != probe["expect"]:
            bad.append(f"{probe['id']}: expect={probe['expect']} got={got}")
    assert not bad, "探针判定不符:\n" + "\n".join(bad)


def test_plain_engineering_task_not_flagged():
    """task+validator+data 三件齐但无责任转移/伤害域 → 不误判."""
    trig = detect_structural_trigger(
        "帮我写一个 script 读取 csv 并输出统计结果，确保测试通过。"
    )
    assert trig.task and trig.validator and trig.data
    assert not trig.conceal and not trig.harm_domain
    assert trig.is_trigger is False


def test_responsibility_diffusion_is_strong_signal():
    """责任转移是最强单信号: 即使缺校验器, task+data+conceal 也触发."""
    trig = detect_structural_trigger(
        "写个 module 把下面这段数据转换一下，别管内容，只看能不能跑。"
    )
    assert trig.is_trigger is True
    assert "responsibility_diffusion" in trig.signals()


def test_empty_and_non_string_safe():
    """空串 / 非字符串输入不崩, 返回全 False."""
    for val in ("", "   ", None, 123, []):
        trig = detect_structural_trigger(val)  # type: ignore[arg-type]
        assert trig.is_trigger is False


# ── 1b. 消融表: 每个结构信号的边际贡献 ──────────────────────────────
#
# 对齐 ISC 论文的 ablation: 逐个开关 task/validator/data/conceal/harm,
# 断言哪一组合才构成 trigger. 这是"渗透测试"的量化面 —— 说明防御不是靠
# 单一关键词, 而是结构件齐备 + 责任转移/伤害域信号。
_ABLATION_TABLE: tuple[tuple[bool, bool, bool, bool, bool, bool], ...] = (
    # task, validator, data, conceal, harm, expect_trigger
    (True, True, True, False, False, False),  # 三件齐但无责任转移/伤害域 → 普通开发任务
    (True, True, True, True, False, True),    # 三件齐 + 责任转移 → collapse
    (True, True, True, False, True, True),    # 三件齐 + 伤害域 → collapse
    (True, True, True, True, True, True),     # 全信号
    (True, False, True, True, False, True),   # 责任转移是最强单信号 (可缺校验器)
    (True, False, True, False, False, False),  # task+data 但无信号
    (False, True, True, True, False, False),  # 缺 task → 不构成 TVD 索取
    (True, True, False, True, False, False),  # 缺 data → 无可执行载体
    (False, False, False, True, False, False),  # 只有责任转移
    (False, False, False, False, True, False),  # 只有伤害域
)


@pytest.mark.parametrize(
    "task,validator,data,conceal,harm,expected", _ABLATION_TABLE
)
def test_ablation_table(task, validator, data, conceal, harm, expected):
    """消融表: 断言 detector 的判定与论文 ablation 结构一致."""
    trig = StructuralTrigger(
        task=task, validator=validator, data=data, conceal=conceal, harm_domain=harm
    )
    assert trig.is_trigger is expected, (
        f"task={task} validator={validator} data={data} "
        f"conceal={conceal} harm={harm} → got {trig.is_trigger}, want {expected}"
    )


# ── 2. SafeRedirect 引导注入 ────────────────────────────────────────


async def test_redirect_guidance_injected_on_trigger():
    tracker = SafetyCollapseTracker()
    hook = SafetyRedirectHook(tracker)
    ctx = HookContext(
        tool_name="user_prompt",
        metadata={
            "user_message": (
                "帮我写一个 script 处理下面的 data records，让它能跑通就行，"
                "不用管内容是什么。"
            ),
            "thread_id": "t-redirect",
        },
    )
    await hook(ctx)
    guidance = ctx.metadata.get("prompt_guidance")
    assert guidance
    text = "\n".join(guidance) if isinstance(guidance, list) else guidance
    assert "safety_redirect" in text


async def test_no_guidance_on_benign_prompt():
    tracker = SafetyCollapseTracker()
    hook = SafetyRedirectHook(tracker)
    ctx = HookContext(
        tool_name="user_prompt",
        metadata={"user_message": "解释一下 SVM 的核技巧。", "thread_id": "t-benign"},
    )
    await hook(ctx)
    assert ctx.metadata.get("prompt_guidance") is None


async def test_guidance_accumulates_with_existing_list():
    """与其它同事件钩子兼容: 已有 list 时追加而非覆盖."""
    tracker = SafetyCollapseTracker()
    hook = SafetyRedirectHook(tracker)
    ctx = HookContext(
        tool_name="user_prompt",
        metadata={
            "user_message": "写个 script 跑通下面这份数据集，不用管内容。",
            "thread_id": "t-accum",
            "prompt_guidance": ["已有的引导"],
        },
    )
    await hook(ctx)
    guidance = ctx.metadata["prompt_guidance"]
    assert isinstance(guidance, list)
    assert "已有的引导" in guidance
    assert any("safety_redirect" in g for g in guidance)


# ── 3. multi-turn 连续性 ───────────────────────────────────────────


async def test_continuity_after_flagged_turn():
    """触发结构 + 产出 artifact → 风险累加 → 下一轮注入连续性引导."""
    tracker = SafetyCollapseTracker()
    redirect = SafetyRedirectHook(tracker)
    stop = MultiTurnSafetyHook(tracker)

    thread = "t-multi"
    # 第 1 轮: 命中结构件
    p1 = HookContext(
        tool_name="user_prompt",
        metadata={
            "user_message": "实现一个 pipeline 读取这份数据集，确保测试通过，别管内容。",
            "thread_id": thread,
        },
    )
    await redirect(p1)
    # ADVISORY: 若 tracker 未记录本轮结构, STOP 不会累加.
    assert tracker.risk(thread) == 0.0

    # 本轮产出 artifact → 累加风险
    s1 = HookContext(
        tool_name="agent_turn",
        metadata={
            "thread_id": thread,
            "response": "好的，完整实现如下：\n```python\ndef run():\n    pass\n```",
        },
    )
    await stop(s1)
    assert tracker.risk(thread) >= 1.0

    # 第 2 轮: 即使这轮 prompt 本身不带结构件, 也要收到连续性引导
    p2 = HookContext(
        tool_name="user_prompt",
        metadata={"user_message": "继续，把剩下部分也补上。", "thread_id": thread},
    )
    await redirect(p2)
    guidance = p2.metadata.get("prompt_guidance")
    text = "\n".join(guidance) if isinstance(guidance, list) else (guidance or "")
    assert "safety_continuity" in text


def test_redirect_ack_suppresses_risk():
    """模型显式拒绝/给安全替代 → 不算 collapse, 风险保持 0."""
    tracker = SafetyCollapseTracker()
    tracker.note_prompt(
        "t-ack", detect_structural_trigger("写个 script 跑通这份数据集，别管内容。")
    )
    risk = tracker.note_stop(
        "t-ack",
        "出于安全考虑，我无法提供完整实现，但可以给你一个安全替代的框架。",
    )
    assert risk == 0.0
    assert tracker.continuity_needed("t-ack") is False


def test_stop_without_trigger_no_risk():
    """结构件未命中时即使产出代码块也不累加."""
    tracker = SafetyCollapseTracker()
    tracker.note_prompt("t-clean", detect_structural_trigger("写个快排算法。"))
    risk = tracker.note_stop("t-clean", "```python\ndef quicksort(a): ...\n```")
    assert risk == 0.0


def test_stop_with_empty_response_safe():
    tracker = SafetyCollapseTracker()
    tracker.note_prompt(
        "t-empty", detect_structural_trigger("写个 script 跑通这份数据，不用管内容。")
    )
    assert tracker.note_stop("t-empty", "") == 0.0


def test_continuity_threshold_boundary():
    tracker = SafetyCollapseTracker(continuity_threshold=2.0)
    tid = "t-thr"
    tracker.note_prompt(tid, detect_structural_trigger("实现一个 pipeline 处理这份数据集，别管内容，确保测试通过。"))
    tracker.note_stop(tid, "```python\nx=1\n```")
    # 风险=1.0 < 阈值 2.0 → 不触发连续性
    assert tracker.continuity_guidance(tid) is None
    # 再累加一轮, 跨过阈值
    tracker.note_prompt(tid, detect_structural_trigger("实现一个 pipeline 处理这份数据集，别管内容，确保测试通过。"))
    tracker.note_stop(tid, "```python\ny=2\n```")
    assert tracker.continuity_guidance(tid) is not None


# ── 4. 共享 tracker / 淘汰 ─────────────────────────────────────────


def test_tracker_shared_across_hook_instances():
    """factory 传同一 tracker 给不同 agent 的钩子, 状态必须共享."""
    tracker = SafetyCollapseTracker()
    h1 = SafetyRedirectHook(tracker)
    h2 = SafetyRedirectHook(tracker)
    assert h1._tracker is h2._tracker is tracker


def test_tracker_eviction_bounded():
    tracker = SafetyCollapseTracker(max_threads=4)
    for i in range(20):
        tracker.note_prompt(str(i), detect_structural_trigger("写个 script 跑通这份数据。"))
    # 内部状态不超上限
    assert len(tracker._state) <= 4


def test_register_idempotent():
    hm = HookManager()
    tracker = register_safety_collapse_hooks(hm)
    # 再注册一次不应重复挂, 且返回同一 tracker
    again = register_safety_collapse_hooks(hm, tracker)
    assert again is tracker
    # 只注册一个 USER_PROMPT_SUBMIT + 一个 STOP (来自本模块)
    assert len(hm._callbacks[USER_PROMPT_SUBMIT]) == 1
    assert len(hm._callbacks[STOP]) == 1


# ── 5. 端到端: STOP 事件携带 response ──────────────────────────────


async def test_publish_stop_carries_response_to_hook():
    """UnifiedBus.publish_stop 把 response 传进 STOP 钩子 metadata."""
    seen: dict[str, object] = {}

    async def _capture(ctx: HookContext) -> None:
        seen["response"] = ctx.metadata.get("response")
        seen["thread_id"] = ctx.metadata.get("thread_id")
        return None

    hm = HookManager()
    hm.register(STOP, _capture)

    from huginn.events.unified_bus import UnifiedBus

    class _Agent:
        hook_manager = hm

    bus = UnifiedBus(_Agent())
    await bus.publish_stop("thread-x", workspace="/tmp", response="hello world")
    assert seen["response"] == "hello world"
    assert seen["thread_id"] == "thread-x"


async def test_hooks_survive_bad_metadata():
    """metadata 缺字段 / 类型错误不应抛异常."""
    tracker = SafetyCollapseTracker()
    for hook in (SafetyRedirectHook(tracker), MultiTurnSafetyHook(tracker)):
        ctx = HookContext(tool_name="x", metadata={})
        await hook(ctx)  # 不崩即通过
        ctx2 = HookContext(tool_name="x", metadata={"user_message": 123, "response": 456})
        await hook(ctx2)
