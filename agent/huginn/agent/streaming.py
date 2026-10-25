"""Chat streaming loop, phase management, and context compaction."""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from huginn.context_manager import (
    calculate_context_usage,
    format_context_usage,
)
from huginn.env_access import env_float
from huginn.hooks import (
    PRE_COMPACT,
    USER_PROMPT_SUBMIT,
    HookContext,
)
from huginn.interaction.interrupt import InterruptCancelled, get_interrupt_manager
from huginn.llm_retry import (
    _exponential_backoff,
    _get_retry_after,
    _is_context_overflow,
    _is_overloaded,
    _is_rate_limit,
    _is_transient_network,
    _jitter,
)
from huginn.models.registry import compact_threshold_for
from huginn.pet import PetMood, get_pet_bus
from huginn.phases import BudgetSpec, ResearchPhase
from huginn.privacy import redact_secrets, scan_for_secrets
from huginn.utils.context import (
    compact_messages,
    estimate_message_tokens,
    summarize_compact_messages,
)
from huginn.utils.runtime import HUGINN_DIR_NAME
from huginn.utils.session_context import (
    set_thread_id,
    set_user_message,
)
from huginn.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

# Marker the LLM can embed in its response to request a phase transition.
_PHASE_MARKER = re.compile(r"\[PHASE:\s*(\w+)\s*\]", re.IGNORECASE)


# AV5: σ₂ 补丁默认值下沉 — 生产路径长对话也会丢 system checklist / winner plan.
# 这些 marker 在普通对话里不匹配, 无副作用; benchmark-style prompt 自动受益.
_DEFAULT_ROOT_MARKERS = (
    "## Methodology Checklist;## Selected Execution Plan;"
    "## Report Coverage Compass;## Intuitive Gamer"
)

# A3: 流式 watchdog — 空闲超时后中止流式, 走非流式 ainvoke 降级.
# 60s 默认值覆盖大多数 LLM 首 token 延迟 + 中间停顿. 调高无意义, 调低误杀.
_STREAM_IDLE_TIMEOUT = float(os.environ.get("HUGINN_STREAM_IDLE_TIMEOUT", "60"))

# D1: 剩余挂钟预算 (秒), 由 autoloop 每步设置. 降级收集时用它给空闲阈值封顶,
# 不启动/不维持一个注定越过 goal 剩余预算的动作. 未设置 = None → 完全保持现行为;
# 用 contextvar 传递, **避免 streaming 反向依赖 autoloop** (依赖方向单向).
remaining_budget_s: contextvars.ContextVar[float | None] = contextvars.ContextVar(
    "huginn_remaining_budget_s", default=None
)

# D7: 挂钟预算的**绝对单调时钟 deadline**, 与上面相对值同时设置.
# 动机 (run83 实测): 相对值在**长片段内会僵死** — autoloop 每步 (或孵化器每次派
# 分支前) 把"此刻剩余"写进 remaining_budget_s, 但一个切片可能跑 300s+, 期间该值
# 不再更新. 于是"剩 30s"被冻结, 而降级收集的**总**超时 (下面 _ainvoke_timeout)
# 当时并未按预算封顶 → 仍能阻塞 300s, 直接烧穿挂钟 (run.log `fallback stream
# collect timed out (idle=30s, total=300s)`).
# deadline 一设即成**绝对时刻**, 任何时刻读取都能算出**实时**剩余 = deadline - now,
# 不受冻结影响. 这是"不设硬时长、只做更细切片"的最小实现: 每个阻塞动作自限到
# **当前**剩余 (而非某个固定档), 天然随预算收敛.
budget_deadline_monotonic: contextvars.ContextVar[float | None] = (
    contextvars.ContextVar("huginn_budget_deadline_monotonic", default=None)
)


def set_budget_deadline(remaining_s: float | None) -> None:
    """把相对剩余预算存成**绝对** deadline (单调时钟). None = 不限制 (旧行为).

    与 ``remaining_budget_s`` 配套使用: 后者保留给按相对值判断的旧调用点, 这里
    额外提供可实时求值的绝对时刻. 失败不抛 (预算源异常不应影响 LLM 主流程).
    """
    budget_deadline_monotonic.set(
        None if remaining_s is None else time.monotonic() + float(remaining_s)
    )


def live_budget_left() -> float | None:
    """**实时**剩余挂钟秒数: 优先绝对 deadline, 无则回退相对 contextvar.

    None = 不受预算约束 (非长程 / 无 goal) → 调用方保持旧行为. 这是 D7 的单一
    取值口: 冻结的相对值只在无 deadline 时兜底, 有 deadline 时永远算实时的.
    """
    deadline = budget_deadline_monotonic.get()
    if deadline is not None:
        return deadline - time.monotonic()
    return remaining_budget_s.get()


def _cap_timeout_by_budget(
    total: float,
    budget_left: float | None,
    *,
    reserve: float | None = None,
) -> float:
    """D7: 把单个阻塞动作的**总**超时封顶到**实时**剩余预算 (留收尾余量).

    ``budget_left=None`` (非长程 / 无 goal) → 原样返回, 行为 100% 不变. 下限避免
    0/负值造成病态立即超时 —— 预算真耗尽时应快速失败交还控制权, 而非再阻塞.

    收尾余量与下限经配置面取值 (``HUGINN_BUDGET_RESERVE_S`` / ``_MIN_SLICE_S``,
    声明见 ``env_schema``), 让"留多少收尾 / 最短自限"可按任务调, 不必改码. 显式传
    ``reserve`` 时以入参为准 (供调用方覆盖).
    """
    if budget_left is None:
        return total
    _reserve = env_float("HUGINN_BUDGET_RESERVE_S") if reserve is None else reserve
    _floor = env_float("HUGINN_BUDGET_MIN_SLICE_S")
    return max(_floor, min(total, budget_left - _reserve))


# ainvoke 超时随 thinking 强度放宽 — 我们鼓励深度思考 (thinking=high) 却给
# 固定 300s, 长推理一超就 kill, 自相矛盾. 按档位给足预算, 避免"三思而后行
# 却被限时"的错配. ponytail: 离散档位映射, env 可覆盖. ceiling: 连续缩放需
# 按 max_tokens / context 动态估算, 留作升级.
def _thinking_scale_timeout() -> float:
    t = os.environ.get("HUGINN_THINKING", "").lower()
    if t in ("high", "deep", "max", "extreme"):
        return 900.0
    if t in ("medium", "med", "normal"):
        return 600.0
    return 300.0


def _thinking_stream_idle() -> float:
    """流式空闲超时也随 thinking 放宽 — 深度推理时首 token 前的 thinking 阶段
    可能长时间无 content chunk, 60s 会误杀后降级 ainvoke. 与 ainvoke 超时同源,
    但只有 ainvoke 的 1/5 (chunk 间空闲比整段推理短). ponytail: 离散档位.
    """
    t = os.environ.get("HUGINN_THINKING", "").lower()
    if t in ("high", "deep", "max", "extreme"):
        return 180.0
    if t in ("medium", "med", "normal"):
        return 120.0
    return _STREAM_IDLE_TIMEOUT


def _fallback_stream_idle(
    total_timeout: float,
    budget_left: float | None = None,
) -> float:
    """降级收集的空闲超时 — 必须比刚失败的那次更宽, 否则只是重演同一失败.

    进入降级 = 主流已证实会空闲超过 ``_thinking_stream_idle()``. 长耗时工具
    (code_lab 可跑 ~150s, 期间无任何 chunk) 正是这种"合法长空闲"; 用同一阈值
    再跑一遍注定同样超时, 白烧整个 ``total_timeout`` (run72 实测两次各 300s).
    故降级期把空闲阈值放宽到总预算量级: 只在真正卡死时提前终止, 其余交给外层
    ``wait_for(total_timeout)`` 兜底. ``HUGINN_FALLBACK_STREAM_IDLE`` 可覆盖.

    D1: ``budget_left`` 给定时, 再对本阈值**封顶**, 不让降级消耗超过 goal 剩余
    挂钟预算 (下限 1s, 避免 0/负值造成病态立即超时). 未给定 (None) 完全等价旧行为.
    """
    override = os.environ.get("HUGINN_FALLBACK_STREAM_IDLE")
    if override:
        try:
            widened = float(override)
            if budget_left is not None:
                widened = min(widened, max(1.0, budget_left))
            return widened
        except ValueError:
            logger.debug("bad HUGINN_FALLBACK_STREAM_IDLE=%r", override)
    widened = max(_thinking_stream_idle(), total_timeout)
    if budget_left is not None:
        widened = min(widened, max(1.0, budget_left))
    return widened


def _fallback_collect_inputs(states_yielded: int, inputs: Any) -> Any:
    """降级收集该喂什么输入 — 有进度就续跑, 别重放整个 turn.

    主流已 yield 过状态 = 有 checkpoint 进度. 此时回 ``None``, langgraph 会从最后
    checkpoint 原生恢复 (见 ``runtime/checkpoint.py``), 不重做已完成步骤 — 这正是
    run72 的根因: 旧代码一律重放 ``inputs``, 把 40+ 步重做一遍 → 累计必然超
    ``_ainvoke_timeout``, 两次各烧 300s 且产出为空. 零进度时无 checkpoint 可续,
    只能回 ``inputs`` 重放. ``HUGINN_STREAM_FALLBACK_RESUME=0`` 强制重放 (回滚开关).
    """
    if (
        states_yielded > 0
        and os.environ.get("HUGINN_STREAM_FALLBACK_RESUME", "1") == "1"
    ):
        return None
    return inputs


def _fallback_params(turn: Any) -> tuple[float, float, Any]:
    """算降级收集的三个参数: 总超时 / 空闲阈值 / 续跑输入 (均按预算封顶)."""
    ainvoke_base = float(
        os.environ.get("HUGINN_AINVOKE_TIMEOUT", str(_thinking_scale_timeout()))
    )
    budget_left = live_budget_left()
    ainvoke_timeout = _cap_timeout_by_budget(ainvoke_base, budget_left)
    collect_idle = _fallback_stream_idle(ainvoke_timeout, budget_left=budget_left)
    collect_inputs = _fallback_collect_inputs(turn.states_yielded, turn.inputs)
    return ainvoke_timeout, collect_idle, collect_inputs


async def _fallback_collect(
    collect_states: Any,
    turn: Any,
    collect_inputs: Any,
    collect_idle: float,
    ainvoke_timeout: float,
) -> Any:
    """跑降级收集, 超时/空状态转成明确的异常."""
    try:
        final_state = await asyncio.wait_for(
            collect_states(turn.graph, turn.config, collect_inputs, collect_idle),
            timeout=ainvoke_timeout,
        )
    except TimeoutError:
        logger.warning(
            "fallback stream collect timed out (idle=%ds, total=%ds)",
            collect_idle,
            ainvoke_timeout,
        )
        raise TimeoutError(
            f"stream and fallback collect both timed out "
            f"(collect idle={collect_idle}s, total={ainvoke_timeout}s)"
        ) from None
    if final_state is None:
        raise RuntimeError("fallback stream collect produced no state") from None
    return final_state


async def _astream_with_watchdog(
    aiter: AsyncIterator,
    idle_timeout: float = _STREAM_IDLE_TIMEOUT,
    total_timeout: float | None = None,
) -> AsyncIterator:
    """包装 async iterator, 空闲超时抛 asyncio.TimeoutError.

    每次取下一个 chunk 用 asyncio.wait_for 限时. 超时 → 上层捕获后走 ainvoke 降级.
    ponytail: 不在这里做降级, 只负责报时. 降级逻辑在 chat() 里, 因为需要 graph + inputs.

    D7 补漏 (run86): ``total_timeout`` 给定时, 整段流另受**总时长**约束. 原实现只有
    空闲超时 —— 一条**涓流** (chunk 间隔始终 < ``idle_timeout``, 但整体极慢) 会不断
    重置空闲计时, 永不触发, 单段 phase 被拖到远超挂钟 (run86 实测: 预算 700s, 实跑
    1300s+, `/proc/<pid>/io` 的 rchar 以 ~3.8KB/s 持续增长). 每次取 chunk 的实际等待
    取 ``min(idle_timeout, 距 deadline 剩余)``, 并每轮先查 deadline ⇒ 到点即抛, 与
    "chunk 是否还在到"无关. ``total_timeout=None`` (非长程) 完全等价旧行为.
    """
    _deadline = None if total_timeout is None else time.monotonic() + total_timeout
    while True:
        if _deadline is None:
            _wait = idle_timeout
        else:
            _left = _deadline - time.monotonic()
            if _left <= 0:
                raise TimeoutError("stream total timeout")
            _wait = min(idle_timeout, _left)
        try:
            item = await asyncio.wait_for(aiter.__anext__(), timeout=_wait)
        except StopAsyncIteration:
            logger.debug("best-effort op failed", exc_info=True)
            return
        yield item


def _strip_dangling_tool_calls(messages: list) -> int:
    """C1: 剥掉 AIMessage 里没有对应 ToolMessage 的 dangling tool_calls.

    返回剥掉的 tool_call 总数. 原地修改 messages 列表 (替换 AIMessage).
    全 dangling 的 AIMessage 退化为纯 AIMessage (保留 content).

    与 context_builder.conversation_tree_to_messages 的剥掉逻辑同源,
    但作用在 _build_input_messages 返回的列表上 (checkpointer 直读路径).
    """
    if not messages:
        return 0
    # 收集所有 ToolMessage 的 tool_call_id
    answered_ids: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage):
            tc_id = getattr(m, "tool_call_id", None)
            if tc_id:
                answered_ids.add(tc_id)
    # 剥掉 AIMessage 里未应答的 tool_calls
    n_stripped = 0
    for i, m in enumerate(messages):
        if not isinstance(m, AIMessage):
            continue
        tcs = getattr(m, "tool_calls", None)
        if not tcs:
            continue
        kept = [tc for tc in tcs if tc.get("id") in answered_ids]
        if len(kept) == len(tcs):
            continue  # 全应答, 不动
        n_stripped += len(tcs) - len(kept)
        if kept:
            # 部分剥掉: 保留已应答的
            messages[i] = AIMessage(
                content=m.content, tool_calls=kept, id=getattr(m, "id", None)
            )
        else:
            # 全 dangling: 退化为纯 AIMessage
            messages[i] = AIMessage(
                content=m.content, tool_calls=[], id=getattr(m, "id", None)
            )
    return n_stripped


def _extract_plan_steps_for_compact(final_state: dict[str, Any] | None) -> list[Any]:
    """从 final_state 里 duck-typed 抽取计划步骤, 供 OnlineCompact 门控(advisory).

    纯 best-effort: 状态里可能有 ``steps`` / ``plan_steps`` / ``plan``(其下
    ``steps`` 或 ``tasks``) 任一形态; 什么都不在就不对该轮做候选门控(返回空)。
    不抛出: 找不到就静默返回空表。
    """
    if not final_state:
        return []
    candidates: list[Any] | None = None
    for key in ("steps", "plan_steps", "tasks"):
        val = final_state.get(key)
        if isinstance(val, list):
            candidates = val
            break
    if candidates is None:
        plan = final_state.get("plan")
        if isinstance(plan, dict):
            for key in ("steps", "tasks"):
                val = plan.get(key)
                if isinstance(val, list):
                    candidates = val
                    break
    return candidates or []


def _c1_self_check() -> int:
    """C1 self-check: 验证 _strip_dangling_tool_calls 三场景.

    ponytail: 纯函数测试, 不依赖 LLM/checkpointer. ceiling: 不验发送路径集成.
      升级路径: 跑真 agent, 故意造 dangling, 看 400 是否消失.
    """
    # 1. 全应答: 不动
    msgs_ok = [
        AIMessage(content="ok", tool_calls=[{"id": "tc1", "name": "f", "args": {}}]),
        ToolMessage(content="r1", tool_call_id="tc1"),
    ]
    assert _strip_dangling_tool_calls(msgs_ok) == 0, "全应答不应剥掉"
    assert len(msgs_ok[0].tool_calls) == 1, "全应答 tool_calls 不应变"

    # 2. 全 dangling: 退化为纯 AIMessage
    msgs_dangling = [
        AIMessage(content="d", tool_calls=[{"id": "tc2", "name": "f", "args": {}}]),
    ]
    n = _strip_dangling_tool_calls(msgs_dangling)
    assert n == 1, f"全 dangling 应剥掉 1, got {n}"
    assert msgs_dangling[0].tool_calls == [], "全 dangling 应退化为空 tool_calls"
    assert msgs_dangling[0].content == "d", "content 应保留"

    # 3. 部分 dangling: 保留已应答的
    msgs_partial = [
        AIMessage(
            content="p",
            tool_calls=[
                {"id": "tc3", "name": "f", "args": {}},
                {"id": "tc4", "name": "g", "args": {}},
            ],
        ),
        ToolMessage(content="r3", tool_call_id="tc3"),
    ]
    n = _strip_dangling_tool_calls(msgs_partial)
    assert n == 1, f"部分 dangling 应剥掉 1, got {n}"
    assert len(msgs_partial[0].tool_calls) == 1, "部分 dangling 应保留 1"
    assert msgs_partial[0].tool_calls[0]["id"] == "tc3", "应保留已应答的 tc3"

    print("[CHECK C1.1] all-answered: no strip OK")
    print("[CHECK C1.2] all-dangling: degrade to pure AIMessage OK")
    print("[CHECK C1.3] partial-dangling: keep answered, strip dangling OK")
    print("[CHECK C1] ALL ASSERTS PASSED")
    return 0


def _load_root_markers() -> list[str] | None:
    """F3+AV5: 从 HUGINN_ROOT_MARKERS env 读内容 marker (分号分隔).

    用于 compact_messages 的 root_content_markers 参数 — 标记 checklist prompt、
    FCM winner_plan 等关键 early-turn message 永不被 drop.
    ponytail: 模块级缓存, env 不变就只读一次.
    """
    raw = os.environ.get("HUGINN_ROOT_MARKERS", _DEFAULT_ROOT_MARKERS).strip()
    if not raw:
        return None
    return [m.strip() for m in raw.split(";") if m.strip()]


def _is_root_message(msg: Any) -> bool:
    """判断消息是否被显式标记为 root (永不被 compaction drop).

    调用方在构建关键 early-turn 消息 (checklist prompt / FCM winner_plan 等)
    时设置 ``msg.additional_kwargs["is_root"] = True`` (或 ``msg.metadata["is_root"]``
    / 简写 ``root``), 该消息即可跨压缩保留, 不再依赖位置切片 (keep_root_n).

    支持的标记位 (任一为真即视为 root):
      - ``msg.additional_kwargs["is_root"]``
      - ``msg.additional_kwargs["root"]``  (简写兼容)
      - ``msg.metadata["is_root"]``

    无标记返回 False — 由调用方回退到按位置 (前 keep_root_n 条) 判断.
    """
    for container in (
        getattr(msg, "additional_kwargs", None),
        getattr(msg, "metadata", None),
    ):
        if not isinstance(container, dict):
            continue
        if container.get("is_root") or container.get("root"):
            return True
    return False


def _msg_text(content: Any) -> str:
    """把消息 content (str / 内容块 list) 统一成纯文本."""
    if isinstance(content, list):
        return "".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content) if content else ""


def _has_assistant_text(msgs: list) -> bool:
    """已有非空助手文本 → 无需收尾兜底."""
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and m.content and str(m.content).strip():
            return True
    return False


def _closing_transcript(msgs: list) -> list[str]:
    """抽取 human/assistant 可见文本, 每段截断到 800 字符."""
    transcript: list[str] = []
    for m in msgs:
        role = getattr(m, "type", "unknown")
        content = _msg_text(getattr(m, "content", ""))
        if role in ("human", "user", "assistant") and content:
            transcript.append(f"{role}: {content[:800]}")
    return transcript


_CLOSING_PROMPT = (
    "The conversation below ended without a clear final answer. "
    "Write a concise closing summary (2-5 sentences): what was done, "
    "and the key conclusions or open questions. If information is "
    "insufficient, say so honestly rather than guessing.\n\n"
)


async def _invoke_closing_summary(model: Any, transcript: list[str]) -> str:
    """让模型把对话收尾成一段概要文本."""
    from langchain_core.messages import HumanMessage, SystemMessage

    resp = await model.ainvoke(
        [
            SystemMessage(content="You are a concise, honest research summarizer."),
            HumanMessage(content=_CLOSING_PROMPT + "\n".join(transcript[-12:])),
        ]
    )
    return _msg_text(resp.content).strip()


def _pipeline_context_block() -> str:
    """模拟管线上下文块; 失败返回空串."""
    try:
        from huginn.provenance.pipeline import SimulationPipeline

        return SimulationPipeline().to_context_block()
    except Exception:  # 防御: 管线上下文块跳过则省略
        logger.debug("pipeline context block skipped", exc_info=True)
        return ""


def _provenance_context_blocks() -> tuple[str, str]:
    """溯源上下文块 + 文件 DAG mermaid 块; 失败各返回空串."""
    try:
        from huginn.provenance import to_mermaid_for_context
        from huginn.provenance.registry import ProvenanceRegistry

        reg = ProvenanceRegistry.shared()
        return reg.to_context_block(), to_mermaid_for_context(reg)
    except Exception:  # 防御: 溯源上下文块跳过则省略
        logger.debug("provenance context block skipped", exc_info=True)
        return "", ""


def _task_state_context_block(host: Any) -> str:
    """长程任务状态上下文块; 无 thread / 失败返回空串."""
    try:
        from huginn.memory.task_state import get_tracker

        tid = getattr(host, "thread_id", "") or ""
        return get_tracker().context_block(tid) if tid else ""
    except Exception:  # 防御: 任务状态上下文跳过则省略
        logger.debug("task state context block skipped", exc_info=True)
        return ""


def _synthetic_continue_parts(
    pipeline_block: str,
    prov_block: str,
    dag_block: str,
    task_block: str,
) -> list[str]:
    """拼装 synthetic Continue 的正文分段."""
    parts = ["[System] Continue if you have next steps."]
    if pipeline_block:
        parts.append(pipeline_block)
    if prov_block:
        parts.append(prov_block)
    if dag_block:
        parts.append(f"### File DAG (recent):\n```mermaid\n{dag_block}\n```")
    if task_block:
        parts.append(task_block)
    parts.append(
        "If the pipeline suggests a next step, proceed with it. "
        "If you've completed the task, summarize the results."
    )
    return parts


def _message_token_counts(msgs: list) -> list[int]:
    """逐条消息的 token 估算."""
    from huginn.utils.context import _msg_content, _msg_role
    from huginn.utils.tokens import count_message_tokens

    return [count_message_tokens(_msg_content(m), _msg_role(m)) for m in msgs]


def _over_trim_limits(
    total: int, count: int, budget: int, max_messages: int | None
) -> bool:
    """是否超 token 预算或消息数上限 (两者都未超 → 无需修剪)."""
    under_budget = total <= budget
    under_count = max_messages is None or count <= max_messages
    return not (under_budget and under_count)


def _root_indices(msgs: list, keep_root_n: int) -> set[int]:
    """root 消息下标: 优先 metadata 标记, 无标记回退到按位置前 keep_root_n 条."""
    root_indices = {i for i, m in enumerate(msgs) if _is_root_message(m)}
    if root_indices:
        return root_indices
    return set(range(min(keep_root_n, len(msgs))))


def _select_drop_indices(
    droppable_order: list[int],
    per_msg_tokens: list[int],
    total: int,
    count: int,
    budget: int,
    max_messages: int | None,
) -> list[int]:
    """从最旧开始选要 drop 的下标, 直到 token 与消息数都落回上限内."""
    drop_indices: list[int] = []
    acc = total
    remaining = count
    for i in droppable_order:
        over_budget = acc > budget
        over_count = max_messages is not None and remaining > max_messages
        if not over_budget and not over_count:
            break
        acc -= per_msg_tokens[i]
        remaining -= 1
        drop_indices.append(i)
    return drop_indices


def _drop_message_ids(msgs: list, drop_indices: list[int]) -> list[str]:
    """收集要删除消息的 ID — system 消息不删, 没 ID 的没法删."""
    from langchain_core.messages import SystemMessage

    drop_ids: list[str] = []
    for i in drop_indices:
        m = msgs[i]
        if isinstance(m, SystemMessage):
            continue
        mid = getattr(m, "id", None)
        if mid:
            drop_ids.append(mid)
    return drop_ids


async def _tcm_snapshot_messages(graph: Any, config: dict[str, Any]) -> list:
    """取 checkpointer 现有 messages (含 ID); 空/过短返回空列表."""
    snapshot = await asyncio.to_thread(graph.get_state, config)
    if snapshot is None or not snapshot.values:
        return []
    msgs = snapshot.values.get("messages", [])
    return msgs if len(msgs) > 4 else []


def _tcm_plan(
    msgs: list,
    per_msg_tokens: list[int],
    budget: int,
    max_messages: int | None,
) -> list[int]:
    """算要 drop 的消息下标 (空列表 = 无需修剪)."""
    total = sum(per_msg_tokens)
    if not _over_trim_limits(total, len(msgs), budget, max_messages):
        return []
    keep_last_n = 4
    # AV5: 默认保前 2 条 root (user 初始任务 + system checklist).
    keep_root_n = int(os.environ.get("HUGINN_KEEP_ROOT_N", "2"))
    root_indices = _root_indices(msgs, keep_root_n)
    # 可 drop 候选 = 尾部 keep_last_n 之外、且非 root 的消息, 从最旧开始丢
    droppable_order = [
        i for i in range(len(msgs) - keep_last_n) if i not in root_indices
    ]
    if not droppable_order:
        return []
    return _select_drop_indices(
        droppable_order, per_msg_tokens, total, len(msgs), budget, max_messages
    )


async def _tcm_apply(graph: Any, config: dict[str, Any], drop_ids: list[str]) -> None:
    """用 RemoveMessage 批量删除 checkpointer 里的旧消息."""
    from langchain_core.messages import RemoveMessage

    removals = [RemoveMessage(id=mid) for mid in drop_ids]
    # update_state 是同步方法, 走 to_thread 不卡 event loop (G33 一致性)
    await asyncio.to_thread(graph.update_state, config, {"messages": removals})


def _compute_common_prefix(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    """返回两个 record 列表的公共前缀长度 (相同 type+content 连续计数).

    P2 prefix_merging: 跨 turn 共享前缀去重. 用 ((type, content) 序列) 判相等,
    ts / tool_call_id 等噪声字段忽略, 避免同一轮重复工具序列被误判为 0.
    """
    n = 0
    for x, y in zip(a, b):
        if x.get("type") != y.get("type"):
            break
        if x.get("content") != y.get("content"):
            break
        n += 1
    return n


def _read_completion_records(path: Path) -> list[dict[str, Any]]:
    """整读一个 completion jsonl 文件为 record 列表. 失败/不存在返回 []."""
    import json

    try:
        if not path.exists():
            return []
        out = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("skip bad completion line in %s", path)
        return out
    except Exception:  # 防御: 读取补全记录失败则返回空
        logger.debug("read completion records failed", exc_info=True)
        return []


def _completion_prefix_ref(
    comp_dir: Path, turn_id: str, records: list[dict[str, Any]]
) -> dict[str, Any] | None:
    """找同 thread 最新的前一 turn 文件, 计算可复用的公共前缀."""
    prev_path = _latest_completion_path(comp_dir, exclude=f"{turn_id}.jsonl")
    if prev_path is None:
        return None
    prev_recs = _read_completion_records(prev_path)
    if not prev_recs:
        return None
    common = _compute_common_prefix(prev_recs, records)
    # 阈值 2: 至少共享 2 条才值得去重 (1 条引用开销不划算).
    if common < 2:
        return None
    return {
        "type": "prefix_ref",
        "prev_file": prev_path.name,
        "prefix_len": common,
    }


def _completion_lines(
    records: list[dict[str, Any]], prefix_ref: dict[str, Any] | None
) -> list[str]:
    """把记录渲染成 jsonl 行; 有 prefix_ref 时只写后缀增量."""
    import json

    if prefix_ref is None:
        return [json.dumps(rec, ensure_ascii=False, default=str) for rec in records]
    lines = [json.dumps(prefix_ref, ensure_ascii=False)]
    # 只写后缀增量 records[common:], 前缀从 prev_file 重建.
    lines.extend(
        json.dumps(rec, ensure_ascii=False, default=str)
        for rec in records[prefix_ref["prefix_len"] :]
    )
    return lines


def _dump_completion_records(
    records: list[dict[str, Any]],
    thread_id: str,
    turn_id: str,
) -> Any:
    """把 wire-level completion records 落盘为 jsonl, 供 red_team / RL 训练消费.

    每条 record 是 ``_process_stream_state`` 抓的 assistant/tool_call/tool_result
    结构化快照. 一个 turn 一个文件:
    ``<runtime_home>/completions/<thread_id>/<turn_id>.jsonl``.

    P2 prefix_merging: 同一 thread 连续 turn 若存在公共前缀 (如重复的工具调用/
    重放序列), 只写一条 prefix_ref 引用前一文件 + 后缀增量, 减少冗余存储.
    Consumers 读侧据 prefix_ref 重建完整序列 (见 _reconstruct_completion_records).
    无公共前缀时写全量, 行为不变. 失败静默 (返回 None).
    """
    if not records:
        return None
    try:
        from huginn.utils.runtime import get_runtime_home

        comp_dir = get_runtime_home() / "completions" / thread_id
        comp_dir.mkdir(parents=True, exist_ok=True)
        comp_path = comp_dir / f"{turn_id}.jsonl"

        lines = _completion_lines(
            records, _completion_prefix_ref(comp_dir, turn_id, records)
        )
        with open(comp_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return comp_path
    except Exception:  # 防御: 补全记录落盘失败则放弃
        logger.debug("completion dump failed", exc_info=True)
        return None


def _latest_completion_path(comp_dir: Path, exclude: str = "") -> Path | None:
    """返回 comp_dir 下按文件名倒序的最新 jsonl (排除 exclude). 无则 None."""
    try:
        cands = [p for p in comp_dir.glob("*.jsonl") if p.name != exclude]
        if not cands:
            return None
        return max(cands, key=lambda p: p.name)
    except OSError:
        logger.debug("latest completion path failed", exc_info=True)
        return None


def _reconstruct_completion_records(path: Path) -> list[dict[str, Any]]:
    """整读一个 completion jsonl, 按 prefix_ref 重建完整 record 序列.

    P2 prefix_merging 读侧: 首行是 prefix_ref 时, 递归读 prev_file 的完整序列,
    取前 prefix_len 条 + 当前文件后缀增量. 供 red_team / RL 消费端使用.
    """
    recs = _read_completion_records(path)
    if not recs:
        return recs
    if recs[0].get("type") == "prefix_ref":
        prefix_len = int(recs[0].get("prefix_len", 0))
        prev_path = path.with_name(str(recs[0].get("prev_file", "")))
        prev_recs = _reconstruct_completion_records(prev_path)
        return list(prev_recs[:prefix_len]) + recs[1:]
    return recs


@dataclass
class _TurnStream:
    """Per-turn mutable context shared by the helpers split out of ``chat``.

    ``chat`` stays an async generator (it must ``yield`` events and honour
    early ``return``s); everything else moves into helpers that mutate this
    object instead of a dozen non-local variables.
    """

    graph: Any
    inputs: dict[str, Any]
    config: dict[str, Any]
    use_sync_stream: bool
    turn_span: Any
    thread_id: str
    pet: Any
    completion_records: list[dict[str, Any]]
    ubus: Any
    message: str
    primary_idle: float
    tps_t0: float
    tps_t_first: float | None = None
    tps_chunk_chars: int = 0
    states_yielded: int = 0
    final_state: Any = None
    last_reasoning: str = ""
    stopped: bool = False
    final_ai_content: str = ""
    include_history: bool | None = None
    vision_content: Any = None
    cv_hints: Any = None
    capture_turn_id: str = ""
    budget_override: Any = None


class StreamingMixin:
    """The chat() async generator and all streaming-adjacent logic."""

    # ── Phase management ──────────────────────────────────────────

    @property
    def phase(self) -> str:
        """Current research phase as a string."""
        return self._phase_manager.phase.value

    @property
    def phase_history(self) -> list[str]:
        return [p.value for p in self._phase_manager.history]

    def set_phase(self, phase: str) -> bool:
        """Transition to a new research phase.

        Returns True if the transition was allowed, False otherwise.
        Rebuilds the graph so the phase-specific prompt prefix takes effect.
        """
        try:
            target = ResearchPhase(phase)
        except ValueError:
            logger.debug("best-effort op failed", exc_info=True)
            return False
        if not self._phase_manager.transition(target):
            return False
        self._agent_graph = None
        self._invalidate_tool_description_cache()
        logger.info("Research phase -> %s", target.value)
        return True

    def transition_phase(self, target_phase: ResearchPhase) -> bool:
        """Transition to target_phase and invalidate the cached agent graph."""
        if not self._phase_manager.transition(target_phase):
            return False
        self._agent_graph = None
        self._invalidate_tool_description_cache()
        # OAK 启发: 进入 hypothesis/planning 时 fork 对话树, 标记新实验分支
        # 失败的分支保留在树里, 成功路径合并回主干 — 树状研究历史
        try:
            if target_phase.value in ("hypothesis", "planning"):
                self._conversation_tree.fork_from_active()
            # P1-2 接线: 进入 reporting 时把所有兄弟分支的 findings/evidence
            # CRDT 合并回当前 active leaf. 不切 active, 只灌 metadata.
            # 之前只有 fork 没 merge, 跨分支信息丢在树里没人读.
            if target_phase.value == "reporting":
                branches = self._conversation_tree.get_branches()
                active_leaf = self._conversation_tree.active_leaf_id
                for branch in branches:
                    if not branch:
                        continue
                    leaf = branch[-1]
                    if leaf == active_leaf:
                        continue
                    try:
                        self._conversation_tree.merge_branch_into_active(leaf)
                    except Exception:  # 防御: 分支合并失败则跳过该分支
                        logger.debug(
                            "ConversationTree merge skipped for branch %s",
                            leaf,
                            exc_info=True,
                        )
        except Exception:  # 防御: 对话树切换操作失败则忽略
            logger.debug("ConversationTree fork/merge skipped", exc_info=True)
        logger.info("Research phase -> %s", target_phase.value)
        return True

    def _check_phase_transition(self, ai_content: str) -> ResearchPhase | None:
        """Extract a phase transition request from the LLM's output."""
        match = _PHASE_MARKER.search(ai_content)
        if not match:
            return None
        phase_name = match.group(1).upper()
        try:
            return ResearchPhase[phase_name]
        except KeyError:
            logger.debug("best-effort op failed", exc_info=True)
            return None

    @staticmethod
    def _extract_last_ai_content(state: dict[str, Any]) -> str:
        """Pull the text of the most recent assistant message from a graph state."""
        msgs = state.get("messages", [])
        for msg in reversed(msgs):
            if isinstance(msg, AIMessage):
                content = msg.content
                if isinstance(content, str):
                    return content
                parts = [
                    block.get("text", "")
                    for block in content
                    if isinstance(block, dict) and "text" in block
                ]
                return "".join(parts)
        return ""

    def _find_local_model(self) -> Any | None:
        """Find a local provider (ollama/vllm/local) model from the router."""
        if self.model_router is None:
            return None
        for entry in getattr(self.model_router, "_models", {}).values():
            m = entry.model
            llm_type = (getattr(m, "_llm_type", "") or "").lower()
            cls_name = m.__class__.__name__.lower()
            if any(k in llm_type or k in cls_name for k in ("ollama", "vllm", "local")):
                return m
        return None

    # ── Stream state processing ───────────────────────────────────

    def _process_stream_state(
        self,
        state: dict[str, Any],
        turn_span: Any,
        thread_id: str,
        pet: Any,
        records: list[dict[str, Any]] | None = None,
    ) -> None:
        """Update memory, branch tree, telemetry, and pet status from one graph state.

        records: 可选 list, 传入则按 wire-level 抓 prompt/response/tool_call/tool_result.
        Polar 启发: stream 层统一抓, 不在 harness 里散写 logger.
        """
        msgs = state.get("messages", [])
        offset = self._state_msg_offsets.get(thread_id, 0)
        new_msgs = msgs[offset:]
        self._state_msg_offsets[thread_id] = len(msgs)
        # Class-qualified calls: tests drive this method with a mock `self`, so
        # attribute dispatch would resolve to auto-created mocks, not the helpers.
        for msg in new_msgs:
            if isinstance(msg, AIMessage):
                StreamingMixin._record_ai_message(self, msg, records)
            elif isinstance(msg, ToolMessage):
                StreamingMixin._record_tool_message(self, msg, records)
        StreamingMixin._publish_cache_stats(self, msgs, turn_span, thread_id, pet)

    def _record_ai_message(
        self, msg: AIMessage, records: list[dict[str, Any]] | None
    ) -> None:
        """Persist one assistant message: memory, reasoning, tree, tool calls."""
        self.memory.add_message("assistant", msg.content)
        # COT 资产化: 把模型暴露的 reasoning_content (如 DeepSeek) 落进
        # session.reasoning_trace, 供下游蒸馏 (knowledge_distiller /
        # evolution) 消费。无需去重: 每条 AIMessage 经 offset 只处理一次,
        # 且 reasoning 是整段累积在 final message 上的。
        _reasoning = (
            msg.additional_kwargs.get("reasoning_content", "")
            if getattr(msg, "additional_kwargs", None)
            else ""
        )
        if _reasoning:
            self.memory.add_reasoning(_reasoning)
        StreamingMixin._detect_thought_loop(self, msg)
        meta: dict[str, Any] = {}
        if getattr(msg, "tool_calls", None):
            meta["tool_calls"] = msg.tool_calls
        self._conversation_tree.add_message("assistant", msg.content, metadata=meta)
        for tc in getattr(msg, "tool_calls", None) or []:
            StreamingMixin._remember_tool_call(self, tc)
        if records is not None:
            records.append(
                {
                    "type": "assistant",
                    "content": msg.content,
                    "tool_calls": getattr(msg, "tool_calls", None),
                    "ts": time.time(),
                }
            )

    def _detect_thought_loop(self, msg: AIMessage) -> None:
        """Feed one assistant output to the thought-loop detector."""
        if not getattr(self, "_thought_detector", None):
            return
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        if not content or len(content) <= 20:
            return
        if not self._thought_detector.record_output(content):
            return
        _, reason, should_terminate = self._thought_detector.should_break()
        if should_terminate:
            self._thought_loop_terminated = True
            logger.warning("Thought loop detected and terminated: %s", reason)
        else:
            logger.info("Thought loop detected, injecting break: %s", reason)

    def _remember_tool_call(self, tc: dict[str, Any]) -> None:
        """Record one tool call and cache its args for the matching ToolMessage."""
        self.memory.add_tool_call(
            tool_name=tc.get("name", "unknown"),
            input_args=tc.get("args", {}),
        )
        _tc_id = tc.get("id")
        if not _tc_id:
            return
        if not hasattr(self, "_pending_tool_inputs"):
            self._pending_tool_inputs = {}
        self._pending_tool_inputs[_tc_id] = tc.get("args", {})

    def _record_tool_message(
        self, msg: ToolMessage, records: list[dict[str, Any]] | None
    ) -> None:
        """Persist one tool result and honour the break-after-tool flag."""
        self.memory.add_message("tool", msg.content)
        self._conversation_tree.add_message(
            "tool",
            msg.content,
            metadata={
                "tool_call_id": msg.tool_call_id,
                "name": getattr(msg, "name", None),
            },
        )
        if records is not None:
            records.append(
                {
                    "type": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                    "name": getattr(msg, "name", None),
                    "ts": time.time(),
                }
            )
        StreamingMixin._track_tool_result(self, msg)
        if self._break_after_tool:
            self._break_flag = True

    def _track_tool_result(self, msg: ToolMessage) -> None:
        """Push the tool result into session_state, tolerating any tracker failure."""
        try:
            _tool_input = getattr(self, "_pending_tool_inputs", {}).pop(
                msg.tool_call_id, {}
            )
            self._session_state.add_tool_result(
                {
                    "tool_name": getattr(msg, "name", None) or "unknown",
                    "content": msg.content,
                    "success": not (
                        isinstance(msg.content, str) and msg.content.startswith("Error")
                    ),
                    "tool_input": _tool_input,
                }
            )
        except Exception:  # 防御: 工具结果追踪失败则忽略
            logger.debug("session_state tool tracking failed", exc_info=True)

    def _publish_cache_stats(
        self, msgs: list[Any], turn_span: Any, thread_id: str, pet: Any
    ) -> None:
        """Publish prompt-cache stats and pet status for the finished turn."""
        cache_stats = self._extract_cache_stats(msgs)
        if not cache_stats:
            return
        self._last_cache_stats = cache_stats
        turn_span.metadata.update(cache_stats)
        # token/cost 计数走 UsageCallback.on_llm_end (挂在每个 ChatOpenAI 上),
        # 这里不再重复调 track_llm_usage — 修复 Bug 2 后两条路径都读得到
        # token_usage, 同一调用会双计数. pet/span 是 streaming 独有, 保留.
        pet.publish(
            PetMood.SUCCESS,
            "Turn complete",
            {"thread_id": thread_id, **cache_stats},
        )

    def _extract_usage_tokens(self) -> dict[str, int]:
        """Pull token usage from the last LLM call's cache_stats."""
        stats = self._last_cache_stats or {}
        usage: dict[str, int] = {}
        for key in (
            "input_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
        ):
            if key in stats:
                usage[key] = int(stats[key] or 0)
            elif f"usage_{key}" in stats:
                usage[key] = int(stats[f"usage_{key}"] or 0)
        return usage

    async def _check_loop_interrupt(self, thread_id: str) -> dict[str, Any] | None:
        """Check for user intervention between stream states.

        Returns None when there's no intervention, or a dict describing
        the cancel/modify action.
        """
        try:
            mgr = get_interrupt_manager()
        except Exception:  # 防御: 中断管理器失效则跳过检查
            logger.debug("best-effort op failed", exc_info=True)
            return None
        await mgr.wait_if_paused(thread_id)
        evt = mgr.check_interrupt(thread_id)
        if evt is None:
            return None
        if evt.type == "cancel":
            return {"cancelled": True, "reason": evt.message or "cancelled by user"}
        if evt.type == "modify":
            try:
                self.memory.add_message("user", f"[modified] {evt.message}")
                self._conversation_tree.add_message("user", f"[modified] {evt.message}")
            except Exception:  # 防御: 用户输入保存失败仅告警
                logger.warning("Failed to save user input to memory", exc_info=True)
            return {"modified": True, "message": evt.message}
        return None

    def _compact_thresholds(self) -> tuple[int, int]:
        """按模型成本模式算压缩阈值 + 告警阈值."""
        # G34b/成本自适应: 按模型长上下文成本模式决定主动压缩阈值, 而非一刀切 60%.
        # linear(KDA)/hybrid(SALA) 线性注意力骨架便宜, 过早压缩 = 用信息换本不必省的
        # 算力 → 推迟; quadratic(旧全注意力) 维持既往 60%. 见 models.registry.compact_threshold_for.
        model_name = (
            getattr(self.model, "model_name", None)
            or getattr(self.model, "model", "")
            or ""
        )
        threshold = compact_threshold_for(str(model_name))
        # 告警阈值 = 压缩阈值 - 10 (仅 log, 不动作), 禁止 < 0.
        return threshold, max(0, threshold - 10)

    def _compact_precheck(
        self,
        usage: dict[str, int],
        warn_pct: int,
        threshold: int,
        ccap_trimmed: int,
        turn_span: Any,
    ) -> dict[str, Any] | None:
        """判断是否该压缩; 该压则返回 before 用量, 否则 None."""
        if not any(usage.values()):
            # P1①: 无 token usage → 无法可靠计算上下文百分比, 跳过 token-%
            # compaction/summary, 但上面 P0 的消息数兜底已执行. 不再是纯 return None
            # (旧逻辑在无 usage 时连 G34 修剪都不做 → checkpointer 无限膨胀).
            if ccap_trimmed > 0:
                turn_span.metadata["compact_p0_count_cap"] = True
            return None

        before = calculate_context_usage(usage, self._model_context_window)

        # 告警 (log only, no action) — used 落 [warn_pct, threshold)
        if before["used"] >= warn_pct and before["used"] < threshold:
            logger.info(
                "Context usage %d%%, approaching compaction threshold %d%%",
                before["used"],
                threshold,
            )
            return None

        if before["used"] <= threshold:
            return None

        logger.info(
            "Context usage %d%%, triggering auto-compact (threshold %d%%)",
            before["used"],
            threshold,
        )
        return before

    async def _maybe_auto_compact(
        self,
        final_state: dict[str, Any],
        turn_span: Any,
        thread_id: str,
        *,
        graph: Any = None,
        config: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """Trigger PRE_COMPACT hook + promote session summary when context > 60%.

        50% = warning (log only), 60% = trigger compaction.
        Previous 70% was too late — a single large tool result could
        push from 60% to 95%+ in one turn, bypassing compaction.

        G34: 当 ``HUGINN_COMPACT_STRATEGY`` 含 ``trim`` 且 graph + config +
        checkpointer 都在时, 用 ``RemoveMessage`` 真修剪 checkpointer 持久化
        状态. 之前 compact_messages 只修 inputs["messages"] 临时 list, 历史
        在 checkpointer 里无限累积 → 1.30 GB checkpoint (报告 17 维度 3 差距 3).

        Returns ``{"before_pct": int, "after_pct": int}`` if compaction ran, else None.
        """
        if self._model_context_window <= 0:
            return None

        threshold, warn_pct = self._compact_thresholds()

        # P0: 无条件 checkpointer 消息数上限兜底 — 不依赖 token usage / 上下文
        # 百分比. 长程任务消息条数可能无界增长 (即便每条 token 很小, 总量也常压
        # 在 token 预算内 → 纯 G34 不触发). 这里按条数硬上限收敛, 保证即使
        # compaction 门控失效 (如 LLM 不回 usage) 也有兜底.
        ccap_trimmed = await self._enforce_checkpointer_count_cap(
            graph, config, turn_span
        )

        usage = self._extract_usage_tokens()
        before = self._compact_precheck(
            usage, warn_pct, threshold, ccap_trimmed, turn_span
        )
        if before is None:
            return None

        after_pct = await self._compact_apply(
            final_state, before, usage, turn_span, thread_id, graph, config
        )
        return {"before_pct": before["used"], "after_pct": after_pct}

    async def _compact_apply(
        self,
        final_state: dict[str, Any],
        before: dict[str, Any],
        usage: dict[str, int],
        turn_span: Any,
        thread_id: str,
        graph: Any,
        config: dict[str, Any] | None,
    ) -> int:
        """执行压缩动作 + 落 span/事件, 返回压缩后百分比."""
        await self._run_online_compact_gate(final_state, before, turn_span)
        await self._run_pre_compact_hooks(usage, before, thread_id)
        await self._trim_checkpointer_state(final_state, graph, config, turn_span)
        after_pct = self._estimate_after_compact_pct()
        self._apply_belief_entropy_adaptive()

        logger.info("Context compacted (%d%% -> %d%%)", before["used"], after_pct)
        turn_span.metadata["compact_before_pct"] = before["used"]
        turn_span.metadata["compact_after_pct"] = after_pct
        await self._publish_compact_events(before["used"], after_pct, thread_id)
        return after_pct

    async def _run_online_compact_gate(
        self, final_state: dict[str, Any], before: dict[str, Any], turn_span: Any
    ) -> None:
        """SoL-Pi OnlineCompact 接线(④, 非侵入 fail-closed).

        压缩前先做"候选驱动 + 经济/证据密封门控"。若能从状态里读到计划步骤, 只把
        已验证+已密封步骤当可压候选, 未验证步骤标记 preserve_evidence 保留; 门控
        决策记入 turn_span 供审计。纯 advisory: 异常/无步骤列表都不改变既有修剪行为。
        """
        try:
            plan_steps = _extract_plan_steps_for_compact(final_state)
            if not plan_steps:
                return
            from huginn.autoloop.online_compact import gate_compaction

            g = gate_compaction(
                plan_steps,
                window_pct=float(before["used"]),
                cache_write_read_ratio=float(
                    getattr(
                        getattr(self, "_session_state", None),
                        "cache_write_read_ratio",
                        12.5,
                    )
                    or 12.5
                ),
            )
            turn_span.metadata["compact_candidates"] = g["compact_candidates"]
            turn_span.metadata["compact_preserve_evidence"] = g["preserve_evidence"]
            if g["should_compact"]:
                logger.info(
                    "online_compact: fold %d verified+sealed steps, preserve %d under-evidence",
                    len(g["compact_candidates"]),
                    len(g["preserve_evidence"]),
                )
        except Exception:  # noqa: BLE001 — 门控是 advisory, 失败不拖垮压缩主路径
            logger.debug("online_compact gating skipped (advisory)", exc_info=True)

    async def _run_pre_compact_hooks(
        self, usage: dict[str, int], before: dict[str, Any], thread_id: str
    ) -> None:
        """Fire PRE_COMPACT and promote the session summary before trimming."""
        pre_ctx = HookContext(
            tool_name="context_compact",
            metadata={
                "before_pct": before["used"],
                "usage": usage,
                "thread_id": thread_id,
            },
        )
        try:
            await self.hook_manager.trigger(PRE_COMPACT, pre_ctx)
        except Exception:  # 防御: 压缩前置钩子失败仅告警
            logger.warning("PRE_COMPACT hook raised", exc_info=True)

        try:
            summary = self.memory.promote_session_summary()
            if summary:
                self._conversation_summary = (
                    f"{self._conversation_summary}\n{summary}".strip()
                    if self._conversation_summary
                    else summary
                )
        except Exception:  # 防御: 会话摘要提升失败仅告警
            logger.warning("promote_session_summary failed", exc_info=True)

    async def _trim_checkpointer_state(
        self,
        final_state: dict[str, Any],
        graph: Any,
        config: dict[str, Any] | None,
        turn_span: Any,
    ) -> None:
        """G34: 真修剪 checkpointer 持久化状态.

        compact_messages 只修 inputs 临时 list, checkpointer 里的历史从不被修 →
        checkpoint 文件无限膨胀. 用 LangGraph 官方 RemoveMessage + update_state 删
        旧消息. 不引入新框架, 跟 lean "参考设计不引入编译器" 一个套路.
        """
        strategy = self._compact_strategy()
        if not (
            "trim" in strategy
            and graph is not None
            and config is not None
            and self.checkpointer is not None
        ):
            return
        try:
            removed = await self._trim_checkpointer_messages(
                final_state,
                graph,
                config,
                max_messages=self._checkpointer_max_messages(),
            )
            if removed > 0:
                logger.info("checkpointer trimmed %d old messages (G34)", removed)
                turn_span.metadata["checkpointer_trimmed"] = removed
        except Exception:  # 防御: checkpoint修剪失败仅告警
            logger.warning("checkpointer trim failed (G34)", exc_info=True)

    def _compact_strategy(self) -> list[str]:
        """Parse ``HUGINN_COMPACT_STRATEGY`` into an ordered strategy list."""
        strategy = (
            os.environ.get("HUGINN_COMPACT_STRATEGY", "trim,summarize")
            .lower()
            .split(",")
        )
        return [s.strip() for s in strategy if s.strip()]

    def _estimate_after_compact_pct(self) -> int:
        """Re-estimate context usage from prompt + tools + summary after compaction."""
        try:
            after_tokens = (
                count_tokens(self.system_prompt)
                + count_tokens(self._get_tool_description_text())
                + count_tokens(self._conversation_summary)
            )
            after = calculate_context_usage(
                {"input_tokens": after_tokens},
                self._model_context_window,
            )
            return after["used"]
        except Exception:  # 防御: 压缩后token统计失败置零
            logger.debug("best-effort op failed", exc_info=True)
            return 0

    def _apply_belief_entropy_adaptive(self) -> None:
        """Belief Entropy adaptive: adjust next-round compaction params."""
        try:
            from huginn.utils.belief_entropy import get_belief_entropy

            be = get_belief_entropy()
            last = getattr(be, "_last_result", None)
            if last is None:
                return
            if last.adaptive_keep_last_n is not None:
                self._adaptive_keep_last_n = max(
                    2,
                    getattr(self, "_adaptive_keep_last_n", 6)
                    + last.adaptive_keep_last_n,
                )
            if last.adaptive_budget_ratio is not None:
                base_budget = getattr(self, "_adaptive_budget_ratio", 1.0)
                self._adaptive_budget_ratio = max(
                    0.5, min(2.0, base_budget * last.adaptive_budget_ratio)
                )
            if last.h_belief < be.config.threshold_high:
                return
            logger.warning(
                "high belief entropy (%.3f) after compaction, "
                "promoting extra memory to long-term",
                last.h_belief,
            )
            try:
                self.memory.promote_session_summary(tier="long")
            except Exception:  # 防御: 长时记忆提升失败仅告警
                logger.warning("memory promote_session_summary failed", exc_info=True)
        except Exception:  # 防御: 自适应压缩失败则跳过
            logger.warning("adaptive compaction skipped", exc_info=True)

    async def _publish_compact_events(
        self, before_pct: int, after_pct: int, thread_id: str
    ) -> None:
        """统一事件总线: compact → 内部EventBus + POST_COMPACT hook + PetBus."""
        from huginn.events.unified_bus import get_unified_bus

        _ubus = get_unified_bus(self)
        await _ubus.publish_pet_mood(
            PetMood.SUCCESS,
            f"Context compacted ({before_pct}% -> {after_pct}%)",
            {"thread_id": thread_id},
        )
        await _ubus.publish_compact(before_pct, after_pct, thread_id)

    def _checkpointer_max_messages(self) -> int | None:
        """P0: 返回 checkpointer 消息数硬上限 (env 可配). None = 不启用.

        默认 120 条, 覆盖长程对话消息条目无界累积的场景. 设为 0 或负数表示关闭,
        仅保留原 token-budget 修剪 (保持与旧行为 100% 兼容).
        """
        raw = os.environ.get("HUGINN_CHECKPOINTER_MAX_MESSAGES", "120")
        try:
            v = int(raw)
        except (TypeError, ValueError):
            return 120
        if v <= 0:
            return None
        return v

    async def _enforce_checkpointer_count_cap(
        self,
        graph: Any,
        config: dict[str, Any] | None,
        turn_span: Any,
    ) -> int:
        """P0: 无条件按消息数上限修剪 checkpointer.

        与 G34 的 token-budget 修剪解耦, 不依赖 usage/上下文百分比. 只要
        graph+config+checkpointer 齐备、策略含 trim、且消息数超过上限, 就收敛.
        幂等 — 不超过上限或超过上限但退场 (root/keep_last) 已保护无剩余可丢时返回 0.
        """
        if (
            graph is None
            or config is None
            or getattr(self, "checkpointer", None) is None
        ):
            return 0
        max_msgs = self._checkpointer_max_messages()
        if max_msgs is None:
            return 0
        strategy = (
            os.environ.get("HUGINN_COMPACT_STRATEGY", "trim,summarize")
            .lower()
            .replace(" ", "")
        )
        if "trim" not in strategy.split(","):
            return 0
        try:
            removed = await self._trim_checkpointer_messages(
                None, graph, config, max_messages=max_msgs
            )
            if removed > 0:
                logger.info(
                    "checkpointer count-cap trimmed %d msgs (P0, cap=%d)",
                    removed,
                    max_msgs,
                )
                turn_span.metadata["checkpointer_p0_trimmed"] = removed
            return removed
        except Exception:  # 防御: 消息数封顶修剪失败仅告警
            logger.warning("checkpointer count-cap trim failed (P0)", exc_info=True)
            return 0

    async def _trim_checkpointer_messages(
        self,
        final_state: dict[str, Any],
        graph: Any,
        config: dict[str, Any],
        *,
        max_messages: int | None = None,
    ) -> int:
        """G34: 真修剪 checkpointer 持久化的 messages state.

        从 ``graph.get_state(config)`` 取现有 messages (带 ID), 用与
        ``compact_messages`` 同款的 drop-oldest 逻辑算要删几条, 再用
        ``RemoveMessage`` + ``update_state`` 批量删. root 标记优先读消息
        metadata (is_root/root), 无标记时回退到按位置 (前 keep_root_n 条).

        ``max_messages`` (P0): 可选的消息数硬上限, 不依赖 token usage 的兜底.
        Returns 实际删除的消息数.
        """
        if self.context_budget_tokens <= 0 and max_messages is None:
            return 0

        msgs = await _tcm_snapshot_messages(graph, config)
        if not msgs:
            return 0

        drop_indices = _tcm_plan(
            msgs,
            _message_token_counts(msgs),
            self.context_budget_tokens,
            max_messages,
        )
        if not drop_indices:
            return 0

        drop_ids = _drop_message_ids(msgs, drop_indices)
        if not drop_ids:
            return 0

        await _tcm_apply(graph, config, drop_ids)
        return len(drop_ids)

    async def _maybe_inject_synthetic_continue(
        self,
        final_state: dict[str, Any],
        thread_id: str,
    ) -> None:
        """Inject synthetic Continue message after compaction or tool boundary."""
        try:
            messages = final_state.get("messages", []) if final_state else []
            if not messages:
                return

            ended_at_tool = isinstance(messages[-1], ToolMessage)

            pipeline_block = _pipeline_context_block()
            prov_block, dag_block = _provenance_context_blocks()
            task_block = _task_state_context_block(self)

            if (
                not pipeline_block
                and not task_block
                and not ended_at_tool
                and not dag_block
            ):
                return

            synthetic = HumanMessage(
                content="\n\n".join(
                    _synthetic_continue_parts(
                        pipeline_block, prov_block, dag_block, task_block
                    )
                )
            )

            if hasattr(self, "_pending_synthetic_messages"):
                self._pending_synthetic_messages.append(synthetic)
            else:
                self._pending_synthetic_messages = [synthetic]

            logger.info(
                "Synthetic Continue injected (tool_boundary=%s, has_pipeline=%s)",
                ended_at_tool,
                bool(pipeline_block),
            )
        except Exception:  # 防御: 合成续接注入失败则忽略
            logger.debug("synthetic continue injection skipped", exc_info=True)

    async def _maybe_inject_proactive_suggestion(self) -> None:
        """Check pipeline state after each turn, inject suggestions when ready."""
        try:
            from huginn.provenance.pipeline import get_pipeline

            pipeline = get_pipeline()
            suggestions = pipeline._latest
            if not suggestions:
                entry = pipeline._latest_entry()
                if entry is not None:
                    suggestions = pipeline.suggest_next(
                        entry.produced_by, entry.parameters, {}
                    )
            if not suggestions:
                return
            ready = [s for s in suggestions if s.prerequisite_met]
            if not ready:
                return

            parts = ["[Pipeline Suggestion] Based on current progress:"]
            for s in ready[:3]:
                parts.append(f"  - [{s.stage.value}] {s.tool_hint}: {s.description}")
            parts.append(
                "Consider proceeding with one of these steps, "
                "or explain why a different approach is needed."
            )
            msg = HumanMessage(content="\n".join(parts))
            if hasattr(self, "_pending_synthetic_messages"):
                self._pending_synthetic_messages.append(msg)
            else:
                self._pending_synthetic_messages = [msg]
            logger.info(
                "Proactive suggestion injected: %d ready steps",
                len(ready),
            )
        except Exception:  # 防御: 主动建议注入失败则忽略
            logger.debug("proactive suggestion skipped", exc_info=True)

    # ── The main chat loop ────────────────────────────────────────

    async def chat(
        self,
        message: str,
        thread_id: str = "default",
        image_path: str | None = None,
        budget_override: BudgetSpec | None = None,
        include_history: bool | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a message to the Agent and stream responses.

        Stores messages in session memory and tracks tool calls for auto-promotion
        to long-term memory. *image_path* routes through the vision fallback chain;
        *budget_override* (BudgetSpec) temporarily overrides max_tool_calls /
        recursion_limit for this turn; *include_history*=False skips pulling
        ConversationTree history (Step 3 retry uses a fresh thread with all
        necessary state already injected into the prompt).
        """
        # 超长纯文本落盘, prompt 里只留"已保存"提示 (全文由 file_read_tool 读).
        from huginn.tools.paste_offload import maybe_offload_pasted_text

        message, _ = maybe_offload_pasted_text(message)

        state: dict[str, Any] = {}

        # ponytail: CodeAct 早返回 — 不走 langgraph / vision / cognitive engine,
        # 直接进 code_act_loop. CodeAct 模式下 LLM 输出 Python 代码块替代 JSON
        # tool_call, 工具作为 namespace 函数注入. 连续 3 次代码异常自动降级回
        # tool_call (走下面的原逻辑). 见 huginn/agent/code_act_loop.py.
        if getattr(self, "mode", "tool_call") == "code_act":
            async for ev in self._chat_code_act(state, message, thread_id):
                yield ev
            if not state.get("degraded"):
                return

        async for event in self._chat_preamble(
            state, message, thread_id, image_path, include_history
        ):
            yield event
        if state.get("stop"):
            return

        turn = self._chat_make_turn(state, thread_id, budget_override)
        async for event in self._chat_streaming_turn(turn):
            yield event

    async def _chat_code_act(self, state: dict[str, Any], message: str, thread_id: str):
        """CodeAct 早返回路径. state['degraded'] 标记是否降级继续."""
        from huginn.agent.code_act_loop import run_code_act_turn

        degraded = False
        async for ev in run_code_act_turn(self, message, thread_id):
            if ev.get("type") == "code_act_degraded":
                degraded = True
                break
            yield ev
        state["degraded"] = degraded
        if degraded:
            # 降级路径: 切回 tool_call 并继续执行下面的原 chat 逻辑
            self.mode = "tool_call"
            logger.warning("CodeAct degraded to tool_call after repeated code errors")

    async def _chat_preamble(
        self,
        state: dict[str, Any],
        message: str,
        thread_id: str,
        image_path: str | None,
        include_history: bool | None,
    ):
        """turn 前置阶段: 会话上下文 / mode banner / 澄清 / 视觉路由 / 隐私扫描."""
        set_thread_id(thread_id)
        set_user_message(message)
        # ponytail: 不写 self.thread_id / self._current_user_message 实例属性,
        # 并发 chat() 调用会互相覆盖. contextvars 已隔离每协程副本, core.py
        # 的 _build_graph 用 get_user_message() 读, 无竞争.

        # P4 Task-Dynamic Tool Router: task 变化时 refresh tool_filter.
        self._refresh_task_router(message)

        # OAK 启发: ConversationTree 通电 — 把每条 user/ai 消息写进树
        self._record_user_message_in_tree(message, thread_id)

        # ── Mode banner: 告诉前端当前 agent 工作模式 (端到端通信) ──
        # exec_mode: tool_call / code_act; user_mode: chat / research / plan
        yield {
            "type": "mode_banner",
            "exec_mode": self.mode,
            "user_mode": self.get_mode(),
            "flags": ["plan_mode"] if self.is_plan_mode() else [],
            # OAK 启发: trace_id 贯穿, 前端按 trace 聚合事件
            "trace_id": thread_id,
        }

        clarification_event = self._clarification_event(message)
        if clarification_event is not None:
            yield clarification_event
            # 不 return — yield 完继续走 agent loop, 用户可以选择回答或忽略

        async for event in self._chat_route_and_scrub(
            state, message, thread_id, image_path
        ):
            yield event
        if state.get("stop"):
            return
        state["include_history"] = include_history

    async def _chat_route_and_scrub(
        self,
        state: dict[str, Any],
        message: str,
        thread_id: str,
        image_path: str | None,
    ):
        """事件总线启动 → 视觉路由 → 隐私扫描 → turn 副作用."""
        # 统一事件总线: 桥接 HookManager + 内部EventBus + 插件EventBus + PetBus
        ubus = await self._begin_cognitive_turn(message, thread_id)

        # Vision routing
        message, vision_content, cv_hints = await self._resolve_vision_route(
            message, image_path
        )

        # Privacy scan on the raw user message.
        message, privacy_block = self._scrub_user_message(message)
        if privacy_block is not None:
            yield privacy_block
            state["stop"] = True
            return

        await self._start_turn_side_effects(message, thread_id, ubus)

        state.update(
            message=message,
            ubus=ubus,
            vision_content=vision_content,
            cv_hints=cv_hints,
        )

    def _chat_make_turn(
        self,
        state: dict[str, Any],
        thread_id: str,
        budget_override: Any,
    ) -> _TurnStream:
        """构造本轮可变状态容器 (TPS/TTFT 计时 + completion capture)."""
        # TPS / TTFT 实时监控: t0=turn 起点, t_first_token=首个 chunk 时间.
        # 在 turn_span 作用域外初始化, finally 块里算 tps.
        # ponytail: chunk_chars/4 ≈ tokens (latin). 升级: 用 response_metadata.usage.output_tokens 校准.
        tps_t0 = time.monotonic()
        # wire-level completion capture: 收集 prompt/response/tool_call/tool_result
        # 给 red_team 提供结构化输入 + 未来 RL 训练留数据.
        # 借鉴 Polar: 不在 harness 里手写 logger, 在 stream 层统一抓.
        # ponytail: 只覆盖 langgraph 路径 (chat 主流程). CodeAct/plan-mode 直调
        # 漏掉, 升级路径: 在 model.ainvoke 处再包一层 (monkeypatch BaseChatModel).
        completion_records: list[dict[str, Any]] = []
        capture_turn_id = f"{thread_id}_{int(time.time() * 1000)}"
        return _TurnStream(
            graph=None,
            inputs={},
            config={},
            use_sync_stream=False,
            turn_span=None,
            thread_id=thread_id,
            pet=get_pet_bus(),
            completion_records=completion_records,
            ubus=state["ubus"],
            message=state["message"],
            primary_idle=0.0,
            tps_t0=tps_t0,
            include_history=state["include_history"],
            vision_content=state["vision_content"],
            cv_hints=state["cv_hints"],
            capture_turn_id=capture_turn_id,
            budget_override=budget_override,
        )

    async def _chat_streaming_turn(self, turn: _TurnStream):
        """在 telemetry span 内跑完整一轮: 建 prompt → compact → runtime → stream."""
        with self._telemetry_collector.span(
            "agent_turn", thread_id=turn.thread_id
        ) as turn_span:
            turn.turn_span = turn_span
            turn.graph = self.build_graph()

            # 统一事件总线: ON_MESSAGE_RECEIVED
            await turn.ubus.publish_message_received(
                turn.thread_id, turn.message[:2000]
            )

            prompt_guidance, clarify_event = await self._await_prompt_submit(
                turn.message, turn.thread_id
            )
            if clarify_event is not None:
                yield clarify_event
                return

            messages = await self._chat_build_messages(turn, prompt_guidance)
            turn.inputs = {"messages": messages}

            await self._compact_turn_inputs(turn.inputs, turn.ubus)
            runtime = self._prepare_turn_runtime(
                turn.budget_override, turn.graph, turn.inputs, turn.thread_id
            )
            turn.config = runtime["config"]
            turn.use_sync_stream = runtime["use_sync_stream"]

            # 统一事件总线: ON_LLM_REQUEST + ON_BEFORE_MESSAGE_SENT
            await turn.ubus.publish_llm_request(turn.thread_id, len(messages))
            await turn.ubus.publish_before_message_sent(turn.thread_id, len(messages))

            # 主流空闲阈值 (随 thinking 档位放宽). 提到循环外, 使降级告警能如实
            # 报告"实际用的阈值", 而不是模块默认常量 _STREAM_IDLE_TIMEOUT.
            turn.primary_idle = _thinking_stream_idle()

            try:
                async for event in self._stream_turn(turn):
                    yield event
            finally:
                await self._finish_turn(
                    turn, turn.completion_records, turn.capture_turn_id
                )

    async def _chat_build_messages(self, turn: _TurnStream, prompt_guidance: Any):
        """构造本轮 messages: 历史 + 视觉 + 隐私守卫 graph 重建 + 纪律提醒."""
        messages = await self._build_turn_messages(
            turn.message,
            include_history=turn.include_history,
            vision_content=turn.vision_content,
            cv_hints=turn.cv_hints,
            prompt_guidance=prompt_guidance,
        )
        # 隐私守卫可能切到本地模型, 需要重建 graph.
        messages, turn.graph = self._apply_privacy_guard(messages, turn.graph)

        # M2: 事件驱动认知纪律守护 — event 档位下, 若对话尾部出现工具失败
        # 等纪律偏离, 发送前即时注入一条紧凑提醒 (平时不占常驻 prompt).
        return self._inject_discipline_reminder(messages)

    # ── chat() split helpers ──────────────────────────────────────

    def _refresh_task_router(self, message: str) -> None:
        """P4 Task-Dynamic Tool Router: refresh the tool filter when the task changes.

        flag 关时 set_current_task 内部直接 return (无 op). ponytail: 只在 task
        变化时 refresh, 避免每次 chat 都重建 graph cache.
        """
        from huginn.feature_flags import FeatureFlags

        if not FeatureFlags.shared().is_enabled("task_tool_router"):
            return
        prev_task = self._last_routed_task
        self.set_current_task(message)
        if message != prev_task:
            self.refresh_tools_from_registry()
            self._last_routed_task = message

    def _record_user_message_in_tree(self, message: str, thread_id: str) -> None:
        """OAK 启发: 把用户消息写进 ConversationTree (研究历史是树而非线性序列)."""
        try:
            self._conversation_tree.add_message(
                role="user",
                content=message,
                metadata={"thread_id": thread_id, "trace_id": thread_id},
            )
        except Exception:  # 防御: 对话树写用户消息失败则跳过
            logger.debug("ConversationTree add_message (user) skipped", exc_info=True)

    def _clarification_event(self, message: str) -> dict[str, Any] | None:
        """模糊意图捕捉 + 主动 questioning; 返回要 yield 的事件 (无则 None)."""
        # 1. capture_intuition: 检测直觉/类比信号, 命中则静默存 long tier
        try:
            self.memory.capture_intuition(message)
        except Exception:  # 防御: 直觉捕捉失败则跳过
            logger.debug("intuition capture skipped", exc_info=True)

        from huginn.interaction.clarification import should_ask_clarification

        # AgentMessage 是 dataclass 不是 dict, 直接取 .content 属性
        session_msgs = [
            {"content": m.content if isinstance(m.content, str) else str(m.content)}
            for m in (self.memory.session.messages or [])[-20:]
        ]
        clarification = should_ask_clarification(message, session_msgs)
        if clarification is None:
            return None
        return {
            "type": "clarification_request",
            "reason": clarification["reason"],
            "suggestion": clarification["suggestion"],
            "raw": clarification.get("raw", ""),
            "material": clarification.get("material"),
        }

    async def _begin_cognitive_turn(self, message: str, thread_id: str) -> Any:
        """统一事件总线 + 认知状态机推进; 返回本轮使用的 unified bus."""
        from huginn.events.unified_bus import get_unified_bus

        ubus = get_unified_bus(self)
        if self._turn_count == 0:
            self._init_session_continuity()
            await ubus.publish_session_start(thread_id, message)

        from huginn.cognitive_engine import TransitionSignal

        if self._turn_count == 0:
            self._csm.transition(TransitionSignal("user_goal", {"goal": message}))
        else:
            self._csm.transition(TransitionSignal("new_question", {"message": message}))

        self._session_state._cognitive_prompt = self._csm.get_attention_prompt()
        self._session_state._tool_preferences = self._csm.get_tool_preference()
        self._session_state.l1_coordinates = self._csm.l1_coordinates
        self._session_state.user_goals_history.append(message[:200])
        self._session_state.turns_count += 1
        self._session_state.clear_turn_results()
        return ubus

    async def _resolve_vision_route(
        self, message: str, image_path: str | None
    ) -> tuple[str, list[dict] | None, str | None]:
        """Route an attached image; returns (message, vision_content, cv_hints)."""
        from huginn.vision.router import VisionRoute, VisionRouter

        if not image_path:
            return message, None, None

        # 从 server_core 取共享单例, 避免每个 agent 实例各自加载一遍 ML 模型
        visual_encoder = getattr(self, "_visual_encoder", None)
        image_index = getattr(self, "_image_index", None)
        if visual_encoder is None or image_index is None:
            try:
                from huginn.server_core import get_image_index, get_visual_encoder

                visual_encoder = visual_encoder or get_visual_encoder()
                image_index = image_index or get_image_index()
            except Exception:  # 防御: 视觉组件获取失败则不启用
                logger.debug("visual_encoder/image_index 注入失败", exc_info=True)
        router = VisionRouter(visual_encoder=visual_encoder, image_index=image_index)
        model_name = getattr(self.model, "model", None) or getattr(
            self.model, "model_name", ""
        )
        route = router.route(model_name, message or image_path)

        if route == VisionRoute.BOTH:
            vision_content, cv_hints = router.coordinate(
                message, image_path, model_name
            )
            return message, vision_content, cv_hints
        if route != VisionRoute.CV_TOOLS:
            return message, None, None

        message, handled = await self._route_cv_tools(message, image_path, router)
        if not handled:
            cv_ctx = router.build_context(image_path)
            message = f"{cv_ctx}\n\n{message}"
        return message, None, None

    async def _route_cv_tools(
        self, message: str, image_path: str, router: Any
    ) -> tuple[str, bool]:
        """Current model can't see images: local decoder first, then delegation."""
        decoded: str | None = None
        try:
            from huginn.vision.local_decoder import decode_image

            # 把用户问题当 focus hint 传给视觉解码器, 让多模态模型回答
            # "当前这一步想知道什么", 而不是泛泛描述整张图.
            decoded = await asyncio.to_thread(decode_image, image_path, message or None)
        except Exception:  # 防御: 本地视觉解码不可用则回落
            logger.debug("local vision decoder unavailable", exc_info=True)
        if decoded:
            source = os.environ.get("HUGINN_LOCAL_VISION_MODEL", "qwen2.5-vl")
            return (
                f"[视觉描述 (本地解码 {source})]\n{decoded}\n\n"
                f"[用户问题]\n{message or '请根据上述视觉描述进行分析.'}"
            ), True

        team = getattr(self, "_team_ref", None)
        if team is None:
            return message, False
        return await self._delegate_vision(team, message, image_path)

    async def _delegate_vision(
        self, team: Any, message: str, image_path: str
    ) -> tuple[str, bool]:
        """Fall back to a VISION team member when no local decoder is available."""
        try:
            from huginn.agents.team import TeamRole

            vision_member = team.members.get(TeamRole.VISION)
            if vision_member is None or not vision_member.caps.vision:
                return message, False
            vision_task = (
                "请描述这张图片的内容, 重点关注与材料科学相关的"
                "信息 (如显微结构、晶体形貌、谱图特征等).\n"
                f"图片路径: {image_path}"
            )
            if message:
                vision_task += f"\n用户附加说明: {message}"
            # 把 image_path 注入 ctx, 由 team._run_member 透传给 VISION member
            # 的 agent.chat(image_path=...), 让多模态模型真正收到图像.
            deleg_ctx: dict[str, Any] = {"original_task": message or ""}
            deleg_ctx["image_path"] = image_path
            description = await team._delegate(
                TeamRole.VISION, vision_task, deleg_ctx, []
            )
        except Exception:  # 防御: 跨agent视觉委托失败仅告警
            logger.warning("Cross-agent vision delegation failed", exc_info=True)
            return message, False
        return (
            f"[视觉描述 (由 {vision_member.model_name} 提供)]\n{description}\n\n"
            f"[用户问题]\n{message or '请根据上述视觉描述进行分析.'}"
        ), True

    def _scrub_user_message(self, message: str) -> tuple[str, dict[str, Any] | None]:
        """Block or redact secrets/PII; returns (message, block_event)."""
        if self.privacy_block_on_secrets:
            found = scan_for_secrets(message)
            if found:
                labels = ", ".join(m.label for m in found)
                return message, {
                    "messages": [
                        HumanMessage(content=message),
                        AIMessage(
                            content=(
                                "I can't send this message because it may contain "
                                f"sensitive data: {labels}. Please remove the "
                                "secrets and try again."
                            )
                        ),
                    ]
                }
        if self.privacy_redact_secrets:
            message = redact_secrets(message)
        try:
            from huginn.privacy.scanner import SecretScanner

            message = SecretScanner().redact_pii(message)
        except Exception:  # 防御: PII扫描不可用则跳过脱敏
            logger.debug("PII scanner unavailable, skipping redaction", exc_info=True)
        return message, None

    async def _start_turn_side_effects(
        self, message: str, thread_id: str, ubus: Any
    ) -> None:
        """Memory/tree writes, pet mood, telemetry wiring, rate limit, metrics."""
        self.memory.add_message("user", message)
        self._conversation_tree.add_message("user", message)

        # 统一事件总线: pet mood → PetBus + 内部EventBus
        await ubus.publish_pet_mood(
            PetMood.THINKING, "Thinking...", {"thread_id": thread_id}
        )

        from huginn.telemetry import set_telemetry_collector

        set_telemetry_collector(self._telemetry_collector)

        from huginn.security.rate_limiter import get_rate_limiter

        get_rate_limiter().reset_turn(thread_id=thread_id)

        # Wire Prometheus turn counter
        try:
            from huginn.routes.metrics import track_agent_turn

            track_agent_turn(thread_id)
        except Exception:  # 防御: 指标计数不可用则忽略
            logger.debug("Prometheus turn counter unavailable", exc_info=True)

    async def _await_prompt_submit(
        self, message: str, thread_id: str
    ) -> tuple[Any, dict[str, Any] | None]:
        """Run the USER_PROMPT_SUBMIT hook; returns (prompt_guidance, clarify_event)."""
        prompt_ctx = HookContext(
            tool_name="user_prompt",
            metadata={
                "user_message": message,
                "thread_id": thread_id,
                "available_tools": self._tool_names_for_validation(),
            },
        )
        try:
            prompt_ctx = await self.hook_manager.trigger(USER_PROMPT_SUBMIT, prompt_ctx)
        except Exception:  # 防御: 提交钩子失败则继续主流程
            logger.warning("USER_PROMPT_SUBMIT hook raised", exc_info=True)

        clarify_questions = prompt_ctx.metadata.get("clarify_questions")
        if not clarify_questions:
            return prompt_ctx.metadata.get("prompt_guidance"), None

        q_text = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(clarify_questions))
        clarify_content = (
            f"Please answer the following questions to clarify intent:\n{q_text}"
        )
        self.memory.add_message("assistant", clarify_content)
        return prompt_ctx.metadata.get("prompt_guidance"), {
            "messages": [AIMessage(content=clarify_content)],
            "clarify_questions": clarify_questions,
            "needs_clarification": True,
        }

    async def _build_turn_messages(
        self,
        message: str,
        include_history: bool | None,
        vision_content: list[dict] | None,
        cv_hints: str | None,
        prompt_guidance: Any,
    ) -> list[Any]:
        """Assemble the message list sent to the graph for this turn."""
        # G33: 同步检索 (memory.recall + kb.search) 之前直接在协程里跑会卡住
        # event loop 几百毫秒. 用 to_thread 丢线程池, 不再阻塞.
        memory_text = await asyncio.to_thread(self._build_memory_text, query=message)
        kb_text = await asyncio.to_thread(self._build_kb_text, query=message)
        messages = self._build_input_messages(
            message,
            memory_text=memory_text,
            kb_text=kb_text,
            session_state=self._session_state,
            include_history=include_history,
        )

        if vision_content is not None:
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    msg.content = vision_content
                    break
        if cv_hints:
            messages.insert(-1, SystemMessage(content=cv_hints, id="ctx_cv_hints"))
        if prompt_guidance:
            guidance_text = (
                "\n\n".join(prompt_guidance)
                if isinstance(prompt_guidance, list)
                else prompt_guidance
            )
            messages.insert(-1, SystemMessage(content=guidance_text, id="ctx_guidance"))

        self._inject_style_directive(messages)
        self._inject_pending_synthetic(messages)
        self._strip_dangling_or_log(messages)
        return messages

    def _inject_style_directive(self, messages: list[Any]) -> None:
        if self.style_learner is None:
            return
        try:
            profile = self.style_learner.get_profile()
            if profile.confidence <= 0.3:
                return
            directive = self.style_learner.get_style_directive()
            if directive:
                messages.insert(-1, SystemMessage(content=directive, id="ctx_style"))
        except Exception:  # 防御: 风格指令注入失败则跳过
            logger.warning("style directive injection failed", exc_info=True)

    def _inject_pending_synthetic(self, messages: list[Any]) -> None:
        synthetic_msgs = getattr(self, "_pending_synthetic_messages", None)
        if not synthetic_msgs:
            return
        messages.extend(synthetic_msgs)
        self._pending_synthetic_messages = []
        logger.info("Injected %d synthetic Continue messages", len(synthetic_msgs))

    @staticmethod
    def _strip_dangling_or_log(messages: list[Any]) -> None:
        """C1: 发送前配对校验 — 剥掉 dangling tool_calls 防 400 invalid_request."""
        n_dangling = _strip_dangling_tool_calls(messages)
        if n_dangling:
            logger.warning(
                "C1: stripped %d dangling tool_calls from %d messages "
                "(prevents 400 invalid_request)",
                n_dangling,
                len(messages),
            )

    def _apply_privacy_guard(
        self, messages: list[Any], graph: Any
    ) -> tuple[list[Any], Any]:
        """Apply PrivacyGuard policy; may switch to a local model and rebuild graph."""
        try:
            from huginn.privacy_guard import PrivacyGuard

            guard = PrivacyGuard.shared()
            if not guard.should_send_to_cloud():
                graph = self._apply_local_only_policy(guard, graph)
                return messages, graph
            if not self._has_forced_local_content(messages, guard):
                return guard.redact_messages_for_cloud(messages), graph
            # proactive: tagged-local / ephemeral data → switch to local model
            # instead of just redacting
            local = self._find_local_model()
            if local is None:
                return guard.redact_messages_for_cloud(messages), graph
            logger.info("privacy: proactive local routing (sensitive data detected)")
            return messages, self._switch_to_local_model(graph, local)
        except RuntimeError:
            raise
        except Exception:  # 防御: 隐私守卫失败则原样发送
            logger.warning("PrivacyGuard hook failed", exc_info=True)
            return messages, graph

    def _apply_local_only_policy(self, guard: Any, graph: Any) -> Any:
        """local_only mode: switch model or raise (no silent cloud leak)."""
        if not guard.should_use_local(self._detect_provider()):
            return graph
        local = self._find_local_model()
        if local is None:
            raise RuntimeError(
                "PrivacyGuard is in local_only mode but no local model found. "
                "Configure ollama/vllm/llama.cpp, or set privacy "
                "level='off'/'redact' to allow cloud."
            )
        logger.info("local_only: switched to local model %s", type(local).__name__)
        return self._switch_to_local_model(graph, local)

    def _switch_to_local_model(self, graph: Any, local: Any) -> Any:
        self.model = local
        self._agent_graph = None
        return self.build_graph()

    @staticmethod
    def _has_forced_local_content(messages: list[Any], guard: Any) -> bool:
        for msg in messages:
            content = getattr(msg, "content", "")
            if isinstance(content, str) and guard.should_force_local(content):
                return True
        return False

    @staticmethod
    def _inject_discipline_reminder(messages: list[Any]) -> list[Any]:
        """M2: 事件驱动认知纪律守护 (event 档位下注入紧凑提醒)."""
        try:
            from huginn.cognitive_discipline import inject_discipline_reminder

            return inject_discipline_reminder(messages)
        except Exception:  # 防御: 纪律注入失败则跳过
            logger.debug("event-driven discipline inject skipped", exc_info=True)
            return messages

    async def _compact_turn_inputs(self, inputs: dict[str, Any], ubus: Any) -> None:
        """Compact history when a context budget is configured."""
        if self.context_budget_tokens <= 0:
            return
        # Meta-Trace (Oxelra 启发): .huginn/meta_trace.jsonl 存在时 raw messages
        # 可以更激进地 drop; ponytail: keep_last_n 减半 (min 1).
        trace_avail = self._meta_trace_available()
        keep_mult, root_mult = self._compaction_multipliers()
        summarizer = self._make_summarizer()
        if getattr(self, "_needs_compaction", False):
            # mode 切换 flag: CSM 进入 S3_SWITCH/S6_FEEDBACK 时标记需要 compaction.
            inputs["messages"] = await self._compact_mode_switch(
                inputs["messages"], summarizer, trace_avail, keep_mult, root_mult
            )
            return
        inputs["messages"] = await self._compact_routine(
            inputs["messages"], summarizer, trace_avail, keep_mult, root_mult
        )
        await self._report_context_usage(inputs["messages"], ubus)

    def _meta_trace_available(self) -> bool:
        try:
            return self._ctx_builder.meta_trace_available()
        except Exception:  # 防御: 元轨迹检查失败按无trace处理
            logger.debug("meta_trace_available check failed", exc_info=True)
            return False

    @staticmethod
    def _compaction_multipliers() -> tuple[float, float]:
        """M3: 按模型档位整体放宽/收紧 compaction 力度 (light 档保留更多原始)."""
        try:
            from huginn.plugins.model_tier import compaction_knobs

            knobs = compaction_knobs()
        except Exception:  # 防御: 压缩档位读取失败回退默认
            knobs = None
        return (
            knobs.keep_multiplier if knobs else 1.0,
            knobs.root_multiplier if knobs else 1.0,
        )

    async def _compact_mode_switch(
        self,
        messages: list[Any],
        summarizer: Any,
        trace_avail: bool,
        keep_mult: float,
        root_mult: float,
    ) -> list[Any]:
        """CSM flagged a mode switch — compact aggressively."""
        if summarizer is None:
            return self._drop_old_messages(messages, root_mult)
        self._needs_compaction = False  # 清 flag, 避免重复触发
        logger.info("mode-switch triggered compaction (CSM S3/S6)")
        keep_last_n = max(1, int((4 // (2 if trace_avail else 1)) * keep_mult))
        messages, self._conversation_summary = await summarize_compact_messages(
            messages,
            self.context_budget_tokens,
            keep_last_n=keep_last_n,
            summarizer=summarizer,
            existing_summary=self._build_compact_summary(),
            max_messages=getattr(self, "_context_max_messages", None),
        )
        return messages

    def _drop_old_messages(self, messages: list[Any], root_mult: float) -> list[Any]:
        """No summarizer available — plain token-budget drop keeping root messages."""
        return compact_messages(
            messages,
            self.context_budget_tokens,
            keep_last_n=1,
            # AV5: 默认 2 + markers, σ₂ 补丁下沉到生产路径
            keep_root_n=int(int(os.environ.get("HUGINN_KEEP_ROOT_N", "2")) * root_mult),
            root_content_markers=_load_root_markers(),
        )

    async def _compact_routine(
        self,
        messages: list[Any],
        summarizer: Any,
        trace_avail: bool,
        keep_mult: float,
        root_mult: float,
    ) -> list[Any]:
        """Regular per-turn compaction, adaptive to BeliefEntropy measurements."""
        if summarizer is None:
            return self._drop_old_messages(messages, root_mult)
        self._load_belief_entropy_knobs()
        keep_last_n = getattr(self, "_adaptive_keep_last_n", 4)
        # Meta-Trace: trace 存在时 keep_last_n 减半, trace 携带历史
        if trace_avail and keep_last_n > 1:
            keep_last_n = max(1, keep_last_n // 2)
        keep_last_n = max(1, int(keep_last_n * keep_mult))
        adaptive_budget = int(
            self.context_budget_tokens * getattr(self, "_adaptive_budget_ratio", 1.0)
        )
        messages, self._conversation_summary = await summarize_compact_messages(
            messages,
            adaptive_budget,
            keep_last_n=keep_last_n,
            summarizer=summarizer,
            existing_summary=self._build_compact_summary(),
            max_messages=getattr(self, "_context_max_messages", None),
        )
        return messages

    def _load_belief_entropy_knobs(self) -> None:
        """BeliefEntropy 闭环: 从上次 measure 结果读 adaptive 参数."""
        try:
            from huginn.utils.belief_entropy import get_belief_entropy

            last = getattr(get_belief_entropy(), "_last_result", None)
            if last is not None and last.adaptive_keep_last_n is not None:
                self._adaptive_keep_last_n = last.adaptive_keep_last_n
            if last is not None and last.adaptive_budget_ratio is not None:
                self._adaptive_budget_ratio = last.adaptive_budget_ratio
        except Exception:  # 防御: 熵自适应读取失败回退默认
            logger.debug("belief_entropy adaptive read failed", exc_info=True)

    async def _report_context_usage(self, messages: list[Any], ubus: Any) -> None:
        estimated = (
            count_tokens(self.system_prompt)
            + estimate_message_tokens(messages)
            + count_tokens(self._get_tool_description_text())
        )
        if estimated > self.context_budget_tokens:
            await ubus.publish_pet_mood(
                PetMood.ERROR,
                f"Context budget warning: ~{estimated} tokens",
                {"budget": self.context_budget_tokens},
            )
        if self._model_context_window > 0:
            logger.info(
                "context usage: %s",
                format_context_usage(
                    {"input_tokens": estimated}, self._model_context_window
                ),
            )

    def _prepare_turn_runtime(
        self,
        budget_override: BudgetSpec | None,
        graph: Any,
        inputs: dict[str, Any],
        thread_id: str,
    ) -> dict[str, Any]:
        """Wire per-turn budget / router / loop detectors; resolve stream mode."""
        max_calls, max_per_tool, rec_limit = self._turn_budget_limits(budget_override)
        config = {
            "configurable": {"thread_id": thread_id},
            "recursion_limit": rec_limit,
        }

        from huginn.agents.tool_budget import ToolCallBudget
        from huginn.agents.tool_call_router import ToolCallRouter

        turn_budget = ToolCallBudget(max_calls=max_calls, max_per_tool=max_per_tool)
        self._tool_adapter.set_budget(turn_budget)
        self._tool_adapter.set_router(ToolCallRouter(budget=turn_budget))
        self._inject_budget_message(inputs, turn_budget)
        self._install_loop_detectors()
        self._state_msg_offsets[thread_id] = self._checkpointer_message_offset(
            graph, config
        )
        return {
            "config": config,
            "use_sync_stream": self._use_sync_stream(self.checkpointer),
        }

    def _turn_budget_limits(
        self, budget_override: BudgetSpec | None
    ) -> tuple[int, int, int]:
        """(max_calls, max_per_tool, recursion_limit) for this turn."""
        if budget_override is not None:
            return (
                budget_override.max_calls,
                self._max_tool_calls_per_tool,
                budget_override.recursion_limit,
            )
        # tool 预算随 mode 联动: research/extreme 放宽 max_calls/per_tool
        # (科研长工具链反复迭代同工具, 默认 5 一次就卡死).
        max_calls, max_per_tool = self._effective_tool_budget()
        # P1-5: 取两者最大值, 既保证 max_tool_calls 空间, 又让 mode 联动真正生效.
        return (
            max_calls,
            max_per_tool,
            max(self._effective_recursion_limit(), max_calls * 5),
        )

    @staticmethod
    def _inject_budget_message(inputs: dict[str, Any], turn_budget: Any) -> None:
        """预算感知: 让 LLM 一开始就知道本轮预算上限与重置语义.

        放在 inputs 尾部、用固定 id 替换旧消息, 不碰缓存的 system_prompt,
        也不让预算消息在历史里层层累积 (路线图 P0-3 假耗尽).
        """
        try:
            status = turn_budget.status()
            budget_msg = SystemMessage(
                content=(
                    "## Tool Call Budget\n"
                    f"- This turn: {status['total_calls']}/{status['max_calls']} "
                    f"tool calls used/allowed (per-tool cap {status['max_per_tool']}).\n"
                    "- Budget is per turn and resets on the next turn. Exhausting it is "
                    "NOT game over — wrap up or switch tools, don't abandon the task."
                ),
                id="ctx_tool_budget",
            )
            msgs = [
                m
                for m in inputs.get("messages", [])
                if getattr(m, "id", None) != "ctx_tool_budget"
            ]
            inputs["messages"] = [*msgs, budget_msg]
        except Exception:  # 防御: 预算消息注入失败则跳过
            logger.debug("budget injection skipped", exc_info=True)

    def _install_loop_detectors(self) -> None:
        """AV5: 默认 skip — 两个 detector 在长任务里都会误判 (见下注释).

        升级: mode-aware detector, 区分 "同 tool 同输入" (真死循环) vs
        "同 tool 不同输入" (正常迭代). 统一走 FeatureFlags 而非直接读 env var,
        避免 benchmark 路径双配置冲突.
        """
        from huginn.agents.loop_detector import LoopDetector, ThoughtLoopDetector
        from huginn.feature_flags import FeatureFlags

        enabled = FeatureFlags.shared().is_enabled("loop_detector")
        self._tool_adapter.set_loop_detector(LoopDetector() if enabled else None)
        # F2: σ₈ 半修补全 — ThoughtLoopDetector 也要受同一个开关控制.
        self._thought_detector = ThoughtLoopDetector() if enabled else None
        self._thought_loop_terminated = False

    def _checkpointer_message_offset(self, graph: Any, config: dict[str, Any]) -> int:
        if self.checkpointer is None:
            return 0
        try:
            return len(graph.get_state(config).values.get("messages", []))
        except Exception:  # 防御: 状态偏移读取失败则从零开始
            logger.debug("checkpointer state fetch skipped", exc_info=True)
            return 0

    @staticmethod
    def _use_sync_stream(checkpointer: Any) -> bool:
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver

            return isinstance(checkpointer, SqliteSaver)
        except Exception:  # 防御: 流模式判定失败则用异步流
            logger.debug("best-effort op failed", exc_info=True)
            return False

    # ── streaming retry loop ──────────────────────────────────────

    async def _stream_turn(self, turn: _TurnStream) -> AsyncIterator[dict[str, Any]]:
        """Run the graph with retries, then hand off to post-stream processing."""
        max_retries = 3
        try:
            attempt = 0
            while attempt < max_retries:
                try:
                    if turn.use_sync_stream:
                        async for event in self._stream_sync(turn):
                            yield event
                    else:
                        async for event in self._stream_live(turn):
                            yield event
                    break
                except TimeoutError:
                    # A3: 流式空闲超时 — 降级到进度感知的流式收集, 不重试原流式.
                    async for event in self._stream_fallback(turn):
                        yield event
                    break
                except Exception as exc:
                    action, turn.graph = await self._handle_stream_failure(
                        exc, attempt, max_retries, turn
                    )
                    attempt = 0 if action == "switched" else attempt + 1

            if turn.final_state is not None:
                async for event in self._post_stream(turn):
                    yield event
        except Exception as exc:
            await turn.ubus.publish_pet_mood(
                PetMood.ERROR, f"Error: {exc}", {"thread_id": turn.thread_id}
            )
            raise

    async def _stream_sync(self, turn: _TurnStream) -> AsyncIterator[dict[str, Any]]:
        """Sync-checkpointer path: collect the blocking stream in a thread."""
        states = await asyncio.to_thread(
            lambda: list(
                turn.graph.stream(turn.inputs, turn.config, stream_mode="values")
            )
        )
        for state in states:
            # Sync stream doesn't get per-chunk messages, so extract reasoning
            # from the accumulated AIMessage before yielding state.
            msgs = state.get("messages", [])
            if msgs and hasattr(msgs[-1], "additional_kwargs"):
                reasoning = msgs[-1].additional_kwargs.get("reasoning_content", "")
                if reasoning and reasoning != turn.last_reasoning:
                    yield {"_reasoning": reasoning}
                    turn.last_reasoning = reasoning
            async for event in self._emit_state(state, turn):
                yield event
            if turn.stopped:
                return

    async def _stream_live(self, turn: _TurnStream) -> AsyncIterator[dict[str, Any]]:
        """Primary streaming path guarded by the idle / total watchdog."""
        # D7: 主流空闲阈值按**实时**剩余预算封顶 — 预算见底时不再维持一段
        # 注定越预算的阻塞, 尽早降级/失败. 无预算 (None) 时原样 = 旧行为.
        budget_left = live_budget_left()
        # D7 补漏 (run86): 仅封顶**空闲**挡不住涓流 — chunk 间隔始终 < idle 时
        # 会不断重置计时. 长程下再给整段流加**总时长**上限 = 实时剩余.
        total_timeout = (
            None
            if budget_left is None
            else _cap_timeout_by_budget(float("inf"), budget_left)
        )
        async for mode, data in _astream_with_watchdog(
            turn.graph.astream(
                turn.inputs, turn.config, stream_mode=["values", "messages"]
            ),
            idle_timeout=_cap_timeout_by_budget(turn.primary_idle, budget_left),
            total_timeout=total_timeout,
        ):
            if mode == "messages":
                async for event in self._emit_chunk(data, turn):
                    yield event
                continue
            async for event in self._emit_state(data, turn):
                yield event
            if turn.stopped:
                return

    @staticmethod
    async def _emit_chunk(
        data: Any, turn: _TurnStream
    ) -> AsyncIterator[dict[str, Any]]:
        """Token / reasoning events for one streamed chunk."""
        chunk, _meta = data
        if not type(chunk).__name__.startswith("AIMessage"):
            return
        text = ""
        if hasattr(chunk, "content") and isinstance(chunk.content, str):
            text = chunk.content
        if text:
            # TPS: 首次 chunk 记 TTFT, 每个 chunk 累加字符数.
            if turn.tps_t_first is None:
                turn.tps_t_first = time.monotonic()
                turn.turn_span.metadata["llm_ttft_ms"] = int(
                    (turn.tps_t_first - turn.tps_t0) * 1000
                )
            turn.tps_chunk_chars += len(text)
            yield {"_token": text}
        reasoning = ""
        if hasattr(chunk, "additional_kwargs"):
            reasoning = chunk.additional_kwargs.get("reasoning_content", "")
        if reasoning:
            yield {"_reasoning": reasoning}

    async def _emit_state(
        self, state: Any, turn: _TurnStream
    ) -> AsyncIterator[dict[str, Any]]:
        """Emit one graph state plus the loop-break / interrupt checks."""
        self._process_stream_state(
            state, turn.turn_span, turn.thread_id, turn.pet, turn.completion_records
        )
        turn.states_yielded += 1
        turn.final_state = state
        yield state

        if getattr(self, "_thought_loop_terminated", False):
            logger.warning("Terminating chat due to persistent thought loop")
            turn.stopped = True
            synth = await self._synthesize_closing_answer(state)
            if synth and isinstance(state, dict):
                state["messages"] = state.get("messages", []) + synth
            yield {"thought_loop_terminated": True, "state": state}
            return
        if self._break_flag:
            self._break_flag = False
            turn.stopped = True
            yield {"tool_break": True, "state": state}
            return
        interrupt = await self._check_loop_interrupt(turn.thread_id)
        if interrupt and interrupt.get("cancelled"):
            raise InterruptCancelled(interrupt.get("reason", ""))

    async def _stream_fallback(
        self, turn: _TurnStream
    ) -> AsyncIterator[dict[str, Any]]:
        """A3: 流式空闲超时后的进度感知降级收集.

        不用无进度的 ainvoke: 它只能猜一个固定时长, 长推理会误杀. 这里用
        "空闲超时"语义 — 只要模型还在产出就不中断; 外层再套 thinking 缩放的
        总兜底, 防无限 thinking 循环.
        """
        logger.warning(
            "stream watchdog timeout after %ds (states_yielded=%d), "
            "falling back to progress-aware stream collect",
            turn.primary_idle,
            turn.states_yielded,
        )
        turn.turn_span.metadata["stream_watchdog_timeout"] = True
        ainvoke_timeout, collect_idle, collect_inputs = _fallback_params(turn)
        if collect_inputs is None:
            turn.turn_span.metadata["stream_fallback_resumed"] = True
        turn.final_state = None
        final_state = await _fallback_collect(
            self._collect_states, turn, collect_inputs, collect_idle, ainvoke_timeout
        )
        self._process_stream_state(
            final_state,
            turn.turn_span,
            turn.thread_id,
            turn.pet,
            turn.completion_records,
        )
        turn.states_yielded += 1
        turn.final_state = final_state
        yield final_state

    @staticmethod
    async def _collect_states(
        graph: Any, config: dict[str, Any], inputs: Any, idle_timeout: float
    ) -> Any:
        state = None
        async for mode, data in _astream_with_watchdog(
            graph.astream(inputs, config, stream_mode=["values", "messages"]),
            idle_timeout=idle_timeout,
        ):
            if mode == "values":
                state = data
        return state

    async def _handle_stream_failure(
        self, exc: Exception, attempt: int, max_retries: int, turn: _TurnStream
    ) -> tuple[str, Any]:
        """Decide retry / fallback-model switch / re-raise.

        Returns ``("retry" | "switched", graph)``; raises when the error is fatal.
        """
        if isinstance(exc, InterruptCancelled):
            raise
        if turn.states_yielded > 0:
            raise
        retryable = (
            _is_rate_limit(exc)
            or _is_overloaded(exc)
            or _is_transient_network(exc)
            or _is_context_overflow(exc)
        )
        if not retryable or attempt == max_retries - 1:
            fallback = self._maybe_switch_fallback_model(exc, attempt)
            if fallback is None:
                raise
            await asyncio.sleep(_jitter(_exponential_backoff(1)))
            return "switched", self.build_graph()

        wait = self._retry_wait(exc, attempt)
        logger.warning(
            "Graph invocation failed (attempt %d/%d), retrying in %.2fs: %s",
            attempt + 1,
            max_retries,
            wait,
            exc,
        )
        # 对齐 Kimi Code stepRetryService 的 turn.step.retrying: 统一走
        # unified_bus.publish_step_retry, 让 retry 信号对 SSE / audit 可见.
        await self._publish_step_retry(
            exc, attempt, max_retries, wait, turn.states_yielded, turn.thread_id
        )
        await asyncio.sleep(wait)
        return "retry", turn.graph

    def _maybe_switch_fallback_model(self, exc: Exception, attempt: int) -> Any | None:
        """On 529-overload, switch the main model once and reset the attempt count."""
        if not _is_overloaded(exc) or self._main_fallback_override is not None:
            return None
        fallback = self._select_main_fallback_model()
        if fallback is None:
            return None
        logger.warning(
            "main chat 529 overloaded after %d attempts, "
            "switching to fallback model: %s",
            attempt + 1,
            getattr(fallback, "model", type(fallback).__name__),
        )
        self._main_fallback_override = fallback
        self._agent_graph = None
        return fallback

    @staticmethod
    def _retry_wait(exc: Exception, attempt: int) -> float:
        if _is_rate_limit(exc):
            wait = _get_retry_after(exc)
            if wait is not None:
                return _jitter(wait, jitter_ratio=0.1)
        return _jitter(_exponential_backoff(attempt + 1))

    async def _publish_step_retry(
        self,
        exc: Exception,
        attempt: int,
        max_retries: int,
        wait: float,
        states_yielded: int,
        thread_id: str,
    ) -> None:
        # ponytail: best-effort, publish 失败不影响主流程.
        try:
            from huginn.events.unified_bus import get_unified_bus

            await get_unified_bus(self).publish_step_retry(
                thread_id=thread_id,
                attempt=attempt + 1,
                max_attempts=max_retries,
                error_type=type(exc).__name__,
                error_message=str(exc)[:200],
                wait_ms=int(wait * 1000),
                states_yielded=states_yielded,
            )
        except Exception:  # 防御: 事件上报失败仅记 debug
            logger.debug("step retry event publish failed", exc_info=True)

    async def _post_stream(self, turn: _TurnStream) -> AsyncIterator[dict[str, Any]]:
        """Post-stream bookkeeping: plan sync, compaction, synthetic continue, phase."""
        final_state = turn.final_state
        # Sync any executing plan from PlanStore to session_state
        self._sync_plan_from_store()
        # reflection 已移到 finally 块, 确保 timeout/取消也能跑
        await self._maybe_inject_proactive_suggestion()
        # Auto-compact when context > 60% (50% = warning)
        compact_info = await self._maybe_auto_compact(
            final_state,
            turn.turn_span,
            turn.thread_id,
            graph=turn.graph,
            config=turn.config,
        )
        if compact_info:
            yield {"_compacted": compact_info}

        # Synthetic Continue after compaction / tool boundary
        await self._maybe_inject_synthetic_continue(final_state, turn.thread_id)
        # If synthetic messages were injected, signal auto-continue so the WS
        # handler can trigger another turn immediately.
        if getattr(self, "_pending_synthetic_messages", None):
            yield {"_auto_continue": True}

        ai_content = self._extract_last_ai_content(final_state)
        turn.final_ai_content = ai_content
        # 统一事件总线: ON_LLM_RESPONSE + ON_AFTER_MESSAGE_SENT
        await turn.ubus.publish_llm_response(turn.thread_id, ai_content[:2000])
        await turn.ubus.publish_after_message_sent(turn.thread_id, ai_content[:2000])
        if ai_content:
            self._record_assistant_message(ai_content, turn.thread_id, turn.message)

    def _record_assistant_message(
        self, ai_content: str, thread_id: str, message: str
    ) -> None:
        # OAK 启发: ai 消息写进 ConversationTree
        try:
            self._conversation_tree.add_message(
                role="assistant",
                content=ai_content,
                metadata={"thread_id": thread_id, "phase": self.phase},
            )
        except Exception:  # 防御: 对话树写AI消息失败则跳过
            logger.debug("ConversationTree add_message (ai) skipped", exc_info=True)
        self._observe_style(message, ai_content)
        phase_target = self._check_phase_transition(ai_content)
        if phase_target is not None:
            self.transition_phase(phase_target)
            logger.info("Phase auto-transitioned to %s", phase_target.value)

    def _observe_style(self, message: str, ai_content: str) -> None:
        if self.style_learner is None:
            return
        try:
            self.style_learner.observe(message, ai_content)
        except Exception:  # 防御: 风格学习观察失败仅告警
            logger.warning("style_learner.observe failed", exc_info=True)

    async def _finish_turn(
        self,
        turn: _TurnStream,
        completion_records: list[dict[str, Any]],
        capture_turn_id: str,
    ) -> None:
        """Always-run teardown: reflection, TPS, trajectory, STOP, snapshot."""
        self._run_turn_reflection()
        self._record_tps(turn)
        self._clear_turn_runtime()
        self._save_trajectory(turn.thread_id, turn.message)
        # wire-level completion dump: prompt/response/tool_call/tool_result.
        _dump_completion_records(completion_records, turn.thread_id, capture_turn_id)

        from huginn.events.unified_bus import get_unified_bus

        ubus = get_unified_bus(self)
        # STOP event: 一次调用桥接 STOP hook + SESSION_END hook + EventBus.
        await ubus.publish_stop(
            turn.thread_id, self.workspace, response=turn.final_ai_content
        )
        await ubus.publish_session_end(turn.thread_id, self._turn_count)
        self._persist_memory_after_turn(turn.thread_id)
        await ubus.publish_pet_mood(
            PetMood.IDLE, "Ready", {"thread_id": turn.thread_id}
        )
        self._turn_count += 1
        self._maybe_memory_maintenance(turn.pet)

    def _run_turn_reflection(self) -> None:
        """反思闭环: 即使 timeout/取消, 也要跑 reflection 让 evolution 记录失败."""
        import sys

        n_results = len(self._session_state.tool_results_this_turn)
        print(
            f"[FINALLY-REFLECT] tool_results_this_turn={n_results}",
            file=sys.stderr,
            flush=True,
        )
        try:
            if self._session_state.tool_results_this_turn:
                self._run_post_turn_reflection()
        except Exception:  # 防御: 收尾反思失败则忽略
            logger.debug("finally reflection failed", exc_info=True)

    def _record_tps(self, turn: _TurnStream) -> None:
        """TPS 收尾: chunk_chars/4 ≈ tokens (latin). 写 turn_span + Prometheus."""
        if turn.tps_t_first is None or turn.tps_chunk_chars <= 0:
            return
        elapsed = time.monotonic() - turn.tps_t_first
        if elapsed <= 0:
            return
        tps = (turn.tps_chunk_chars / 4.0) / elapsed
        turn.turn_span.metadata["llm_tps"] = round(tps, 1)
        turn.turn_span.metadata["llm_output_chars"] = turn.tps_chunk_chars
        try:
            from huginn.routes.metrics import track_llm_tps

            track_llm_tps(
                model=getattr(self.model, "model", "unknown"),
                ttft_ms=turn.turn_span.metadata.get("llm_ttft_ms", 0),
                tps=tps,
            )
        except Exception:  # 防御: 速度指标上报失败则忽略
            logger.debug("TPS prometheus publish failed", exc_info=True)

    def _clear_turn_runtime(self) -> None:
        self._tool_adapter.set_budget(None)
        self._tool_adapter.set_router(None)
        self._tool_adapter.set_loop_detector(None)
        if self._main_fallback_override is not None:
            self._main_fallback_override = None
            self._agent_graph = None

    def _save_trajectory(self, thread_id: str, message: str) -> None:
        try:
            from huginn.telemetry import save_trajectory

            traj_dir = self.workspace / HUGINN_DIR_NAME / "trajectories"
            traj_path = traj_dir / f"{thread_id}_{int(time.time())}.json"
            save_trajectory(
                self._telemetry_collector,
                traj_path,
                metadata={
                    "thread_id": thread_id,
                    "user_message": message[:200],
                    "turn_count": self._turn_count,
                },
            )
        except Exception:  # 防御: 轨迹落盘失败则忽略
            logger.debug("trajectory save failed", exc_info=True)

    def _persist_memory_after_turn(self, thread_id: str) -> None:
        """Research-mode promotion + session-state snapshot for the next session."""
        if self.is_research_mode():
            try:
                self.memory.promote_session_summary(tier="long")
            except Exception:  # 防御: 研究模式记忆提升失败则忽略
                logger.debug("research-mode memory promote failed", exc_info=True)
        try:
            self._session_state.l1_coordinates = self._csm.l1_coordinates
            snapshot = self._session_state.to_snapshot()
            csm_snap = self._csm.get_snapshot()
            snapshot["cognitive_state"] = csm_snap.get("state", "s0_blank")
            tags = ["session_snapshot"]
            l1 = csm_snap.get("l1_coordinates", "")
            if l1:
                tags.append(f"l1:{l1[:200]}")
            self.memory.longterm.store(
                content=(
                    f"Session snapshot: "
                    f"{snapshot.get('l1_coordinates', 'no coordinates')} | "
                    f"cognitive_state: {snapshot.get('cognitive_state', '?')}"
                ),
                category="conversation",
                tags=tags,
                source=f"session:{thread_id}",
                importance=0.6,
                tier="mid",
            )
        except Exception:  # 防御: 会话快照保存失败则忽略
            logger.debug("session snapshot save failed", exc_info=True)

    def _maybe_memory_maintenance(self, pet: Any) -> None:
        if not (
            self.memory_decay_enabled
            and self.memory_decay_interval_turns > 0
            and self._turn_count % self.memory_decay_interval_turns == 0
        ):
            return
        try:
            summary = self.memory.maintenance(
                prune_threshold=self.memory_decay_prune_threshold
            )
            pet.publish(PetMood.SUCCESS, "Memory maintenance", {"summary": summary})
        except Exception as exc:
            logger.warning("Memory maintenance failed: %s", exc, exc_info=True)

    async def _synthesize_closing_answer(self, final_state: Any) -> list[Any]:
        """若本轮结束却没有可用的助手文本答案 (循环终止 / 模型空响应 /
        纯工具轮), 用模型把已有对话合成一段收尾概要, 保证用户总能拿到有用回应.

        返回需要追加到 messages 的消息列表; 无需兜底时返回 [].
        纯防御式: 任何失败都静默降级为不追加, 不阻塞 turn.
        """
        try:
            msgs = (
                final_state.get("messages", [])
                if isinstance(final_state, dict)
                else (final_state or [])
            )
            if _has_assistant_text(msgs):
                return []

            model = self.select_model("agent")
            if model is None or not hasattr(model, "ainvoke"):
                return []

            transcript = _closing_transcript(msgs)
            if not transcript:
                return []

            text = await _invoke_closing_summary(model, transcript)
            if text:
                return [AIMessage(content=text)]
        except Exception:  # 防御: 收尾合成失败则返回空
            logger.debug("closing-answer synthesis skipped", exc_info=True)
        return []


# A3.4 self-check: watchdog 超时 + 正常透传 + thinking block 保护.
# 运行: python -m huginn.agent.streaming
if __name__ == "__main__":
    import sys as _sys

    if "--self-check-c1" in _sys.argv:
        _sys.exit(_c1_self_check())

    async def _test_watchdog():
        # 1. 正常流: chunks 及时到达, watchdog 不触发
        async def _fast_stream():
            for i in range(5):
                await asyncio.sleep(0.01)
                yield i

        results = []
        async for item in _astream_with_watchdog(_fast_stream(), idle_timeout=2.0):
            results.append(item)
        assert results == [0, 1, 2, 3, 4], f"watchdog mangled normal stream: {results}"

        # 2. 超时流: chunks 间隔超过 idle_timeout → asyncio.TimeoutError
        async def _slow_stream():
            yield "first"
            await asyncio.sleep(5.0)  # 远超 idle_timeout
            yield "second"

        timed_out = False
        try:
            async for _ in _astream_with_watchdog(_slow_stream(), idle_timeout=0.5):
                pass
        except TimeoutError:
            logger.debug("best-effort op failed", exc_info=True)
            timed_out = True
        assert timed_out, "watchdog failed to raise TimeoutError on idle stream"

        # 3. 空流: 立即结束, 不超时
        async def _empty_stream():
            return
            yield  # never reached, makes it an async generator

        empty_results = []
        async for item in _astream_with_watchdog(_empty_stream(), idle_timeout=1.0):
            empty_results.append(item)
        assert empty_results == [], (
            f"empty stream should yield nothing: {empty_results}"
        )

        # 4. 单 chunk 后超时: 第一个 chunk 透传, 第二个超时
        async def _one_then_slow():
            yield "fast"
            await asyncio.sleep(3.0)
            yield "slow"

        seen = []
        try:
            async for item in _astream_with_watchdog(
                _one_then_slow(), idle_timeout=0.5
            ):
                seen.append(item)
        except TimeoutError:
            logger.debug("best-effort op failed", exc_info=True)
        assert seen == ["fast"], f"first chunk not yielded before timeout: {seen}"

        # 5. D7 补漏: 总时长封顶挡涓流 (chunk 间隔 < idle, 整体超 total)
        async def _trickle():
            for i in range(100):
                await asyncio.sleep(0.02)
                yield i

        _trick_seen = []
        try:
            async for item in _astream_with_watchdog(
                _trickle(), idle_timeout=1.0, total_timeout=0.1
            ):
                _trick_seen.append(item)
        except TimeoutError:
            logger.debug("best-effort op failed", exc_info=True)
        assert 0 < len(_trick_seen) < 100, (
            f"total_timeout failed to cap trickle: {len(_trick_seen)}"
        )

        # 6. total_timeout=None → 完全等价旧行为 (全量透传)
        _all = []
        async for item in _astream_with_watchdog(
            _trickle(), idle_timeout=1.0, total_timeout=None
        ):
            _all.append(item)
            if len(_all) >= 5:
                break
        assert _all == [0, 1, 2, 3, 4], f"None total_timeout changed behaviour: {_all}"

    asyncio.run(_test_watchdog())

    # 5. _STREAM_IDLE_TIMEOUT 默认 60s, 可被 env 覆盖
    assert _STREAM_IDLE_TIMEOUT == 60.0, (
        f"default idle timeout should be 60, got {_STREAM_IDLE_TIMEOUT}"
    )

    print(
        "A3 self-check OK (watchdog timeout + passthrough + thinking block protection)"
    )
