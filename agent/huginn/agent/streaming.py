"""Chat streaming loop, phase management, and context compaction."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from huginn.context_manager import (
    calculate_context_usage,
    format_context_usage,
)
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


async def _astream_with_watchdog(
    aiter: AsyncIterator,
    idle_timeout: float = _STREAM_IDLE_TIMEOUT,
) -> AsyncIterator:
    """包装 async iterator, 空闲超时抛 asyncio.TimeoutError.

    每次取下一个 chunk 用 asyncio.wait_for 限时. 超时 → 上层捕获后走 ainvoke 降级.
    ponytail: 不在这里做降级, 只负责报时. 降级逻辑在 chat() 里, 因为需要 graph + inputs.
    """
    while True:
        try:
            item = await asyncio.wait_for(aiter.__anext__(), timeout=idle_timeout)
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
        AIMessage(content="p", tool_calls=[
            {"id": "tc3", "name": "f", "args": {}},
            {"id": "tc4", "name": "g", "args": {}},
        ]),
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


def _dump_completion_records(
    records: list[dict[str, Any]],
    thread_id: str,
    turn_id: str,
) -> Any:
    """把 wire-level completion records 落盘为 jsonl, 供 red_team / RL 训练消费.

    每条 record 是 ``_process_stream_state`` 抓的 prompt/response/tool_call/
    tool_result 结构化快照. 一个 turn 一个文件:
    ``<runtime_home>/completions/<thread_id>/<turn_id>.jsonl``.

    ponytail: 只 dump 非空 records, 失败静默 (返回 None). 升级路径: 加 prefix_merging
    (跨 turn 共享 prompt 前缀去重, 减少冗余存储) — 当前每 turn 独立文件, 不做合并.
    """
    if not records:
        return None
    try:
        import json

        from huginn.utils.runtime import get_runtime_home

        comp_dir = get_runtime_home() / "completions" / thread_id
        comp_dir.mkdir(parents=True, exist_ok=True)
        comp_path = comp_dir / f"{turn_id}.jsonl"
        with open(comp_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        return comp_path
    except Exception:
        logger.debug("completion dump failed", exc_info=True)
        return None


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
                    except Exception:
                        logger.debug(
                            "ConversationTree merge skipped for branch %s",
                            leaf, exc_info=True,
                        )
        except Exception:
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
            if any(
                k in llm_type or k in cls_name
                for k in ("ollama", "vllm", "local")
            ):
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
        for msg in new_msgs:
            if isinstance(msg, AIMessage):
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
                # Thought loop detection
                if hasattr(self, "_thought_detector") and self._thought_detector:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    if content and len(content) > 20:
                        is_loop = self._thought_detector.record_output(content)
                        if is_loop:
                            should_break, reason, should_terminate = (
                                self._thought_detector.should_break()
                            )
                            if should_terminate:
                                self._thought_loop_terminated = True
                                logger.warning(
                                    "Thought loop detected and terminated: %s",
                                    reason,
                                )
                            else:
                                logger.info(
                                    "Thought loop detected, injecting break: %s",
                                    reason,
                                )
                meta: dict[str, Any] = {}
                if getattr(msg, "tool_calls", None):
                    meta["tool_calls"] = msg.tool_calls
                self._conversation_tree.add_message(
                    "assistant", msg.content, metadata=meta
                )
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        self.memory.add_tool_call(
                            tool_name=tc.get("name", "unknown"),
                            input_args=tc.get("args", {}),
                        )
                        # 缓存 tool_input, 供 ToolMessage 的 add_tool_result 用
                        _tc_id = tc.get("id")
                        if _tc_id:
                            if not hasattr(self, "_pending_tool_inputs"):
                                self._pending_tool_inputs = {}
                            self._pending_tool_inputs[_tc_id] = tc.get("args", {})
                if records is not None:
                    records.append(
                        {
                            "type": "assistant",
                            "content": msg.content,
                            "tool_calls": getattr(msg, "tool_calls", None),
                            "ts": time.time(),
                        }
                    )
            elif isinstance(msg, ToolMessage):
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
                try:
                    _tool_input = getattr(self, "_pending_tool_inputs", {}).pop(
                        msg.tool_call_id, {}
                    )
                    self._session_state.add_tool_result(
                        {
                            "tool_name": getattr(msg, "name", None) or "unknown",
                            "content": msg.content,
                            "success": not (
                                isinstance(msg.content, str)
                                and msg.content.startswith("Error")
                            ),
                            "tool_input": _tool_input,
                        }
                    )
                except Exception:
                    logger.debug("session_state tool tracking failed", exc_info=True)
                if self._break_after_tool:
                    self._break_flag = True
        cache_stats = self._extract_cache_stats(msgs)
        if cache_stats:
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
        except Exception:
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
                self._conversation_tree.add_message(
                    "user", f"[modified] {evt.message}"
                )
            except Exception:
                logger.warning("Failed to save user input to memory", exc_info=True)
            return {"modified": True, "message": evt.message}
        return None

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

        usage = self._extract_usage_tokens()
        if not any(usage.values()):
            return None

        before = calculate_context_usage(usage, self._model_context_window)

        # 50% warning — log only, no action
        if before["used"] >= 50 and before["used"] < 60:
            logger.info(
                "Context usage %d%%, approaching compaction threshold",
                before["used"],
            )
            return None

        if before["used"] <= 60:
            return None

        logger.info(
            "Context usage %d%%, triggering auto-compact",
            before["used"],
        )

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
        except Exception:
            logger.warning("PRE_COMPACT hook raised", exc_info=True)

        try:
            summary = self.memory.promote_session_summary()
            if summary:
                self._conversation_summary = (
                    f"{self._conversation_summary}\n{summary}".strip()
                    if self._conversation_summary
                    else summary
                )
        except Exception:
            logger.warning("promote_session_summary failed", exc_info=True)

        # G34: 真修剪 checkpointer 持久化状态. compact_messages 只修 inputs 临时
        # list, checkpointer 里的历史从不被修 → checkpoint 文件无限膨胀. 用
        # LangGraph 官方 RemoveMessage + update_state 删旧消息. 不引入新框架,
        # 跟 lean "参考设计不引入编译器" 一个套路 — 用 langgraph 自带机制.
        strategy = os.environ.get(
            "HUGINN_COMPACT_STRATEGY", "trim,summarize"
        ).lower().split(",")
        strategy = [s.strip() for s in strategy if s.strip()]
        if (
            "trim" in strategy
            and graph is not None
            and config is not None
            and self.checkpointer is not None
        ):
            try:
                removed = await self._trim_checkpointer_messages(
                    final_state, graph, config
                )
                if removed > 0:
                    logger.info(
                        "checkpointer trimmed %d old messages (G34)", removed
                    )
                    turn_span.metadata["checkpointer_trimmed"] = removed
            except Exception:
                logger.warning(
                    "checkpointer trim failed (G34)", exc_info=True
                )

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
            after_pct = after["used"]
        except Exception:
            logger.debug("best-effort op failed", exc_info=True)
            after_pct = 0

        # Belief Entropy adaptive: adjust next-round params
        try:
            from huginn.utils.belief_entropy import get_belief_entropy
            be = get_belief_entropy()
            last = getattr(be, "_last_result", None)
            if last is not None:
                if last.adaptive_keep_last_n is not None:
                    self._adaptive_keep_last_n = max(2, (
                        getattr(self, "_adaptive_keep_last_n", 6)
                        + last.adaptive_keep_last_n
                    ))
                if last.adaptive_budget_ratio is not None:
                    base_budget = getattr(self, "_adaptive_budget_ratio", 1.0)
                    self._adaptive_budget_ratio = max(
                        0.5, min(2.0, base_budget * last.adaptive_budget_ratio)
                    )
                if last.h_belief >= be.config.threshold_high:
                    logger.warning(
                        "high belief entropy (%.3f) after compaction, "
                        "promoting extra memory to long-term",
                        last.h_belief,
                    )
                    try:
                        self.memory.promote_session_summary(tier="long")
                    except Exception:
                        logger.warning("memory promote_session_summary failed", exc_info=True)
        except Exception:
            logger.warning("adaptive compaction skipped", exc_info=True)

        logger.info(
            "Context compacted (%d%% -> %d%%)",
            before["used"],
            after_pct,
        )
        turn_span.metadata["compact_before_pct"] = before["used"]
        turn_span.metadata["compact_after_pct"] = after_pct
        get_pet_bus().publish(
            PetMood.SUCCESS,
            f"Context compacted ({before['used']}% -> {after_pct}%)",
            {"thread_id": thread_id},
        )
        # 统一事件总线: compact → 内部EventBus + PRE/POST_COMPACT hook
        from huginn.events.unified_bus import get_unified_bus
        _ubus = get_unified_bus(self)
        await _ubus.publish_compact(before["used"], after_pct, thread_id)
        return {"before_pct": before["used"], "after_pct": after_pct}

    async def _trim_checkpointer_messages(
        self,
        final_state: dict[str, Any],
        graph: Any,
        config: dict[str, Any],
    ) -> int:
        """G34: 真修剪 checkpointer 持久化的 messages state.

        从 ``graph.get_state(config)`` 取 checkpointer 现有 messages (带 ID),
        用与 ``compact_messages`` 同款的 drop-oldest 逻辑算要删几条, 再用
        LangGraph 官方 ``RemoveMessage`` + ``update_state`` 批量删.

        跟 lean "参考设计不引入编译器" 一个套路: 不引新框架, 只用 langgraph
        自带机制. ponytail: keep_last_n / keep_root_n 默认值与 compact_messages
        对齐. root 标记已升级为按消息 metadata (additional_kwargs/metadata 的
        is_root) 标记, 无 metadata 标记时回退到按位置 (前 keep_root_n 条).

        Returns 实际删除的消息数.
        """
        if self.context_budget_tokens <= 0:
            return 0

        # 取 checkpointer 现有 state (含 messages + IDs)
        snapshot = await asyncio.to_thread(graph.get_state, config)
        if snapshot is None or not snapshot.values:
            return 0
        msgs = snapshot.values.get("messages", [])
        if len(msgs) <= 4:
            return 0

        # 单条 msg 的 token 估算 (复用 utils.context 的 helper)
        from huginn.utils.context import _msg_content, _msg_role
        from huginn.utils.tokens import count_message_tokens

        per_msg_tokens = [
            count_message_tokens(_msg_content(m), _msg_role(m)) for m in msgs
        ]
        total = sum(per_msg_tokens)
        if total <= self.context_budget_tokens:
            return 0

        keep_last_n = 4
        # AV5: 默认保前 2 条 root (user 初始任务 + system checklist) — σ₂ compaction
        # 丢 Step 1 checklist 不只影响 benchmark, 生产路径长对话也会丢 system prompt.
        keep_root_n = int(os.environ.get("HUGINN_KEEP_ROOT_N", "2"))
        body_end = len(msgs) - keep_last_n

        # root 标记: 优先读消息 metadata / additional_kwargs 里的 is_root 标记
        # (调用方在 checklist / winner_plan 等关键 early-turn 消息上设置).
        # 没有任何 metadata 标记时, 回退到按位置 (前 keep_root_n 条) — 与历史
        # 行为完全一致, 不破坏现有调用路径. 有 metadata 标记时, 标记位覆盖位置,
        # root 消息可在任意位置 (不止开头) 被保护.
        root_indices: set[int] = set()
        for i, m in enumerate(msgs):
            if _is_root_message(m):
                root_indices.add(i)
        if not root_indices:
            # 回退: 无 metadata 标记 → 按位置保前 keep_root_n 条
            root_indices = set(range(min(keep_root_n, len(msgs))))

        # 可 drop 候选 = 尾部 keep_last_n 之外、且非 root 的消息, 从最旧开始丢
        droppable_order = [i for i in range(body_end) if i not in root_indices]
        if not droppable_order:
            return 0

        # 算要 drop 几条才能到 budget (从最旧开始扣 token)
        drop_indices: list[int] = []
        acc = total
        for i in droppable_order:
            if acc <= self.context_budget_tokens:
                break
            acc -= per_msg_tokens[i]
            drop_indices.append(i)

        if not drop_indices:
            return 0

        # 收集要 drop 的 message IDs — system 消息不删, 没 ID 的没法删
        from langchain_core.messages import RemoveMessage, SystemMessage

        drop_ids: list[str] = []
        for i in drop_indices:
            m = msgs[i]
            if isinstance(m, SystemMessage):
                continue
            mid = getattr(m, "id", None)
            if mid:
                drop_ids.append(mid)

        if not drop_ids:
            return 0

        removals = [RemoveMessage(id=mid) for mid in drop_ids]
        # update_state 是同步方法, 走 to_thread 不卡 event loop (G33 一致性)
        await asyncio.to_thread(
            graph.update_state, config, {"messages": removals}
        )
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

            last_msg = messages[-1] if messages else None
            ended_at_tool = isinstance(last_msg, ToolMessage)

            pipeline_block = ""
            try:
                from huginn.provenance.pipeline import SimulationPipeline
                pipeline = SimulationPipeline()
                pipeline_block = pipeline.to_context_block()
            except Exception:
                logger.debug("pipeline context block skipped", exc_info=True)

            prov_block = ""
            dag_block = ""
            try:
                from huginn.provenance.registry import ProvenanceRegistry
                reg = ProvenanceRegistry.shared()
                prov_block = reg.to_context_block()
                # P1-1: 压缩后注入 DAG mermaid, 防止 agent 丢文件产出关系全局视图
                from huginn.provenance import to_mermaid_for_context
                dag_block = to_mermaid_for_context(reg)
            except Exception:
                logger.debug("provenance context block skipped", exc_info=True)

            # Long-horizon task state — gives the agent a view of what it
            # has already done across the full conversation, not just the
            # current context window.
            task_block = ""
            try:
                from huginn.memory.task_state import get_tracker
                _tid = getattr(self, "thread_id", "") or ""
                if _tid:
                    task_block = get_tracker().context_block(_tid)
            except Exception:
                logger.debug("task state context block skipped", exc_info=True)

            if not pipeline_block and not task_block and not ended_at_tool and not dag_block:
                return

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
            synthetic = HumanMessage(content="\n\n".join(parts))

            if hasattr(self, '_pending_synthetic_messages'):
                self._pending_synthetic_messages.append(synthetic)
            else:
                self._pending_synthetic_messages = [synthetic]

            logger.info(
                "Synthetic Continue injected (tool_boundary=%s, has_pipeline=%s)",
                ended_at_tool, bool(pipeline_block),
            )
        except Exception:
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
                parts.append(
                    f"  - [{s.stage.value}] {s.tool_hint}: {s.description}"
                )
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
        except Exception:
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

        Stores messages in session memory and tracks tool calls for
        auto-promotion to long-term memory.

        If *image_path* is provided, the agent routes the image through
        the vision fallback chain.

        If *budget_override* is provided (BudgetSpec), temporarily
        overrides max_tool_calls / recursion_limit for this turn.
        PhaseManager.proposed_budget → Orchestrator → chat(budget_override=...).

        If *include_history* is False, skip pulling ConversationTree history
        into input messages. Used by Step 3 retry — fresh thread 的 prompt
        已结构化注入所有必要 state, 不需要 Step 2 的 1M+ tokens 历史累积.
        """
        # ponytail: CodeAct 早返回 — 不走 langgraph / vision / cognitive engine,
        # 直接进 code_act_loop. CodeAct 模式下 LLM 输出 Python 代码块替代 JSON
        # tool_call, 工具作为 namespace 函数注入. 连续 3 次代码异常自动降级回
        # tool_call (走下面的原逻辑). 见 huginn/agent/code_act_loop.py.
        if getattr(self, "mode", "tool_call") == "code_act":
            from huginn.agent.code_act_loop import run_code_act_turn

            degraded = False
            async for ev in run_code_act_turn(self, message, thread_id):
                if ev.get("type") == "code_act_degraded":
                    degraded = True
                    break
                yield ev
            if not degraded:
                return
            # 降级路径: 切回 tool_call 并继续执行下面的原 chat 逻辑
            self.mode = "tool_call"
            logger.warning(
                "CodeAct degraded to tool_call after repeated code errors"
            )

        set_thread_id(thread_id)
        set_user_message(message)
        # ponytail: 不写 self.thread_id / self._current_user_message 实例属性,
        # 并发 chat() 调用会互相覆盖. contextvars 已隔离每协程副本, core.py
        # 的 _build_graph 用 get_user_message() 读, 无竞争.

        # P4 Task-Dynamic Tool Router: task 变化时 refresh tool_filter.
        # env=0 时 set_current_task 内部直接 return (无 op). ponytail: 只在
        # task 变化时 refresh, 避免每次 chat 都重建 graph cache.
        if os.environ.get("HUGINN_TASK_TOOL_ROUTER", "0") == "1":
            prev_task = self._last_routed_task
            self.set_current_task(message)
            if message != prev_task:
                self.refresh_tools_from_registry()
                self._last_routed_task = message

        # OAK 启发: ConversationTree 通电 — 把每条 user/ai 消息写进树
        # phase 转移时 fork 出新分支, 让研究历史成为树而非线性序列
        try:
            self._conversation_tree.add_message(
                role="user", content=message,
                metadata={"thread_id": thread_id, "trace_id": thread_id},
            )
        except Exception:
            logger.debug("ConversationTree add_message (user) skipped", exc_info=True)

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

        # ── 人机协作: 模糊意图捕捉 + 主动 questioning ──
        # 两件事都在 agent loop 之前做, 不阻断主流程:
        # 1. capture_intuition: 检测直觉/类比信号, 命中则静默存 long tier
        # 2. should_ask_clarification: 检测模糊意图, 命中则 yield 事件让 UI 问
        # 用户 profile: "more questioning环节" + "capturing vague intuitions
        # without judgment". 这里是声明式触发, 不做语义判断.
        try:
            self.memory.capture_intuition(message)
        except Exception:
            logger.debug("intuition capture skipped", exc_info=True)

        from huginn.interaction.clarification import should_ask_clarification
        # AgentMessage 是 dataclass 不是 dict, 直接取 .content 属性
        session_msgs = [
            {"content": m.content if isinstance(m.content, str) else str(m.content)}
            for m in (self.memory.session.messages or [])[-20:]
        ]
        clarification = should_ask_clarification(message, session_msgs)
        if clarification is not None:
            yield {
                "type": "clarification_request",
                "reason": clarification["reason"],
                "suggestion": clarification["suggestion"],
                "raw": clarification.get("raw", ""),
                "material": clarification.get("material"),
            }
            # 不 return — yield 完继续走 agent loop, 用户可以选择回答或忽略

        # 统一事件总线: 桥接 HookManager + 内部EventBus + 插件EventBus + PetBus
        from huginn.events.unified_bus import get_unified_bus
        _ubus = get_unified_bus(self)

        if self._turn_count == 0:
            self._init_session_continuity()
            await _ubus.publish_session_start(thread_id, message)

        from huginn.cognitive_engine import TransitionSignal
        if self._turn_count == 0:
            self._csm.transition(TransitionSignal("user_goal", {"goal": message}))
        else:
            self._csm.transition(TransitionSignal("new_question", {"message": message}))

        self._session_state._cognitive_prompt = self._csm.get_attention_prompt()
        tool_pref = self._csm.get_tool_preference()
        self._session_state._tool_preferences = tool_pref
        self._session_state.l1_coordinates = self._csm.l1_coordinates
        self._session_state.user_goals_history.append(message[:200])

        self._session_state.turns_count += 1
        self._session_state.clear_turn_results()

        # Vision routing
        from huginn.vision.router import VisionRoute, VisionRouter

        _vision_route = VisionRoute.TEXT_ONLY
        _vision_content: list[dict] | None = None
        _cv_hints: str | None = None
        _vision_delegated: str | None = None  # description from vision member
        if image_path:
            # 从 server_core 取共享单例, 避免每个 agent 实例各自加载一遍 ML 模型
            _ve = getattr(self, "_visual_encoder", None)
            _ii = getattr(self, "_image_index", None)
            if _ve is None or _ii is None:
                try:
                    from huginn.server_core import get_image_index, get_visual_encoder

                    _ve = _ve or get_visual_encoder()
                    _ii = _ii or get_image_index()
                except Exception:
                    logger.debug("visual_encoder/image_index 注入失败", exc_info=True)
            _vr = VisionRouter(
                visual_encoder=_ve,
                image_index=_ii,
            )
            model_name = getattr(self.model, "model", None) or getattr(self.model, "model_name", "")
            _vision_route = _vr.route(model_name, message or image_path)
            if _vision_route == VisionRoute.BOTH:
                _vision_content, _cv_hints = _vr.coordinate(message, image_path, model_name)
            elif _vision_route == VisionRoute.CV_TOOLS:
                # Current model can't see images — check if team has a
                # VISION member (e.g., local multimodal model like
                # qwen2.5-vl) that can handle it instead.
                _team = getattr(self, "_team_ref", None)
                if _team is not None:
                    try:
                        from huginn.agents.team import TeamRole
                        vision_member = _team.members.get(TeamRole.VISION)
                        if vision_member and vision_member.caps.vision:
                            # Delegate image description to vision member
                            vision_task = (
                                f"请描述这张图片的内容, 重点关注与材料科学相关的"
                                f"信息 (如显微结构、晶体形貌、谱图特征等).\n"
                                f"图片路径: {image_path}"
                            )
                            if message:
                                vision_task += f"\n用户附加说明: {message}"
                            _vision_traces: list = []
                            # 把 image_path 注入 ctx, 由 team._run_member 透传给
                            # VISION member 的 agent.chat(image_path=...), 让多模态
                            # 模型真正收到图像而非只有文本路径.
                            _deleg_ctx: dict[str, Any] = {
                                "original_task": message or "",
                            }
                            if image_path is not None:
                                _deleg_ctx["image_path"] = image_path
                            _vision_delegated = await _team._delegate(
                                TeamRole.VISION, vision_task,
                                _deleg_ctx,
                                _vision_traces,
                            )
                            # Inject vision description as context for the
                            # text model, replacing the raw image
                            message = (
                                f"[视觉描述 (由 {vision_member.model_name} 提供)]\n"
                                f"{_vision_delegated}\n\n"
                                f"[用户问题]\n{message or '请根据上述视觉描述进行分析.'}"
                            )
                            _vision_route = VisionRoute.TEXT_ONLY
                    except Exception:
                        logger.warning(
                            "Cross-agent vision delegation failed",
                            exc_info=True,
                        )

                if _vision_route == VisionRoute.CV_TOOLS:
                    cv_ctx = _vr.build_context(image_path)
                    message = f"{cv_ctx}\n\n{message}"

        # Privacy scan on the raw user message.
        if self.privacy_block_on_secrets:
            found = scan_for_secrets(message)
            if found:
                labels = ", ".join(m.label for m in found)
                yield {
                    "messages": [
                        HumanMessage(content=message),
                        AIMessage(
                            content=f"I can't send this message because it may contain sensitive data: {labels}. Please remove the secrets and try again."
                        ),
                    ]
                }
                return

        if self.privacy_redact_secrets:
            message = redact_secrets(message)

        try:
            from huginn.privacy.scanner import SecretScanner
            scanner = SecretScanner()
            message = scanner.redact_pii(message)
        except Exception:
            logger.debug("PII scanner unavailable, skipping redaction", exc_info=True)

        self.memory.add_message("user", message)
        self._conversation_tree.add_message("user", message)

        pet = get_pet_bus()
        # 统一事件总线: pet mood → PetBus + 内部EventBus
        await _ubus.publish_pet_mood(PetMood.THINKING, "Thinking...", {"thread_id": thread_id})

        from huginn.telemetry import set_telemetry_collector
        set_telemetry_collector(self._telemetry_collector)

        from huginn.security.rate_limiter import get_rate_limiter
        get_rate_limiter().reset_turn(thread_id=thread_id)

        # Wire Prometheus turn counter
        try:
            from huginn.routes.metrics import track_agent_turn
            track_agent_turn(thread_id)
        except Exception:
            logger.debug("Prometheus turn counter unavailable", exc_info=True)

        # TPS / TTFT 实时监控: t0=turn 起点, t_first_token=首个 chunk 时间.
        # 在 turn_span 作用域外初始化, finally 块里算 tps.
        # ponytail: chunk_chars/4 ≈ tokens (latin). 升级: 用 response_metadata.usage.output_tokens 校准.
        _tps_t0 = time.monotonic()
        _tps_t_first: float | None = None
        _tps_chunk_chars = 0
        # wire-level completion capture: 收集 prompt/response/tool_call/tool_result
        # 给 red_team 提供结构化输入 + 未来 RL 训练留数据.
        # 借鉴 Polar: 不在 harness 里手写 logger, 在 stream 层统一抓.
        # ponytail: 只覆盖 langgraph 路径 (chat 主流程). CodeAct/plan-mode 直调
        # 漏掉, 升级路径: 在 model.ainvoke 处再包一层 (monkeypatch BaseChatModel).
        _completion_records: list[dict[str, Any]] = []
        _capture_turn_id = f"{thread_id}_{int(time.time() * 1000)}"

        with self._telemetry_collector.span(
            "agent_turn", thread_id=thread_id
        ) as turn_span:
            graph = self.build_graph()

            # 统一事件总线: ON_MESSAGE_RECEIVED
            from huginn.events.unified_bus import get_unified_bus
            _ubus = get_unified_bus(self)
            await _ubus.publish_message_received(thread_id, message[:2000])

            prompt_ctx = HookContext(
                tool_name="user_prompt",
                metadata={
                    "user_message": message,
                    "thread_id": thread_id,
                    "available_tools": self._tool_names_for_validation(),
                },
            )
            try:
                prompt_ctx = await self.hook_manager.trigger(
                    USER_PROMPT_SUBMIT, prompt_ctx
                )
            except Exception:
                logger.warning(
                    "USER_PROMPT_SUBMIT hook raised", exc_info=True
                )

            clarify_questions = prompt_ctx.metadata.get("clarify_questions")
            if clarify_questions:
                q_text = "\n".join(
                    f"{i + 1}. {q}" for i, q in enumerate(clarify_questions)
                )
                clarify_content = f"Please answer the following questions to clarify intent:\n{q_text}"
                self.memory.add_message("assistant", clarify_content)
                from langchain_core.messages import AIMessage as _AIMsg
                yield {
                    "messages": [_AIMsg(content=clarify_content)],
                    "clarify_questions": clarify_questions,
                    "needs_clarification": True,
                }
                return

            prompt_guidance = prompt_ctx.metadata.get("prompt_guidance")

            # G33: 同步检索 (memory.recall + kb.search) 之前直接在协程里跑,
            # 长查询会卡住整个 event loop 几百毫秒. 用 to_thread 丢线程池, 不再阻塞.
            memory_text = await asyncio.to_thread(self._build_memory_text, query=message)
            kb_text = await asyncio.to_thread(self._build_kb_text, query=message)
            messages = self._build_input_messages(
                message,
                memory_text=memory_text,
                kb_text=kb_text,
                session_state=self._session_state,
                include_history=include_history,
            )

            if _vision_content is not None:
                for msg in reversed(messages):
                    if isinstance(msg, HumanMessage):
                        msg.content = _vision_content
                        break

            if _cv_hints:
                messages.insert(-1, SystemMessage(content=_cv_hints, id="ctx_cv_hints"))
            if prompt_guidance:
                guidance_text = (
                    "\n\n".join(prompt_guidance)
                    if isinstance(prompt_guidance, list)
                    else prompt_guidance
                )
                messages.insert(-1, SystemMessage(content=guidance_text, id="ctx_guidance"))

            if self.style_learner is not None:
                try:
                    profile = self.style_learner.get_profile()
                    if profile.confidence > 0.3:
                        directive = self.style_learner.get_style_directive()
                        if directive:
                            messages.insert(-1, SystemMessage(content=directive, id="ctx_style"))
                except Exception:
                    logger.warning(
                        "style directive injection failed", exc_info=True
                    )

            synthetic_msgs = getattr(self, '_pending_synthetic_messages', None)
            if synthetic_msgs:
                messages.extend(synthetic_msgs)
                self._pending_synthetic_messages = []
                logger.info("Injected %d synthetic Continue messages", len(synthetic_msgs))

            # C1: 发送前配对校验 — 剥掉 dangling tool_calls 防 400 invalid_request.
            # context_builder 的 conversation_tree_to_messages 已剥, 但 _build_input_messages
            # 从 checkpointer 直读可能带 dangling. 这里是最后一道防线.
            # ponytail: 裁剪而非补占位 — 补占位会让 LLM 以为工具执行了但结果丢失,
            #   裁剪只保留已应答的 tool_calls, 语义更干净.
            #   ceiling: 裁掉后 LLM 看不到"我曾想调工具"的意图, 可能重试.
            #   升级路径: 补占位 ToolMessage(content="[context lost]") 保留意图.
            _n_dangling = _strip_dangling_tool_calls(messages)
            if _n_dangling:
                logger.warning(
                    "C1: stripped %d dangling tool_calls from %d messages "
                    "(prevents 400 invalid_request)", _n_dangling, len(messages),
                )

            try:
                from huginn.privacy_guard import PrivacyGuard

                _pg = PrivacyGuard.shared()
                _force_local = False
                if not _pg.should_send_to_cloud():
                    _prov = self._detect_provider()
                    if _pg.should_use_local(_prov):
                        _local = self._find_local_model()
                        if _local is None:
                            raise RuntimeError(
                                "PrivacyGuard is in local_only mode but no local model found. "
                                "Configure ollama/vllm/llama.cpp, or set privacy "
                                "level='off'/'redact' to allow cloud."
                            )
                        self.model = _local
                        self._agent_graph = None
                        graph = self.build_graph()
                        logger.info(
                            "local_only: switched to local model %s",
                            type(_local).__name__,
                        )
                else:
                    # proactive: if any message contains tagged-local or ephemeral data,
                    # switch to local model instead of just redacting
                    for _m in messages:
                        _c = getattr(_m, "content", "")
                        if isinstance(_c, str) and _pg.should_force_local(_c):
                            _force_local = True
                            break
                    if _force_local:
                        _local = self._find_local_model()
                        if _local is not None:
                            self.model = _local
                            self._agent_graph = None
                            graph = self.build_graph()
                            logger.info(
                                "privacy: proactive local routing (sensitive data detected)"
                            )
                        else:
                            # no local model, fall back to redact
                            messages = _pg.redact_messages_for_cloud(messages)
                    else:
                        messages = _pg.redact_messages_for_cloud(messages)
            except RuntimeError:
                raise
            except Exception:
                logger.warning(
                    "PrivacyGuard hook failed", exc_info=True
                )

            inputs = {"messages": messages}

            # Compact initial messages if a context budget is configured.
            # mode 切换 flag: CSM 进入 S3_SWITCH/S6_FEEDBACK 时标记需要 compaction.
            # ponytail: flag 模式避开 async/sync 边界. 升级: CSM 直接 emit 事件.
            #
            # Meta-Trace (Oxelra 启发): 当 .huginn/meta_trace.jsonl 存在时,
            # 结构化历史已经蒸馏进 prompt, raw messages 可以更激进地 drop.
            # ponytail: keep_last_n 减半 (min 1), 让 trace 替代 raw 携带信息.
            #   升级路径: 按 trace entry 数动态调, trace 长 → keep_last_n 更小.
            _trace_avail = False
            try:
                _trace_avail = self._ctx_builder.meta_trace_available()
            except Exception:
                logger.debug("meta_trace_available check failed", exc_info=True)
            _trace_kln_divisor = 2 if _trace_avail else 1

            if getattr(self, "_needs_compaction", False) and self.context_budget_tokens > 0:
                summarizer = self._make_summarizer()
                if summarizer is not None:
                    self._needs_compaction = False  # 清 flag, 避免重复触发
                    logger.info("mode-switch triggered compaction (CSM S3/S6)")
                    inputs["messages"], self._conversation_summary = (
                        await summarize_compact_messages(
                            inputs["messages"],
                            self.context_budget_tokens,
                            keep_last_n=max(1, 4 // _trace_kln_divisor),
                            summarizer=summarizer,
                            existing_summary=self._build_compact_summary(),
                            max_messages=getattr(self, "_context_max_messages", None),
                        )
                    )
                else:
                    inputs["messages"] = compact_messages(
                        inputs["messages"],
                        self.context_budget_tokens,
                        keep_last_n=1,
                        # AV5: 默认 2 + markers, σ₂ 补丁下沉到生产路径
                        keep_root_n=int(os.environ.get("HUGINN_KEEP_ROOT_N", "2")),
                        root_content_markers=_load_root_markers(),
                    )
            elif self.context_budget_tokens > 0:
                summarizer = self._make_summarizer()
                if summarizer is not None:
                    # BeliefEntropy 闭环: 从上次 measure 结果读 adaptive 参数.
                    # 之前断在 _last_result 存了但没人读, 导致自适应参数永远是默认值.
                    try:
                        from huginn.utils.belief_entropy import get_belief_entropy
                        be = get_belief_entropy()
                        last = getattr(be, "_last_result", None)
                        if last is not None and last.adaptive_keep_last_n is not None:
                            self._adaptive_keep_last_n = last.adaptive_keep_last_n
                        if last is not None and last.adaptive_budget_ratio is not None:
                            self._adaptive_budget_ratio = last.adaptive_budget_ratio
                    except Exception:
                        logger.debug("belief_entropy adaptive read failed", exc_info=True)
                    adaptive_kln = getattr(self, "_adaptive_keep_last_n", 4)
                    # Meta-Trace: trace 存在时 keep_last_n 减半, trace 携带历史
                    if _trace_avail and adaptive_kln > 1:
                        adaptive_kln = max(1, adaptive_kln // 2)
                    adaptive_budget = int(
                        self.context_budget_tokens
                        * getattr(self, "_adaptive_budget_ratio", 1.0)
                    )
                    inputs["messages"], self._conversation_summary = (
                        await summarize_compact_messages(
                            inputs["messages"],
                            adaptive_budget,
                            keep_last_n=adaptive_kln,
                            summarizer=summarizer,
                            existing_summary=self._build_compact_summary(),
                            max_messages=getattr(self, "_context_max_messages", None),
                        )
                    )
                else:
                    inputs["messages"] = compact_messages(
                        inputs["messages"],
                        self.context_budget_tokens,
                        keep_last_n=1,
                        # AV5: 默认 2 + markers, σ₂ 补丁下沉到生产路径
                        keep_root_n=int(os.environ.get("HUGINN_KEEP_ROOT_N", "2")),
                        root_content_markers=_load_root_markers(),
                    )
                estimated = (
                    count_tokens(self.system_prompt)
                    + estimate_message_tokens(inputs["messages"])
                    + count_tokens(self._get_tool_description_text())
                )
                if estimated > self.context_budget_tokens:
                    get_pet_bus().publish(
                        PetMood.ERROR,
                        f"Context budget warning: ~{estimated} tokens",
                        {"budget": self.context_budget_tokens},
                    )
                if self._model_context_window > 0:
                    logger.info(
                        "context usage: %s",
                        format_context_usage(
                            {"input_tokens": estimated},
                            self._model_context_window,
                        ),
                    )

            # langgraph recursion: 每个 tool call 约 2-3 次 (agent + tool + routing)
            # max_tool_calls=100 需要 ~500 recursion. 默认 250 只够 ~80 calls.
            # budget_override: PhaseManager 转移后通过 Orchestrator 传入, 打通 phase→budget
            if budget_override is not None:
                _mc = budget_override.max_calls
                _rec_limit = budget_override.recursion_limit
            else:
                _mc = self._max_tool_calls or 50
                # P1-5: 之前 max(250, _mc*5) 覆盖了 mode 联动 — research/extreme
                # 模式期望 500/400 recursion, 实际拿到 250. 现在取两者最大值,
                # 既保证 max_tool_calls 空间, 又让 mode 联动真正生效.
                _rec_limit = max(self._effective_recursion_limit(), _mc * 5)
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": _rec_limit,
            }

            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                use_sync_stream = isinstance(self.checkpointer, SqliteSaver)
            except Exception:
                logger.debug("best-effort op failed", exc_info=True)
                use_sync_stream = False

            from huginn.agents.loop_detector import LoopDetector
            from huginn.agents.tool_budget import ToolCallBudget
            from huginn.agents.tool_call_router import ToolCallRouter

            turn_budget = ToolCallBudget(
                max_calls=_mc,
                max_per_tool=self._max_tool_calls_per_tool,
            )
            self._tool_adapter.set_budget(turn_budget)
            turn_router = ToolCallRouter(budget=turn_budget)
            self._tool_adapter.set_router(turn_router)
            # 预算感知: 让 LLM 一开始就知道本轮预算上限与重置语义, 避免
            # 把预算耗尽误判成"死刑"而提前交卷 (路线图 P0-3 假耗尽).
            # 放在 inputs 尾部、用固定 id 替换旧消息, 不碰缓存的 system_prompt,
            # 也不让预算消息在历史里层层累积.
            try:
                _bs = turn_budget.status()
                _budget_msg = SystemMessage(
                    content=(
                        "## Tool Call Budget\n"
                        f"- This turn: {_bs['total_calls']}/{_bs['max_calls']} tool calls used/allowed "
                        f"(per-tool cap {_bs['max_per_tool']}).\n"
                        "- Budget is per turn and resets on the next turn. Exhausting it is "
                        "NOT game over — wrap up or switch tools, don't abandon the task."
                    ),
                    id="ctx_tool_budget",
                )
                _msgs = list(inputs.get("messages", []))
                _msgs = [
                    m for m in _msgs if getattr(m, "id", None) != "ctx_tool_budget"
                ]
                inputs["messages"] = _msgs + [_budget_msg]
            except Exception:
                logger.debug("budget injection skipped", exc_info=True)
            # AV5: 默认 skip — ToolLoopDetector + ThoughtLoopDetector 在长任务 (benchmark 或
            # 真实研究) 都会误判: agent 反复跑 code_tool 是正常, 写报告反复用术语
            # ("band gap"/"MAE") Jaccard > 0.85 也正常. 升级: mode-aware detector,
            # 区分 "同 tool 同输入" (真死循环) vs "同 tool 不同输入" (正常迭代).
            # 统一走 FeatureFlags 而非直接读 env var, 避免 benchmark 路径双配置冲突.
            # FeatureFlags 读 HUGINN_FEATURE_LOOP_DETECTOR, benchmark runner 极端模式开 / 普通模式关.
            from huginn.feature_flags import FeatureFlags
            _skip_loop = not FeatureFlags.shared().is_enabled("loop_detector")
            if not _skip_loop:
                turn_loop_detector = LoopDetector()
            else:
                turn_loop_detector = None
            self._tool_adapter.set_loop_detector(turn_loop_detector)

            # F2: σ₈ 半修补全 — ThoughtLoopDetector 也要受同一个 env 控制.
            # 之前只关 ToolLoopDetector, ThoughtLoopDetector 仍开, agent 写报告
            # 反复用 "band gap"/"MAE" 术语时 Jaccard > 0.85 三次就误判死循环.
            from huginn.agents.loop_detector import ThoughtLoopDetector
            self._thought_detector = None if _skip_loop else ThoughtLoopDetector()
            self._thought_loop_terminated = False

            max_retries = 3
            states_yielded = 0
            final_state: dict[str, Any] | None = None
            _last_reasoning = ""

            self._state_msg_offsets[thread_id] = 0
            if self.checkpointer is not None:
                try:
                    snapshot = graph.get_state(config)
                    existing_msgs = snapshot.values.get("messages", [])
                    self._state_msg_offsets[thread_id] = len(existing_msgs)
                except Exception:
                    logger.debug("checkpointer state fetch skipped", exc_info=True)

            # 统一事件总线: ON_LLM_REQUEST + ON_BEFORE_MESSAGE_SENT
            await _ubus.publish_llm_request(thread_id, len(messages))
            await _ubus.publish_before_message_sent(thread_id, len(messages))

            try:
                attempt = 0
                while attempt < max_retries:
                    try:
                        if use_sync_stream:
                            states = await asyncio.to_thread(
                                lambda g=graph: list(
                                    g.stream(
                                        inputs, config, stream_mode="values"
                                    )
                                )
                            )
                            for state in states:
                                # Sync stream doesn't get per-chunk messages, so extract
                                # reasoning from the accumulated AIMessage before yielding state
                                msgs = state.get("messages", [])
                                if msgs:
                                    last_msg = msgs[-1]
                                    if hasattr(last_msg, "additional_kwargs"):
                                        r = last_msg.additional_kwargs.get("reasoning_content", "")
                                        if r and r != _last_reasoning:
                                            yield {"_reasoning": r}
                                            _last_reasoning = r
                                self._process_stream_state(
                                    state, turn_span, thread_id, pet, _completion_records
                                )
                                states_yielded += 1
                                final_state = state
                                yield state
                                if getattr(self, "_thought_loop_terminated", False):
                                    logger.warning(
                                        "Terminating chat due to persistent thought loop"
                                    )
                                    _synth = await self._synthesize_closing_answer(
                                        final_state
                                    )
                                    if _synth and isinstance(final_state, dict):
                                        final_state["messages"] = (
                                            final_state.get("messages", []) + _synth
                                        )
                                    yield {
                                        "thought_loop_terminated": True,
                                        "state": final_state,
                                    }
                                    break
                                if self._break_flag:
                                    self._break_flag = False
                                    yield {"tool_break": True, "state": final_state}
                                    break
                                interrupt = await self._check_loop_interrupt(thread_id)
                                if interrupt and interrupt.get("cancelled"):
                                    raise InterruptCancelled(interrupt.get("reason", ""))
                        else:
                            # A3: 流式 watchdog 包裹 — 空闲超时后走 ainvoke 降级.
                            # thinking=high 时首 token 前的深度推理可能 >60s 无 chunk,
                            # 用随 thinking 放宽的空闲超时, 减少不必要的降级.
                            async for mode, data in _astream_with_watchdog(
                                graph.astream(
                                    inputs, config,
                                    stream_mode=["values", "messages"],
                                ),
                                idle_timeout=_thinking_stream_idle(),
                            ):
                                if mode == "messages":
                                    chunk, _meta = data
                                    chunk_type = type(chunk).__name__
                                    if not chunk_type.startswith("AIMessage"):
                                        continue
                                    text = ""
                                    if hasattr(chunk, "content") and isinstance(chunk.content, str):
                                        text = chunk.content
                                    reasoning = ""
                                    if hasattr(chunk, "additional_kwargs"):
                                        reasoning = chunk.additional_kwargs.get("reasoning_content", "")
                                    if text:
                                        # TPS: 首次 chunk 记 TTFT, 每个 chunk 累加字符数.
                                        if _tps_t_first is None:
                                            _tps_t_first = time.monotonic()
                                            turn_span.metadata["llm_ttft_ms"] = int(
                                                (_tps_t_first - _tps_t0) * 1000
                                            )
                                        _tps_chunk_chars += len(text)
                                        yield {"_token": text}
                                    if reasoning:
                                        yield {"_reasoning": reasoning}
                                    continue

                                state = data
                                self._process_stream_state(
                                    state, turn_span, thread_id, pet, _completion_records
                                )
                                states_yielded += 1
                                final_state = state
                                yield state
                                if getattr(self, "_thought_loop_terminated", False):
                                    logger.warning(
                                        "Terminating chat due to persistent thought loop"
                                    )
                                    _synth = await self._synthesize_closing_answer(
                                        final_state
                                    )
                                    if _synth and isinstance(final_state, dict):
                                        final_state["messages"] = (
                                            final_state.get("messages", []) + _synth
                                        )
                                    yield {
                                        "thought_loop_terminated": True,
                                        "state": final_state,
                                    }
                                    break
                                if self._break_flag:
                                    self._break_flag = False
                                    yield {"tool_break": True, "state": final_state}
                                    break
                                interrupt = await self._check_loop_interrupt(thread_id)
                                if interrupt and interrupt.get("cancelled"):
                                    raise InterruptCancelled(interrupt.get("reason", ""))
                        break
                    except TimeoutError:
                        # A3: 流式空闲超时 — 降级到流式收集, 不重试原流式.
                        # 不用无进度的 ainvoke: 它只能猜一个固定时长, 长推理会误杀.
                        # 这里用 _astream_with_watchdog 的"空闲超时"语义做进度感知 —
                        # 只要模型还在产出 (reasoning 或 content chunk) 就不中断,
                        # 只有完全空闲 (thinking 阶段也无 chunk 太久) 才终止.
                        # 外层再套一个 thinking 缩放的总兜底, 防无限 thinking 循环.
                        logger.warning(
                            "stream idle timeout after %ds (states_yielded=%d), "
                            "falling back to progress-aware stream collect",
                            _STREAM_IDLE_TIMEOUT, states_yielded,
                        )
                        turn_span.metadata["stream_watchdog_timeout"] = True
                        _ainvoke_timeout = float(os.environ.get(
                            "HUGINN_AINVOKE_TIMEOUT",
                            str(_thinking_scale_timeout()),
                        ))
                        _collect_idle = _thinking_stream_idle()
                        final_state = None
                        try:
                            async def _collect(
                                g=graph, idle_timeout=_collect_idle
                            ):
                                _st = None
                                async for mode, data in _astream_with_watchdog(
                                    g.astream(
                                        inputs, config,
                                        stream_mode=["values", "messages"],
                                    ),
                                    idle_timeout=idle_timeout,
                                ):
                                    if mode == "values":
                                        _st = data
                                return _st

                            final_state = await asyncio.wait_for(
                                _collect(), timeout=_ainvoke_timeout
                            )
                        except TimeoutError:
                            logger.warning(
                                "fallback stream collect timed out (idle=%ds, total=%ds)",
                                _collect_idle, _ainvoke_timeout,
                            )
                            raise TimeoutError(
                                f"stream and fallback collect both timed out "
                                f"(collect idle={_collect_idle}s, total={_ainvoke_timeout}s)"
                            ) from None
                        if final_state is None:
                            raise RuntimeError(
                                "fallback stream collect produced no state"
                            ) from None
                        self._process_stream_state(
                            final_state, turn_span, thread_id, pet, _completion_records
                        )
                        states_yielded += 1
                        yield final_state
                        break
                    except Exception as exc:
                        if isinstance(exc, InterruptCancelled):
                            raise
                        if states_yielded > 0:
                            raise
                        retryable = (
                            _is_rate_limit(exc)
                            or _is_overloaded(exc)
                            or _is_transient_network(exc)
                            or _is_context_overflow(exc)
                        )
                        if not retryable or attempt == max_retries - 1:
                            if (
                                _is_overloaded(exc)
                                and self._main_fallback_override is None
                                and (fb := self._select_main_fallback_model()) is not None
                            ):
                                logger.warning(
                                    "main chat 529 overloaded after %d attempts, "
                                    "switching to fallback model: %s",
                                    attempt + 1,
                                    getattr(fb, "model", type(fb).__name__),
                                )
                                self._main_fallback_override = fb
                                self._agent_graph = None
                                graph = self.build_graph()
                                await asyncio.sleep(_jitter(_exponential_backoff(1)))
                                attempt = 0
                                continue
                            raise
                        if _is_rate_limit(exc):
                            wait = _get_retry_after(exc)
                            if wait is None:
                                wait = _jitter(_exponential_backoff(attempt + 1))
                            else:
                                wait = _jitter(wait, jitter_ratio=0.1)
                        else:
                            wait = _jitter(_exponential_backoff(attempt + 1))
                        logger.warning(
                            "Graph invocation failed (attempt %d/%d), "
                            "retrying in %.2fs: %s",
                            attempt + 1,
                            max_retries,
                            wait,
                            exc,
                        )
                        # 对齐 Kimi Code stepRetryService 的 turn.step.retrying:
                        # 把重试信号发到 EventBus, 让 SSE / audit / 监控可见.
                        # ponytail: best-effort, publish 失败不影响主流程.
                        try:
                            from huginn.events.event_bus import AgentEvent, EventBus
                            from huginn.events.event_types import STEP_RETRY

                            await EventBus.shared().publish(AgentEvent(
                                type=STEP_RETRY,
                                timestamp=time.time(),
                                data={
                                    "attempt": attempt + 1,
                                    "max_attempts": max_retries,
                                    "error_type": type(exc).__name__,
                                    "error_message": str(exc)[:200],
                                    "wait_ms": int(wait * 1000),
                                    "states_yielded": states_yielded,
                                },
                                thread_id=thread_id,
                                source="agent.streaming",
                            ))
                        except Exception:
                            logger.debug(
                                "step retry event publish failed", exc_info=True
                            )
                        await asyncio.sleep(wait)
                        attempt += 1

                # Post-stream processing
                if final_state is not None:
                    # Sync any executing plan from PlanStore to session_state
                    self._sync_plan_from_store()

                    # reflection 已移到 finally 块, 确保timeout/取消也能跑

                    # Proactive pipeline suggestions
                    await self._maybe_inject_proactive_suggestion()

                    # Auto-compact when context > 60% (50% = warning)
                    compact_info = await self._maybe_auto_compact(
                        final_state, turn_span, thread_id,
                        graph=graph, config=config,
                    )
                    if compact_info:
                        yield {"_compacted": compact_info}

                    # Synthetic Continue after compaction / tool boundary
                    await self._maybe_inject_synthetic_continue(
                        final_state, thread_id
                    )

                    # If synthetic messages were injected, signal auto-continue
                    # so the WS handler can trigger another turn immediately
                    # instead of waiting for the user to send a new message.
                    pending = getattr(self, '_pending_synthetic_messages', None)
                    if pending:
                        yield {"_auto_continue": True}

                    ai_content = self._extract_last_ai_content(final_state)
                    # 统一事件总线: ON_LLM_RESPONSE + ON_AFTER_MESSAGE_SENT
                    await _ubus.publish_llm_response(thread_id, ai_content[:2000])
                    await _ubus.publish_after_message_sent(thread_id, ai_content[:2000])
                    if ai_content:
                        # OAK 启发: ai 消息写进 ConversationTree
                        try:
                            self._conversation_tree.add_message(
                                role="assistant", content=ai_content,
                                metadata={"thread_id": thread_id, "phase": self.phase},
                            )
                        except Exception:
                            logger.debug("ConversationTree add_message (ai) skipped", exc_info=True)
                        if self.style_learner is not None:
                            try:
                                self.style_learner.observe(message, ai_content)
                            except Exception:
                                logger.warning(
                                    "style_learner.observe failed",
                                    exc_info=True,
                                )
                        phase_target = self._check_phase_transition(ai_content)
                        if phase_target is not None:
                            self.transition_phase(phase_target)
                            logger.info(
                                "Phase auto-transitioned to %s",
                                phase_target.value,
                            )
            except Exception as exc:
                await _ubus.publish_pet_mood(PetMood.ERROR, f"Error: {exc}", {"thread_id": thread_id})
                raise
            finally:
                # 反思闭环: 即使 timeout/取消, 也要跑 reflection 让 evolution 记录失败.
                # 之前 reflection 在 try 块内, asyncio.wait_for 取消时直接跳过 ->
                # Benchmark 跑分期间 tool_calls.jsonl 记录为 0, 规则 usage_count 不增长.
                import sys as _sys
                _n_trs = len(self._session_state.tool_results_this_turn)
                print(f"[FINALLY-REFLECT] tool_results_this_turn={_n_trs}", file=_sys.stderr, flush=True)
                try:
                    if self._session_state.tool_results_this_turn:
                        self._run_post_turn_reflection()
                except Exception:
                    logger.debug("finally reflection failed", exc_info=True)
                # TPS 收尾: chunk_chars/4 ≈ tokens (latin). 写 turn_span + Prometheus.
                if _tps_t_first is not None and _tps_chunk_chars > 0:
                    elapsed = time.monotonic() - _tps_t_first
                    if elapsed > 0:
                        tps = (_tps_chunk_chars / 4.0) / elapsed
                        turn_span.metadata["llm_tps"] = round(tps, 1)
                        turn_span.metadata["llm_output_chars"] = _tps_chunk_chars
                        try:
                            from huginn.routes.metrics import track_llm_tps
                            track_llm_tps(
                                model=getattr(self.model, "model", "unknown"),
                                ttft_ms=turn_span.metadata.get("llm_ttft_ms", 0),
                                tps=tps,
                            )
                        except Exception:
                            logger.debug("TPS prometheus publish failed", exc_info=True)
                self._tool_adapter.set_budget(None)
                self._tool_adapter.set_router(None)
                self._tool_adapter.set_loop_detector(None)
                if self._main_fallback_override is not None:
                    self._main_fallback_override = None
                    self._agent_graph = None
                # Auto-save trajectory
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
                except Exception:
                    logger.debug("trajectory save failed", exc_info=True)
                # wire-level completion dump: prompt/response/tool_call/tool_result
                # 落盘 jsonl 给 red_team + 未来 RL 训练消费. 提取为 _dump_completion_records
                # 便于直接测试; prefix_merging (跨 turn 前缀去重) 是升级路径, 见该函数.
                _dump_completion_records(_completion_records, thread_id, _capture_turn_id)
                # STOP event
                # 统一事件总线: 一次调用桥接 STOP hook + SESSION_END hook + 内部EventBus + 插件EventBus
                from huginn.events.unified_bus import get_unified_bus
                _ubus = get_unified_bus(self)
                await _ubus.publish_stop(thread_id, self.workspace)
                await _ubus.publish_session_end(thread_id, self._turn_count)
                # Mode-based memory persistence
                if self.is_research_mode():
                    try:
                        self.memory.promote_session_summary(tier="long")
                    except Exception:
                        logger.debug(
                            "research-mode memory promote failed",
                            exc_info=True,
                        )
                # Session-state snapshot for next session
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
                        content=f"Session snapshot: {snapshot.get('l1_coordinates', 'no coordinates')} | cognitive_state: {snapshot.get('cognitive_state', '?')}",
                        category="conversation",
                        tags=tags,
                        source=f"session:{thread_id}",
                        importance=0.6,
                        tier="mid",
                    )
                except Exception:
                    logger.debug("session snapshot save failed", exc_info=True)
                await _ubus.publish_pet_mood(PetMood.IDLE, "Ready", {"thread_id": thread_id})
                self._turn_count += 1
                if (
                    self.memory_decay_enabled
                    and self.memory_decay_interval_turns > 0
                    and self._turn_count % self.memory_decay_interval_turns == 0
                ):
                    try:
                        summary = self.memory.maintenance(
                            prune_threshold=self.memory_decay_prune_threshold
                        )
                        pet.publish(
                            PetMood.SUCCESS,
                            "Memory maintenance",
                            {"summary": summary},
                        )
                    except Exception as exc:
                        logger.warning("Memory maintenance failed: %s", exc, exc_info=True)


    async def _synthesize_closing_answer(
        self, final_state: Any
    ) -> list[Any]:
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
            # 已有非空助手文本 → 无需兜底
            for m in reversed(msgs):
                if isinstance(m, AIMessage) and m.content and str(m.content).strip():
                    return []

            model = self.select_model("agent")
            if model is None or not hasattr(model, "ainvoke"):
                return []

            transcript: list[str] = []
            for m in msgs:
                role = getattr(m, "type", "unknown")
                content = getattr(m, "content", "")
                if isinstance(content, list):
                    content = "".join(
                        b.get("text", "") for b in content if isinstance(b, dict)
                    )
                if role in ("human", "user", "assistant") and content:
                    transcript.append(f"{role}: {str(content)[:800]}")
            if not transcript:
                return []

            from langchain_core.messages import HumanMessage, SystemMessage

            prompt = (
                "The conversation below ended without a clear final answer. "
                "Write a concise closing summary (2-5 sentences): what was done, "
                "and the key conclusions or open questions. If information is "
                "insufficient, say so honestly rather than guessing.\n\n"
                + "\n".join(transcript[-12:])
            )
            resp = await model.ainvoke(
                [
                    SystemMessage(
                        content="You are a concise, honest research summarizer."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            text = resp.content
            if isinstance(text, list):
                text = "".join(
                    b.get("text", "") for b in text if isinstance(b, dict)
                )
            if text and str(text).strip():
                return [AIMessage(content=str(text).strip())]
        except Exception:
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
        assert empty_results == [], f"empty stream should yield nothing: {empty_results}"

        # 4. 单 chunk 后超时: 第一个 chunk 透传, 第二个超时
        async def _one_then_slow():
            yield "fast"
            await asyncio.sleep(3.0)
            yield "slow"

        seen = []
        try:
            async for item in _astream_with_watchdog(_one_then_slow(), idle_timeout=0.5):
                seen.append(item)
        except TimeoutError:
            logger.debug("best-effort op failed", exc_info=True)
        assert seen == ["fast"], f"first chunk not yielded before timeout: {seen}"

    asyncio.run(_test_watchdog())

    # 5. _STREAM_IDLE_TIMEOUT 默认 60s, 可被 env 覆盖
    assert _STREAM_IDLE_TIMEOUT == 60.0, f"default idle timeout should be 60, got {_STREAM_IDLE_TIMEOUT}"

    print("A3 self-check OK (watchdog timeout + passthrough + thinking block protection)")
