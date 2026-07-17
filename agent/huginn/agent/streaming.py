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
    get_context_window,
)
from huginn.hooks import HookContext, PRE_COMPACT, STOP, USER_PROMPT_SUBMIT
from huginn.interaction.interrupt import InterruptCancelled, get_interrupt_manager
from huginn.llm_retry import (
    _get_retry_after,
    _is_context_overflow,
    _is_overloaded,
    _is_rate_limit,
    _is_transient_network,
    _jitter,
    _exponential_backoff,
)
from huginn.pet import PetMood, get_pet_bus
from huginn.phases import BudgetSpec, ResearchPhase
from huginn.privacy import redact_secrets, scan_for_secrets
from huginn.utils.context import (
    compact_messages,
    estimate_message_tokens,
    summarize_compact_messages,
)
from huginn.utils.session_context import (
    get_thread_id,
    get_user_message,
    set_thread_id,
    set_user_message,
)
from huginn.utils.tokens import count_tokens

logger = logging.getLogger(__name__)

# Marker the LLM can embed in its response to request a phase transition.
_PHASE_MARKER = re.compile(r"\[PHASE:\s*(\w+)\s*\]", re.IGNORECASE)


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
        except Exception:
            logger.debug("ConversationTree fork skipped", exc_info=True)
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
                    self._session_state.add_tool_result(
                        {
                            "tool_name": getattr(msg, "name", None) or "unknown",
                            "content": msg.content,
                            "success": not (
                                isinstance(msg.content, str)
                                and msg.content.startswith("Error")
                            ),
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
            # Wire token usage + cost to Prometheus
            try:
                from huginn.routes.metrics import track_llm_usage
                track_llm_usage(
                    getattr(self, "config", None) and self.config.model or "unknown",
                    cache_stats,
                )
            except Exception:
                logger.debug("metrics track_llm_usage failed", exc_info=True)
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
        对齐, 升级路径是改成按消息 metadata 标记 root 而非按位置.

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
        keep_root_n = int(os.environ.get("HUGINN_KEEP_ROOT_N", "0"))
        body_start = keep_root_n
        body_end = len(msgs) - keep_last_n
        if body_end <= body_start:
            return 0

        # 算要 drop 几条才能到 budget
        drop_count = 0
        acc = total
        for i in range(body_start, body_end):
            if acc <= self.context_budget_tokens:
                break
            acc -= per_msg_tokens[i]
            drop_count += 1

        if drop_count == 0:
            return 0

        # 收集要 drop 的 message IDs — system 消息不删, 没 ID 的没法删
        from langchain_core.messages import RemoveMessage, SystemMessage

        drop_ids: list[str] = []
        for i in range(body_start, body_start + drop_count):
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
            try:
                from huginn.provenance.registry import ProvenanceRegistry
                prov_block = ProvenanceRegistry.shared().to_context_block()
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

            if not pipeline_block and not task_block and not ended_at_tool:
                return

            parts = ["[System] Continue if you have next steps."]
            if pipeline_block:
                parts.append(pipeline_block)
            if prov_block:
                parts.append(prov_block)
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
        budget_override: "BudgetSpec | None" = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a message to the Agent and stream responses.

        Stores messages in session memory and tracks tool calls for
        auto-promotion to long-term memory.

        If *image_path* is provided, the agent routes the image through
        the vision fallback chain.

        If *budget_override* is provided (BudgetSpec), temporarily
        overrides max_tool_calls / recursion_limit for this turn.
        PhaseManager.proposed_budget → Orchestrator → chat(budget_override=...).
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

        # OAK 启发: ConversationTree 通电 — 把每条 user/ai 消息写进树
        # phase 转移时 fork 出新分支, 让研究历史成为树而非线性序列
        try:
            user_node = self._conversation_tree.add_message(
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

        if self._turn_count == 0:
            self._init_session_continuity()

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
                    from huginn.server_core import get_visual_encoder, get_image_index

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
                            _vision_delegated = await _team._delegate(
                                TeamRole.VISION, vision_task,
                                {"original_task": message or ""},
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
        pet.publish(PetMood.THINKING, "Thinking...", {"thread_id": thread_id})

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
            if getattr(self, "_needs_compaction", False) and self.context_budget_tokens > 0:
                summarizer = self._make_summarizer()
                if summarizer is not None:
                    self._needs_compaction = False  # 清 flag, 避免重复触发
                    logger.info("mode-switch triggered compaction (CSM S3/S6)")
                    inputs["messages"], self._conversation_summary = (
                        await summarize_compact_messages(
                            inputs["messages"],
                            self.context_budget_tokens,
                            keep_last_n=4,
                            summarizer=summarizer,
                            existing_summary=self._build_compact_summary(),
                        )
                    )
                else:
                    inputs["messages"] = compact_messages(
                        inputs["messages"],
                        self.context_budget_tokens,
                        keep_last_n=1,
                        keep_root_n=int(os.environ.get("HUGINN_KEEP_ROOT_N", "0")),
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
                        )
                    )
                else:
                    inputs["messages"] = compact_messages(
                        inputs["messages"],
                        self.context_budget_tokens,
                        keep_last_n=1,
                        keep_root_n=int(os.environ.get("HUGINN_KEEP_ROOT_N", "0")),
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
                _rec_limit = max(250, _mc * 5)
            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": _rec_limit,
            }

            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
                use_sync_stream = isinstance(self.checkpointer, SqliteSaver)
            except Exception:
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
            # ponytail: RCB 场景 skip loop detector — agent 反复跑分析脚本是正常行为,
            # loop detector 误判为循环. 升级: mode-aware detector with semantic diff.
            if os.environ.get("HUGINN_SKIP_LOOP_DETECTOR", "").lower() not in ("1", "true", "yes"):
                turn_loop_detector = LoopDetector()
            else:
                turn_loop_detector = None
            self._tool_adapter.set_loop_detector(turn_loop_detector)

            from huginn.agents.loop_detector import ThoughtLoopDetector
            self._thought_detector = ThoughtLoopDetector()
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
            try:
                attempt = 0
                while attempt < max_retries:
                    try:
                        if use_sync_stream:
                            states = await asyncio.to_thread(
                                lambda: list(
                                    graph.stream(
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
                            async for mode, data in graph.astream(
                                inputs, config,
                                stream_mode=["values", "messages"],
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
                                if self._break_flag:
                                    self._break_flag = False
                                    yield {"tool_break": True, "state": final_state}
                                    break
                                interrupt = await self._check_loop_interrupt(thread_id)
                                if interrupt and interrupt.get("cancelled"):
                                    raise InterruptCancelled(interrupt.get("reason", ""))
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
                        await asyncio.sleep(wait)
                        attempt += 1

                # Post-stream processing
                if final_state is not None:
                    # Sync any executing plan from PlanStore to session_state
                    self._sync_plan_from_store()

                    # Run rules-based reflection on this turn's tool results
                    self._run_post_turn_reflection()

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
                pet.publish(PetMood.ERROR, f"Error: {exc}", {"thread_id": thread_id})
                raise
            finally:
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
                    traj_dir = self.workspace / ".huginn" / "trajectories"
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
                # 落盘 jsonl 给 red_team + 未来 RL 训练消费.
                # ponytail: 只 dump 非空 records, 失败静默. 升级路径: 加 prefix_merging.
                if _completion_records:
                    try:
                        import json
                        from huginn.utils.runtime import get_runtime_home
                        comp_dir = get_runtime_home() / "completions" / thread_id
                        comp_dir.mkdir(parents=True, exist_ok=True)
                        comp_path = comp_dir / f"{_capture_turn_id}.jsonl"
                        with open(comp_path, "w", encoding="utf-8") as f:
                            for rec in _completion_records:
                                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                    except Exception:
                        logger.debug("completion dump failed", exc_info=True)
                # STOP event
                try:
                    stop_ctx = HookContext(
                        tool_name="agent_turn",
                        metadata={
                            "thread_id": thread_id,
                            "workspace": self.workspace,
                        },
                    )
                    await self.hook_manager.trigger(STOP, stop_ctx)
                except Exception:
                    logger.warning("STOP hook raised", exc_info=True)
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
                pet.publish(PetMood.IDLE, "Ready", {"thread_id": thread_id})
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
