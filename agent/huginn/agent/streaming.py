"""Chat streaming loop, phase management, and context compaction."""

from __future__ import annotations

import asyncio
import logging
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
from huginn.phases import ResearchPhase
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
    ) -> None:
        """Update memory, branch tree, telemetry, and pet status from one graph state."""
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
    ) -> dict[str, Any] | None:
        """Trigger PRE_COMPACT hook + promote session summary when context > 70%.

        Returns ``{"before_pct": int, "after_pct": int}`` if compaction ran, else None.
        """
        if self._model_context_window <= 0:
            return None

        usage = self._extract_usage_tokens()
        if not any(usage.values()):
            return None

        before = calculate_context_usage(usage, self._model_context_window)
        if before["used"] <= 70:
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

            if not pipeline_block and not ended_at_tool:
                return

            parts = ["[System] Continue if you have next steps."]
            if pipeline_block:
                parts.append(pipeline_block)
            if prov_block:
                parts.append(prov_block)
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
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a message to the Agent and stream responses.

        Stores messages in session memory and tracks tool calls for
        auto-promotion to long-term memory.

        If *image_path* is provided, the agent routes the image through
        the vision fallback chain.
        """
        set_thread_id(thread_id)
        set_user_message(message)
        self.thread_id = thread_id
        self._current_user_message = message

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
        if image_path:
            _vr = VisionRouter(
                visual_encoder=getattr(self, "_visual_encoder", None),
                image_index=getattr(self, "_image_index", None),
            )
            model_name = getattr(self.model, "model", None) or getattr(self.model, "model_name", "")
            _vision_route = _vr.route(model_name, message or image_path)
            if _vision_route == VisionRoute.BOTH:
                _vision_content, _cv_hints = _vr.coordinate(message, image_path, model_name)
            elif _vision_route == VisionRoute.CV_TOOLS:
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

            memory_text = self._build_memory_text(query=message)
            kb_text = self._build_kb_text(query=message)
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
                    messages = _pg.redact_messages_for_cloud(messages)
            except RuntimeError:
                raise
            except Exception:
                logger.warning(
                    "PrivacyGuard hook failed", exc_info=True
                )

            inputs = {"messages": messages}

            # Compact initial messages if a context budget is configured.
            if self.context_budget_tokens > 0:
                summarizer = self._make_summarizer()
                if summarizer is not None:
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

            config = {
                "configurable": {"thread_id": thread_id},
                "recursion_limit": 250,
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
                max_calls=self._max_tool_calls,
                max_per_tool=self._max_tool_calls_per_tool,
            )
            self._tool_adapter.set_budget(turn_budget)
            turn_router = ToolCallRouter(budget=turn_budget)
            self._tool_adapter.set_router(turn_router)
            turn_loop_detector = LoopDetector()
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
                                    state, turn_span, thread_id, pet
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
                                        yield {"_token": text}
                                    if reasoning:
                                        yield {"_reasoning": reasoning}
                                    continue

                                state = data
                                self._process_stream_state(
                                    state, turn_span, thread_id, pet
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

                    # Auto-compact when context > 70%
                    compact_info = await self._maybe_auto_compact(
                        final_state, turn_span, thread_id
                    )
                    if compact_info:
                        yield {"_compacted": compact_info}

                    # Synthetic Continue after compaction / tool boundary
                    await self._maybe_inject_synthetic_continue(
                        final_state, thread_id
                    )

                    ai_content = self._extract_last_ai_content(final_state)
                    if ai_content:
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
