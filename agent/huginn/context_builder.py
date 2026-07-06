"""Context builder — extracts prompt assembly logic from HuginnAgent.

This module isolates the responsibility of building the dynamic context
that gets injected into each LLM call:

* Long-term memory recall (with research-log conjectures)
* Project knowledge graph queries
* Domain knowledge base (first-principles reference) retrieval
* Conversation tree history reconstruction
* Persona emotion tracking

By extracting these into a standalone class we reduce the ``HuginnAgent``
god-class footprint and make the context-building pipeline independently
testable.

The agent delegates to a ``ContextBuilder`` instance — all public methods
on the agent that previously built context now forward here.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

logger = logging.getLogger(__name__)


class ContextBuilder:
    """Builds the dynamic context (memory, KG, KB, emotion) for each turn.

    Parameters
    ----------
    memory_manager
        The agent's :class:`MemoryManager` for long-term recall.
    workspace
        Path to the workspace directory (for KG/KB initialization).
    kg_enabled / kb_enabled
        Whether the knowledge graph / knowledge base are active.
    kg_depth / kg_top_k
        Knowledge graph query parameters.
    emotion_tracker
        Optional persona emotion tracker (may be ``None``).
    checkpointer
        The LangGraph checkpointer (if active — affects history inclusion).
    conversation_tree
        The agent's conversation branch tree (for history reconstruction).
    cache_builder
        The :class:`PromptCacheBuilder` for assembling final message lists.
    """

    def __init__(
        self,
        memory_manager: Any,
        workspace: str | Path,
        *,
        kg_enabled: bool = False,
        kb_enabled: bool = False,
        kg_depth: int = 1,
        kg_top_k: int = 5,
        emotion_tracker: Any | None = None,
        checkpointer: Any | None = None,
        conversation_tree: Any | None = None,
        cache_builder: Any | None = None,
    ) -> None:
        self.memory = memory_manager
        self.workspace = str(workspace)
        self.kg_enabled = kg_enabled
        self.kb_enabled = kb_enabled
        self.kg_depth = kg_depth
        self.kg_top_k = kg_top_k
        self.emotion_tracker = emotion_tracker
        self.checkpointer = checkpointer
        self._conversation_tree = conversation_tree
        self._cache_builder = cache_builder

        # Lazy-init caches
        self._kg: Any | None = None
        self._kb: Any | None = None

    # ── Memory text ────────────────────────────────────────────────

    def build_memory_text(self, query: str | None = None) -> str:
        """Recall relevant long-term memory + research-log conjectures.

        Returns a formatted string suitable for the prompt tail.
        """
        if not query:
            query = "materials science computation"
        parts: list[str] = []
        try:
            mem = self.memory.recall_for_prompt(query, max_entries=3)
            if mem:
                parts.append(mem)
        except Exception:
            logger.warning("memory.recall failed in context injection", exc_info=True)

        # Inject verified/in-progress conjectures from the research log
        try:
            from huginn.research_log import get_research_log
            log = get_research_log()
            verified = log.list_by_status("verified", limit=3)
            in_progress = log.list_by_status("in_progress", limit=2)
            if verified or in_progress:
                lines = ["### Research Log (recent conjectures)"]
                for r in verified:
                    lines.append(f"- [verified] {r.title}")
                for r in in_progress:
                    lines.append(f"- [in_progress] {r.title}")
                lines.append("### End Research Log")
                parts.append("\n".join(lines))
        except Exception:
            logger.warning("research_log read failed", exc_info=True)

        return "\n\n".join(parts) if parts else ""

    # ── Knowledge graph ────────────────────────────────────────────

    def build_kg_text(self, query: str) -> str:
        """Query the project knowledge graph and format results."""
        if not self.kg_enabled:
            return ""
        try:
            from huginn.kg.graph import ProjectKnowledgeGraph

            if self._kg is None:
                self._kg = ProjectKnowledgeGraph(Path(self.workspace) / ".huginn")
            result = self._kg.query(query, depth=self.kg_depth, top_k=self.kg_top_k)
            nodes = {n["id"] for n in result.get("nodes", [])}
            if not nodes:
                return ""
            text = self._kg.to_text(nodes)
            if not text:
                return ""
            return (
                "### Project Knowledge Context\n"
                "The following project-specific facts and relationships may help:\n"
                f"{text}\n"
                "### End Project Knowledge Context"
            )
        except Exception:
            return ""

    # ── Domain knowledge base ──────────────────────────────────────

    def build_kb_text(self, query: str) -> str:
        """Query the domain knowledge base (vector retrieval).

        Also performs cross-reference: when KB chunks are found, their
        text is used as a secondary query to recall related memories,
        creating a memory↔KB cross-reference loop.
        """
        if not self.kb_enabled:
            return ""
        try:
            if self._kb is None:
                from huginn.knowledge.store import get_knowledge_base
                self._kb = get_knowledge_base(self.workspace)
            if self._kb.count() == 0:
                return ""
            chunks = self._kb.query(query, top_k=5)
            if not chunks:
                return ""
            lines = []
            for i, c in enumerate(chunks, 1):
                text = (c.get("text") or "").strip()
                if not text:
                    continue
                if len(text) > 800:
                    text = text[:800] + "…"
                lines.append(f"[{i}] {text}")

                # ── Cross-reference: KB chunk → memory recall ──────
                # When a KB chunk is found, use its text as a query to
                # recall related long-term memories. This creates a
                # bidirectional link: if the agent previously learned
                # something related to this KB content, it surfaces here.
                if self.memory and i <= 2:  # only for top 2 chunks
                    try:
                        related = self.memory.recall_for_prompt(
                            text[:200], max_entries=1
                        )
                        if related:
                            lines.append(f"    ↳ Memory: {related[:200]}")
                    except Exception:
                        pass

            if not lines:
                return ""
            body = "\n".join(lines)
            return (
                "### Domain Knowledge Context\n"
                "The following first-principles reference chunks may ground your answer. "
                "Cite the source numbers when relevant.\n"
                f"{body}\n"
                "### End Domain Knowledge Context"
            )
        except Exception:
            return ""

    # ── Emotion ────────────────────────────────────────────────────

    def build_emotion_text(self, message: str) -> str | None:
        """Update persona emotional trajectory and return mood context."""
        if self.emotion_tracker is None:
            return None
        self.emotion_tracker.update_from_message(message, source="user")
        return self.emotion_tracker.context_prompt()

    # ── Conversation history ────────────────────────────────────────

    def conversation_tree_to_messages(self) -> list[Any]:
        """Convert the active conversation path to LC messages.

        Excludes the last node (the current user message being handled).
        """
        if self._conversation_tree is None:
            return []

        from langchain_core.messages import (
            AIMessage,
            HumanMessage,
            SystemMessage,
            ToolMessage,
        )

        messages: list[Any] = []
        path = self._conversation_tree.active_path()
        for node_id in path[:-1]:
            node = self._conversation_tree.get_node(node_id)
            if node is None:
                continue
            meta = node.metadata or {}
            if node.role == "user":
                messages.append(HumanMessage(content=node.content))
            elif node.role == "assistant":
                tool_calls = meta.get("tool_calls")
                if tool_calls:
                    messages.append(
                        AIMessage(content=node.content, tool_calls=tool_calls)
                    )
                else:
                    messages.append(AIMessage(content=node.content))
            elif node.role == "system":
                messages.append(SystemMessage(content=node.content))
            elif node.role == "tool":
                messages.append(
                    ToolMessage(
                        content=node.content,
                        tool_call_id=meta.get("tool_call_id", ""),
                        name=meta.get("name"),
                    )
                )
        return messages

    # ── Plan & session continuity ──────────────────────────────────

    def build_plan_text(self, session_state=None) -> str:
        """Inject active plan context so the LLM knows where we are in the plan.

        Without this the model has no idea it's supposed to be executing
        step 2 of 3 — the plan-aware piece of loop engineering.
        """
        if session_state is None or not getattr(session_state, "active_plan_id", None):
            return ""

        parts = ["### Current Plan"]
        parts.append(f"Objective: {session_state.active_plan_objective}")
        parts.append(f"Step: {session_state.active_plan_step_index + 1}")
        if session_state.l1_coordinates:
            parts.append(f"Position: {session_state.l1_coordinates}")
        parts.append(f"Cognitive mode: {session_state.cognitive_mode.value}")
        parts.append("### End Current Plan")
        return "\n".join(parts)

    def build_session_continuity(self, session_state=None) -> str:
        """Inject previous session summary for cross-session continuity.

        Each new session would otherwise start blank — this surfaces
        what was discussed last time and the user's recent goals so the
        LLM can reference prior work.
        """
        if session_state is None:
            return ""

        parts = []
        if getattr(session_state, "last_session_summary", ""):
            parts.append("### Previous Session")
            parts.append(session_state.last_session_summary)
            parts.append("### End Previous Session")
        if getattr(session_state, "user_goals_history", []):
            recent_goals = session_state.user_goals_history[-5:]
            parts.append("### Your Recent Goals")
            for i, goal in enumerate(recent_goals, 1):
                parts.append(f"{i}. {goal}")
            parts.append("### End Recent Goals")
        return "\n\n".join(parts) if parts else ""

    # ── Full input messages ────────────────────────────────────────

    def build_input_messages(
        self,
        message: str,
        *,
        memory_text: str | None = None,
        kg_text: str | None = None,
        kb_text: str | None = None,
        include_history: bool | None = None,
        session_state: Any = None,
    ) -> list[Any]:
        """Assemble the full input message list for an LLM call.

        Combines: system prompt (via cache builder) + conversation history
        + memory + KG + KB + emotion + plan status + session continuity
        + current user message.
        """
        if memory_text is None:
            memory_text = self.build_memory_text(query=message)
        if kg_text is None:
            kg_text = self.build_kg_text(query=message)
        if kb_text is None:
            kb_text = self.build_kb_text(query=message)

        if include_history is None:
            include_history = self.checkpointer is None

        history_messages: list[Any] | None = None
        if include_history:
            history_messages = self.conversation_tree_to_messages()

        messages = self._cache_builder.build_input_messages(
            memory_text,
            message,
            kg_text=kg_text,
            history_messages=history_messages,
            kb_text=kb_text,
        )

        emotion_text = self.build_emotion_text(message)
        if emotion_text:
            from langchain_core.messages import SystemMessage
            messages.insert(-1, SystemMessage(content=emotion_text, id="ctx_emotion"))

        # Plan + cross-session context sit right before the user message,
        # mirroring how emotion text is injected above.
        plan_text = self.build_plan_text(session_state)
        if plan_text:
            from langchain_core.messages import SystemMessage
            messages.insert(-1, SystemMessage(content=plan_text, id="ctx_plan"))
        continuity_text = self.build_session_continuity(session_state)
        if continuity_text:
            from langchain_core.messages import SystemMessage
            messages.insert(-1, SystemMessage(content=continuity_text, id="ctx_continuity"))

        return messages
