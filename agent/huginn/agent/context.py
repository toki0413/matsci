"""Context building: system prompt assembly, tool filtering, cache stats."""

from __future__ import annotations

import logging
import os
from typing import Any

from huginn.context_manager import get_system_context, get_user_context
from huginn.project_memory import load_agents_md
from huginn.utils.prompt_cache import PromptCacheBuilder

logger = logging.getLogger(__name__)


class ContextMixin:
    """Methods that assemble the prompt context and manage tool visibility."""

    # Heavy simulation tools filtered out in chat mode.
    _EXPENSIVE_TOOL_NAMES: set[str] = {
        "vasp_tool", "lammps_tool", "cp2k_tool", "transolver_tool",
    }

    # Tools that stay visible regardless of query relevance.
    _ALWAYS_ON_TOOLS: set[str] = {
        "memory_tool", "knowledge_search", "structure_tool",
        "periodic_table_tool", "search_tool",
    }

    # Above this tool count, switch to query-aware retrieval.
    _TOOL_RETRIEVAL_THRESHOLD = 25

    # How many tools to keep after relevance filtering.
    _TOOL_RETRIEVAL_TOP_K = 15

    def _effective_system_prompt(self) -> str:
        """Base system prompt + mode prefix + phase prefix + env context."""
        if self._mode == "research":
            base = (
                "RESEARCH MODE: You are conducting systematic scientific research.\n"
                "- Always cite literature sources for claims\n"
                "- Quantify uncertainty in results\n"
                "- Compare results to published values\n"
                "- Flag unexpected results as potential discoveries\n"
                "- Write findings to knowledge base\n\n"
                f"{self.system_prompt}"
            )
        else:
            base = self.system_prompt

        prefix = self._phase_manager.prompt_prefix()
        base = f"{prefix}\n\n{base}" if prefix else base
        base = (
            f"{base}\n\n"
            "You can request a phase transition by including "
            "[PHASE:TARGET_PHASE] in your response. "
            "Available phases: LITERATURE, HYPOTHESIS, PLANNING, "
            "EXECUTION, VALIDATION, REPORTING."
        )
        # Inject cached system context (date + git status) and project
        # context (.huginn.md / AGENTS.md). Computed once per session
        # to avoid repeated git calls.
        workspace = self.workspace or os.getcwd()
        try:
            sys_ctx = get_system_context(str(workspace))
            if sys_ctx:
                ctx_lines = [f"{k}: {v}" for k, v in sys_ctx.items()]
                base = f"{base}\n\n# Environment\n" + "\n".join(ctx_lines)
            user_ctx = get_user_context(str(workspace))
            if user_ctx:
                base = (
                    f"{base}\n\n# Project Context\n"
                    + "\n".join(f"{k}: {v}" for k, v in user_ctx.items())
                )
            agents_md = load_agents_md(str(workspace))
            if agents_md:
                base = f"{base}\n\n# Project Memory\n{agents_md}"
        except Exception:
            logger.debug("context injection skipped", exc_info=True)
        # User taste profile — injected each turn so the agent adjusts
        # its answer style to the user's thinking preferences.
        try:
            from huginn.personalization import get_taste_directive

            taste = get_taste_directive()
            if taste:
                base = f"{base}\n\n# User Taste Profile\n{taste}"
        except Exception:
            logger.warning("taste profile injection failed", exc_info=True)

        # Hint about missing simulation backends so the LLM doesn't
        # waste turns calling tools that will just error.
        try:
            import shutil as _shutil
            _missing = []
            for _name, _exes in [
                ("VASP", ["vasp", "vasp_std", "vasp_gam"]),
                ("LAMMPS", ["lmp", "lmp_serial", "lammps"]),
                ("CP2K", ["cp2k"]),
            ]:
                if not any(_shutil.which(e) for e in _exes):
                    _missing.append(_name)
            if _missing:
                base += (
                    f"\n\nNote: {', '.join(_missing)} not installed locally. "
                    "Answer parameter questions from knowledge."
                )
        except Exception:
            logger.debug("tool availability check failed", exc_info=True)

        return base

    def _effective_tools(self, query: str | None = None) -> list[Any]:
        """Return the tool list filtered by mode, phase, and query relevance.

        Three layers of filtering, applied in order:
        1. Mode filter: chat mode drops expensive simulation tools;
           research mode keeps everything.
        2. Phase filter: when the current phase has a tool filter, only
           tools whose names appear in it are exposed.
        3. Query retrieval: when tool count exceeds threshold and a query
           is available, keep top-K by keyword relevance + always-on tools.
        """
        tools = list(self.langchain_tools)

        if self._mode == "chat":
            tools = [
                t for t in tools if t.name not in self._EXPENSIVE_TOOL_NAMES
            ]

        phase_tools = self._phase_manager.tool_filter()
        if phase_tools is not None:
            tools = [t for t in tools if t.name in phase_tools]

        # query-aware retrieval: 只在工具数多且无 phase 约束时触发
        if (
            query
            and phase_tools is None
            and len(tools) > self._TOOL_RETRIEVAL_THRESHOLD
        ):
            tools = self._retrieve_relevant_tools(tools, query)

        return tools

    def _retrieve_relevant_tools(
        self, tools: list[Any], query: str,
    ) -> list[Any]:
        """Keyword-based tool retrieval — keep top-K by relevance score.

        Scoring: 1 point per query keyword found in tool name or description,
        plus a small bonus from SkillEvolutionLayer's confidence in the tool.
        Always-on tools bypass scoring.
        """
        query_lower = query.lower()
        # 分词: 空格 + 常见标点
        keywords = {
            w for w in query_lower.replace(",", " ").replace(".", " ")
            .replace("?", " ").replace("/", " ").split()
            if len(w) > 2
        }

        scored: list[tuple[float, Any]] = []
        always_on: list[Any] = []

        for t in tools:
            name = getattr(t, "name", "") or ""
            desc = getattr(t, "description", "") or ""
            blob = f"{name} {desc}".lower()

            if name in self._ALWAYS_ON_TOOLS:
                always_on.append(t)
                continue

            # user mentioned the tool by name → always include it
            if name and name in query_lower:
                always_on.append(t)
                continue

            score = sum(1 for kw in keywords if kw in blob)

            # SkillEvolutionLayer 信念加成: 高置信度工具小幅提分
            try:
                from huginn.skills.evolution import SkillEvolutionLayer
                layer = SkillEvolutionLayer.shared()
                beliefs = layer.get_tool_beliefs(name)
                if beliefs:
                    avg_conf = sum(b.confidence for b in beliefs) / len(beliefs)
                    score += avg_conf * 0.5
            except Exception:
                logger.debug("skill evolution beliefs unavailable", exc_info=True)

            scored.append((score, t))

        # 按分数降序, 取 top-K, 加上 always-on
        scored.sort(key=lambda x: -x[0])
        top_k = self._TOOL_RETRIEVAL_TOP_K - len(always_on)
        retrieved = [t for _, t in scored[:max(0, top_k)]]

        result = always_on + retrieved
        if len(result) < len(tools):
            logger.debug(
                "tool retrieval: %d → %d (query: %s)",
                len(tools), len(result), query[:60],
            )
        return result

    def _tool_names_for_validation(self) -> set[str]:
        """Collect all visible tool names for ToolNameValidationHook."""
        try:
            tools = self._effective_tools()
            names: set[str] = set()
            for t in tools:
                try:
                    name = t.name
                except Exception:
                    continue
                if isinstance(name, str) and name:
                    names.add(name)
            return names
        except Exception:
            logger.warning("_tool_names_for_validation raised", exc_info=True)
            return set()

    def _build_memory_text(self, query: str | None = None) -> str:
        """Recall relevant long-term memory formatted for the prompt tail."""
        return self._ctx_builder.build_memory_text(query)

    def _build_kg_text(self, query: str) -> str:
        """Query the project knowledge graph."""
        return self._ctx_builder.build_kg_text(query)

    def _build_kb_text(self, query: str) -> str:
        """Query the domain knowledge base."""
        builder = getattr(self, "_ctx_builder", None)
        if builder is not None:
            return builder.build_kb_text(query)
        if not getattr(self, "kb_enabled", False):
            return ""
        kb = getattr(self, "_kb", None)
        if kb is None or kb.count() == 0:
            return ""
        chunks = kb.query(query, top_k=5)
        if not chunks:
            return ""
        lines = []
        for i, c in enumerate(chunks, 1):
            text = (c.get("text") or "").strip()
            if not text:
                continue
            if len(text) > 800:
                text = text[:800] + "..."
            lines.append(f"[{i}] {text}")
        if not lines:
            return ""
        return "### Domain Knowledge Context\n" + "\n".join(lines) + "\n"

    def _build_state_modifier(self) -> list:
        """Static system message used as the graph state modifier."""
        return self._cache_builder.build_state_modifier()

    def _build_input_messages(
        self,
        message: str,
        memory_text: str | None = None,
        kg_text: str | None = None,
        include_history: bool | None = None,
        kb_text: str | None = None,
        session_state: Any = None,
    ) -> list[Any]:
        """Dynamic input messages: history + memory + KG + KB + emotion + user."""
        return self._ctx_builder.build_input_messages(
            message,
            memory_text=memory_text,
            kg_text=kg_text,
            kb_text=kb_text,
            include_history=include_history,
            session_state=session_state,
        )

    def _conversation_tree_history_to_messages(self) -> list[Any]:
        """Convert the active conversation path to LC messages."""
        return self._ctx_builder.conversation_tree_to_messages()

    def _build_emotion_text(self, message: str) -> str | None:
        """Update persona emotional trajectory and return mood context."""
        return self._ctx_builder.build_emotion_text(message)

    def _rebuild_cache_builder(self) -> None:
        """Recreate the cache builder when the system prompt changes."""
        self._cache_builder = PromptCacheBuilder(
            system_prompt=self.system_prompt,
            begin_dialogs=self.begin_dialogs,
            cache_control=self.prompt_cache_control,
        )
        self._cache_builder.set_provider(self._detect_provider())
        if hasattr(self, "_ctx_builder"):
            self._ctx_builder._cache_builder = self._cache_builder

    def _get_tool_description_text(self) -> str:
        """Serialize all visible tool schemas for token estimation.

        Must include the full JSON schema (name + description + parameters),
        not just descriptions — description-only underestimates by ~10x and
        causes compact to trigger too late.
        """
        if self._tool_description_text is None:
            import json as _json
            parts = []
            for t in self._effective_tools():
                args_schema = getattr(t, "args_schema", None)
                if args_schema is not None and hasattr(
                    args_schema, "model_json_schema"
                ):
                    params = args_schema.model_json_schema()
                else:
                    params = {}
                schema = {
                    "name": getattr(t, "name", ""),
                    "description": getattr(t, "description", "") or "",
                    "parameters": params,
                }
                parts.append(_json.dumps(schema, ensure_ascii=False))
            self._tool_description_text = "\n".join(parts)
        return self._tool_description_text

    def _invalidate_tool_description_cache(self) -> None:
        """Invalidate cached tool descriptions when tools change."""
        self._tool_description_text = None

    def _extract_cache_stats(self, messages: list[Any]) -> dict[str, Any]:
        """Extract provider cache-hit telemetry from the latest assistant turn."""
        from langchain_core.messages import AIMessage

        stats: dict[str, Any] = {}
        for msg in reversed(messages):
            if not isinstance(msg, AIMessage):
                continue
            meta = getattr(msg, "response_metadata", None) or {}
            for key in (
                "cache_creation_input_tokens",
                "cache_read_input_tokens",
                "input_tokens",
                "output_tokens",
            ):
                if key in meta:
                    stats[key] = meta[key]
            usage = meta.get("usage")
            if isinstance(usage, dict):
                for k, v in usage.items():
                    stats[f"usage_{k}"] = v
            if stats:
                break
        return stats

    def set_persona(
        self,
        persona: Any | None = None,
        system_prompt: str | None = None,
        begin_dialogs: list[tuple[str, str]] | None = None,
        emotion_tracker: Any | None = None,
    ) -> None:
        """Switch the agent's active persona at runtime.

        Rebuilds the prompt-cache builder and invalidates the compiled
        graph so the new system prompt and begin dialogs take effect
        on the next turn.  Memory and knowledge-graph state are preserved.
        """
        if persona is not None:
            self.persona_name = persona.name
            self.system_prompt = system_prompt or persona.system_prompt
            self.begin_dialogs = (
                begin_dialogs
                if begin_dialogs is not None
                else [
                    (d.get("role", "user"), d.get("content", ""))
                    for d in persona.begin_dialogs
                ]
            )
        else:
            if system_prompt is not None:
                self.system_prompt = system_prompt
            if begin_dialogs is not None:
                self.begin_dialogs = begin_dialogs
        if emotion_tracker is not None:
            self.emotion_tracker = emotion_tracker
            if hasattr(self, "_ctx_builder"):
                self._ctx_builder.emotion_tracker = emotion_tracker
        self._rebuild_cache_builder()
        self._agent_graph = None
