"""Prompt-caching utilities for HuginnAgent.

LLM providers (Anthropic Claude, Kimi, etc.) cache the *prefix* of a prompt.
To maximize cache hits we must keep the beginning of every request stable:

1. Static system prompt and persona begin-dialogs come first.
2. Dynamic content (recalled long-term memory, the current user message)
   is appended afterwards.
3. Optional provider-specific ``cache_control`` hints mark the last static
   block so the provider can reuse the KV-cache for the entire prefix.
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class PromptCacheBuilder:
    """Build message lists with a stable, cache-friendly static prefix.

    The builder splits each LLM request into:

    * ``state_modifier`` - the static system prompt only. This becomes the
      LangGraph ``state_modifier`` or the ``system_prompt`` for DeepAgents.
    * ``input_messages`` - persona begin-dialogs + optional memory context +
      the current user message. Begin-dialogs are static, but they live in the
      input stream so that a changing memory/user tail does not invalidate the
      cached prefix that precedes it.

    Provider-specific ``cache_control`` markers are only emitted for providers
    known to support prompt caching (Anthropic Claude, Kimi). Other providers
    still benefit from the stable prefix because identical prefixes share KV
    cache on the provider side even without explicit markers.
    """

    _SUPPORTED_PROVIDERS = {"anthropic", "claude", "kimi", "moonshot"}

    def __init__(
        self,
        system_prompt: str,
        begin_dialogs: list[tuple[str, str]] | None = None,
        cache_control: bool = False,
        provider: str | None = None,
    ):
        self.system_prompt = system_prompt
        self.begin_dialogs = begin_dialogs or []
        self.cache_control = cache_control
        self.provider = (provider or "").lower().strip()

    def set_provider(self, provider: str | None) -> None:
        """Update provider after the builder is constructed."""
        self.provider = (provider or "").lower().strip()

    def _provider_supports_cache_control(self) -> bool:
        if not self.cache_control:
            return False
        if not self.provider:
            # Default to the Anthropic-style ephemeral marker for backward
            # compatibility when no provider is specified.
            return True
        return any(p in self.provider for p in self._SUPPORTED_PROVIDERS)

    def _cache_control_kwargs(self) -> dict[str, Any]:
        if "kimi" in self.provider or "moonshot" in self.provider:
            # Kimi-style context caching uses the same ephemeral marker as
            # Anthropic in most integrations; override here if needed.
            return {"cache_control": {"type": "ephemeral"}}
        return {"cache_control": {"type": "ephemeral"}}

    def _static_prefix(self) -> list[BaseMessage]:
        """Return system prompt + begin-dialogs as a single static prefix.

        When ``cache_control`` is enabled, the *last* static message is
        tagged so the provider can cache everything up to that point.

        Each begin-dialog message carries a stable positional ID (``bd_0``,
        ``bd_1``, …) so LangGraph's ``add_messages`` replaces it in-place
        on subsequent turns rather than appending a fresh copy each time.
        """
        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt, id="sys_prompt")]
        for idx, (role, content) in enumerate(self.begin_dialogs):
            stable_id = f"bd_{idx}"
            if role == "user":
                messages.append(HumanMessage(content=content, id=stable_id))
            elif role == "assistant":
                messages.append(AIMessage(content=content, id=stable_id))
            else:
                # Treat any other role as a system-level instruction.
                messages.append(SystemMessage(content=content, id=stable_id))

        if self._provider_supports_cache_control() and messages:
            messages[-1].additional_kwargs.update(self._cache_control_kwargs())

        return messages

    def build_state_modifier(self) -> list[SystemMessage]:
        """Static system message used as the graph state modifier.

        This is intentionally only the system prompt. Begin-dialogs are kept
        in the input stream so the same message-order logic works for both
        ``create_react_agent`` and DeepAgents.
        """
        prefix = self._static_prefix()
        return [prefix[0]] if prefix else []

    def build_input_messages(
        self,
        memory_text: str,
        user_message: str,
        kg_text: str = "",
        history_messages: list[BaseMessage] | None = None,
        kb_text: str = "",
    ) -> list[BaseMessage]:
        """Messages placed after the system prompt.

        Order: begin-dialogs (static), conversation history (dynamic),
        optional memory context (dynamic),
        optional project knowledge graph context (dynamic),
        optional domain knowledge base context (dynamic),
        current user message (dynamic).

        Dynamic context messages (memory, KG, KB) carry stable IDs so
        LangGraph's ``add_messages`` replaces them each turn instead of
        accumulating unbounded duplicates in the graph state.
        """
        prefix = self._static_prefix()
        messages: list[BaseMessage] = list(prefix[1:])

        if history_messages:
            messages.extend(history_messages)

        if memory_text:
            messages.append(SystemMessage(content=memory_text, id="ctx_memory"))

        if kg_text:
            messages.append(SystemMessage(content=kg_text, id="ctx_kg"))

        if kb_text:
            messages.append(SystemMessage(content=kb_text, id="ctx_kb"))

        messages.append(HumanMessage(content=user_message))
        return messages

    def build_full_messages(
        self,
        memory_text: str,
        user_message: str,
        kg_text: str = "",
        kb_text: str = "",
    ) -> list[BaseMessage]:
        """Convenience: full message list for one-shot callers."""
        return self.build_state_modifier() + self.build_input_messages(
            memory_text, user_message, kg_text=kg_text, kb_text=kb_text
        )
