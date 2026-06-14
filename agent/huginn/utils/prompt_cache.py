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

    * ``state_modifier`` - the static system prompt (used as a LangGraph
      ``state_modifier`` or as the ``system_prompt`` for DeepAgents).
    * ``input_messages`` - begin-dialogs + optional memory context + the
      current user message. Begin-dialogs are static, but they live in the
      input stream so that a changing memory/user tail does not invalidate
      the cached prefix that precedes it.
    """

    def __init__(
        self,
        system_prompt: str,
        begin_dialogs: list[tuple[str, str]] | None = None,
        cache_control: bool = False,
    ):
        self.system_prompt = system_prompt
        self.begin_dialogs = begin_dialogs or []
        self.cache_control = cache_control

    def _static_prefix(self) -> list[BaseMessage]:
        """Return system prompt + begin-dialogs as a single static prefix.

        When ``cache_control`` is enabled, the *last* static message is
        tagged so the provider can cache everything up to that point.
        """
        messages: list[BaseMessage] = [SystemMessage(content=self.system_prompt)]
        for role, content in self.begin_dialogs:
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            else:
                # Treat any other role as a system-level instruction.
                messages.append(SystemMessage(content=content))

        if self.cache_control and messages:
            messages[-1].additional_kwargs["cache_control"] = {"type": "ephemeral"}

        return messages

    def build_state_modifier(self) -> list[SystemMessage]:
        """Messages that should live in the graph's state modifier.

        This is just the static system prompt. Begin-dialogs are kept in the
        input stream so that the same message-order logic works for both
        ``create_react_agent`` and DeepAgents.
        """
        prefix = self._static_prefix()
        return [prefix[0]] if prefix else []

    def build_input_messages(
        self,
        memory_text: str,
        user_message: str,
    ) -> list[BaseMessage]:
        """Messages placed after the system prompt.

        Order: begin-dialogs (static), optional memory context (dynamic),
        current user message (dynamic).
        """
        prefix = self._static_prefix()
        messages: list[BaseMessage] = list(prefix[1:])

        if memory_text:
            messages.append(SystemMessage(content=memory_text))

        messages.append(HumanMessage(content=user_message))
        return messages

    def build_full_messages(
        self,
        memory_text: str,
        user_message: str,
    ) -> list[BaseMessage]:
        """Convenience: full message list for one-shot callers."""
        return self.build_state_modifier() + self.build_input_messages(
            memory_text, user_message
        )
