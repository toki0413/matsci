"""Context-window budgeting and compaction helpers."""

from __future__ import annotations

from typing import Any

from huginn.utils.tokens import rough_token_count_for_text


def estimate_message_tokens(messages: list[Any]) -> int:
    """Rough token estimate for a list of LangChain-like messages."""
    total = 0
    for msg in messages:
        content = ""
        if isinstance(msg, dict):
            content = msg.get("content") or ""
        elif hasattr(msg, "content"):
            content = msg.content or ""
        if isinstance(content, list):
            content = "\n".join(str(block) for block in content)
        total += rough_token_count_for_text(str(content))
    return total


def compact_messages(
    messages: list[Any],
    budget_tokens: int,
    keep_last_n: int = 2,
) -> list[Any]:
    """Drop oldest messages until the remaining list fits the token budget.

    Always preserves the last `keep_last_n` messages (typically the latest user
    message and any system prompt already injected by the caller).
    """
    if budget_tokens <= 0:
        return messages

    trimmed = list(messages)
    while len(trimmed) > keep_last_n and estimate_message_tokens(trimmed) > budget_tokens:
        # Drop the oldest message that is not within the keep-last window.
        trimmed.pop(0)
    return trimmed
