"""Context-window budgeting and compaction helpers.

Provides both drop-oldest (fast fallback) and summarization-based
(smart) compaction strategies. The summarization strategy sends old
messages to a lightweight LLM to produce a concise summary, preserving
research context that would otherwise be lost.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from huginn.utils.tokens import count_message_tokens, count_tokens

logger = logging.getLogger(__name__)

# Role names that must never be summarized or dropped.
_PROTECTED_ROLES = {"system"}


def _msg_role(msg: Any) -> str:
    if isinstance(msg, dict):
        return msg.get("role", "")
    return getattr(msg, "type", "") or getattr(msg, "role", "")


def _msg_content(msg: Any) -> str:
    if isinstance(msg, dict):
        content = msg.get("content") or ""
    else:
        content = getattr(msg, "content", "") or ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(str(block.get("text", "")))
        return "\n".join(parts)
    return str(content)


def estimate_message_tokens(messages: list[Any]) -> int:
    """Accurate token estimate for a list of LangChain-like messages.

    Uses tiktoken when available, falling back to the character heuristic.
    """
    total = 0
    for msg in messages:
        role = _msg_role(msg)
        content = _msg_content(msg)
        total += count_message_tokens(content, role)
    return total


def compact_messages(
    messages: list[Any],
    budget_tokens: int,
    keep_last_n: int = 2,
) -> list[Any]:
    """Drop oldest messages until the remaining list fits the token budget.

    Always preserves the last `keep_last_n` messages. This is the fast
    fallback when no summarizer is available.
    """
    if budget_tokens <= 0:
        return messages

    # Token counts per message — computed once so we don't rescan the
    # whole list on every iteration (the old pop-and-recount loop was O(n^2)).
    per_msg_tokens = [
        count_message_tokens(_msg_content(m), _msg_role(m)) for m in messages
    ]
    total = sum(per_msg_tokens)
    if total <= budget_tokens:
        return list(messages)

    # Walk from the front, subtracting each message's tokens, until we're
    # under budget or we'd dip below keep_last_n.
    max_droppable = max(0, len(messages) - keep_last_n)
    drop_count = 0
    while drop_count < max_droppable and total > budget_tokens:
        total -= per_msg_tokens[drop_count]
        drop_count += 1

    return list(messages[drop_count:])


def _format_messages_for_summary(messages: list[Any]) -> str:
    """Render messages into a compact transcript for the summarizer LLM."""
    lines = []
    for msg in messages:
        role = _msg_role(msg)
        content = _msg_content(msg)
        # Truncate very long tool outputs to keep the summarizer prompt small
        if len(content) > 2000:
            content = content[:1800] + "\n[...truncated...]"
        label = role.upper() if role else "MESSAGE"
        lines.append(f"[{label}]\n{content}")
    return "\n\n".join(lines)


_SUMMARY_SYSTEM = (
    "You are a research conversation summarizer. Condense the following "
    "conversation excerpt into a concise summary that preserves:\n"
    "1. Key decisions and their rationale\n"
    "2. Important numerical results (energies, parameters, convergence criteria)\n"
    "3. Failed approaches and why they failed\n"
    "4. Pending tasks and next steps\n"
    "5. Any file paths, structure IDs, or job IDs referenced\n"
    "Be terse — use bullet points. Do not include greetings or filler."
)

# The accumulated summary is fed back into the summarizer each round, so
# without a cap it grows without bound. Past this many tokens we compress
# it back down before reusing it.
_SUMMARY_TOKEN_CAP = 2000


async def summarize_compact_messages(
    messages: list[Any],
    budget_tokens: int,
    keep_last_n: int = 4,
    summarizer: Callable[[str], Any] | None = None,
    existing_summary: str = "",
) -> tuple[list[Any], str]:
    """Compact messages via LLM summarization, preserving research context.

    Returns ``(compacted_messages, summary_text)``. The summary_text should
    be passed as ``existing_summary`` on the next call so the summary
    accumulates across compaction rounds.

    Strategy:
    1. Split messages into [summarize_zone | keep_zone].
    2. If a summarizer is available, send summarize_zone to it and replace
       with a single SystemMessage carrying the summary.
    3. If no summarizer, fall back to ``compact_messages`` (drop-oldest).

    Args:
        messages: Full message list (system prompt should already be excluded
            — it's managed by PromptCacheBuilder).
        budget_tokens: Target token budget for the compacted list.
        keep_last_n: Minimum messages to always preserve at the tail.
        summarizer: Async callable that takes a transcript string and returns
            the LLM's summary text. If None, falls back to drop-oldest.
        existing_summary: Summary from a previous compaction round, to
            accumulate context.
    """
    if budget_tokens <= 0:
        return messages, existing_summary

    current_tokens = estimate_message_tokens(messages)
    if current_tokens <= budget_tokens:
        return messages, existing_summary

    # Determine how many messages to summarize vs keep.
    # Target: summarize enough to get under budget, keeping at least keep_last_n.
    keep_zone = messages[-keep_last_n:] if len(messages) > keep_last_n else messages[:]
    summarize_zone = messages[:-keep_last_n] if len(messages) > keep_last_n else []

    # Filter out protected (system) messages from the summarize zone —
    # they're managed separately by the prompt cache builder.
    to_summarize = [m for m in summarize_zone if _msg_role(m) not in _PROTECTED_ROLES]
    protected = [m for m in summarize_zone if _msg_role(m) in _PROTECTED_ROLES]

    if not to_summarize or summarizer is None:
        # No summarizer or nothing to summarize — fall back to drop-oldest
        return compact_messages(messages, budget_tokens, keep_last_n), existing_summary

    # Build the transcript for the summarizer
    transcript = _format_messages_for_summary(to_summarize)
    if existing_summary:
        transcript = (
            f"## Previous summary:\n{existing_summary}\n\n"
            f"## New conversation to incorporate:\n{transcript}"
        )

    try:
        result = await summarizer(transcript)
        # Extract text from various LLM response types
        if hasattr(result, "content"):
            summary_text = result.content
        elif isinstance(result, dict):
            summary_text = result.get("content", str(result))
        else:
            summary_text = str(result)
    except Exception as exc:
        logger.warning("Summarization failed (%s), falling back to drop-oldest", exc)
        return compact_messages(messages, budget_tokens, keep_last_n), existing_summary

    # The summary carries forward across rounds, so keep it bounded. If the
    # accumulated text blows past the cap, run it back through the summarizer
    # with a compression prompt to trim it down.
    if count_tokens(summary_text) > _SUMMARY_TOKEN_CAP:
        try:
            compress_prompt = (
                "Compress this research summary, keeping only the most "
                "critical findings and decisions:\n\n" + summary_text
            )
            compressed = await summarizer(compress_prompt)
            if hasattr(compressed, "content"):
                summary_text = compressed.content
            elif isinstance(compressed, dict):
                summary_text = compressed.get("content", str(compressed))
            else:
                summary_text = str(compressed)
        except Exception as exc:
            logger.warning("Summary re-compression failed (%s), keeping original", exc)

    # Build the summary message
    from langchain_core.messages import SystemMessage

    summary_msg = SystemMessage(
        content=f"## Conversation summary (older messages compacted):\n{summary_text}"
    )

    # Assemble: [protected system msgs] + [summary] + [keep_zone]
    compacted = protected + [summary_msg] + keep_zone

    # If still over budget, recursively drop oldest from the keep_zone
    if estimate_message_tokens(compacted) > budget_tokens and len(compacted) > keep_last_n:
        compacted = compact_messages(compacted, budget_tokens, keep_last_n)

    logger.info(
        "Compacted %d messages → %d (summary: %d tokens, total: %d → %d)",
        len(messages),
        len(compacted),
        count_tokens(summary_text),
        current_tokens,
        estimate_message_tokens(compacted),
    )

    # Belief Entropy: 压缩后算自检信号, 给下一轮压缩调参
    try:
        from huginn.utils.belief_entropy import get_belief_entropy
        be = get_belief_entropy()
        be._last_result = be.measure(
            summary=summary_text,
            original_tokens=current_tokens,
            compressed_tokens=estimate_message_tokens(compacted),
        )
    except Exception:
        pass

    return compacted, summary_text
