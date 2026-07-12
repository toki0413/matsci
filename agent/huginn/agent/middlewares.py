"""Agent middleware: dangling-tool-call repair and rate limiting."""

from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, ToolMessage

from huginn.utils.session_context import get_thread_id

logger = logging.getLogger(__name__)


class FixDanglingToolCallsMiddleware(AgentMiddleware):
    """Patch orphan tool_calls left behind by summarization compaction.

    deepagents' built-in PatchToolCallsMiddleware only runs before_agent,
    but summarization can drop ToolMessages mid-turn, leaving orphan
    AIMessage.tool_calls that make DeepSeek reject with 400.  This
    middleware patches at the wrap_model_call layer instead.
    """

    def _patch_messages(self, messages: list) -> list:
        if not messages:
            return messages
        answered_ids = {
            getattr(msg, "tool_call_id", None)
            for msg in messages
            if hasattr(msg, "type") and msg.type == "tool"
        }
        has_orphan = any(
            tc.get("id") is not None and tc["id"] not in answered_ids
            for msg in messages
            if isinstance(msg, AIMessage)
            for tc in (*msg.tool_calls, *getattr(msg, "invalid_tool_calls", []))
        )
        if not has_orphan:
            return messages
        patched = list(messages)
        for msg in patched:
            if not isinstance(msg, AIMessage):
                continue
            for tc in (*msg.tool_calls, *getattr(msg, "invalid_tool_calls", [])):
                tc_id = tc.get("id")
                if tc_id is None or tc_id in answered_ids:
                    continue
                name = tc.get("name") or "unknown"
                content = (
                    f"Tool call {name} (id={tc_id}) was cancelled — "
                    f"summarization compaction removed its result."
                )
                patched.append(
                    ToolMessage(content=content, name=name, tool_call_id=tc_id)
                )
                answered_ids.add(tc_id)
        return patched

    def wrap_model_call(self, request, handler):
        request.messages = self._patch_messages(request.messages)
        return handler(request)

    async def awrap_model_call(self, request, handler):
        request.messages = self._patch_messages(request.messages)
        return await handler(request)

    # Also patch at before_agent — _get_model_input_state runs before
    # wrap_model_call, so we need to fix orphan tool calls earlier.
    def before_agent(self, request, handler=None):
        if hasattr(request, 'messages') and request.messages:
            request.messages = self._patch_messages(request.messages)
        return handler(request) if handler else None

    async def abefore_agent(self, request, handler=None):
        if hasattr(request, 'messages') and request.messages:
            request.messages = self._patch_messages(request.messages)
        return await handler(request) if handler else None

    # deepagents middleware protocol requires all four methods.
    # Tool-call layer doesn't need orphan patching — passthrough.
    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)


class RateLimitMiddleware(AgentMiddleware):
    """Token-rate limiter at the model_call layer.

    Guards against runaway generation by checking before each model
    call and recording usage from the returned AIMessage afterwards.
    """

    def __init__(self) -> None:
        from huginn.security.rate_limiter import get_rate_limiter

        self._limiter = get_rate_limiter()

    def _estimate_tokens(self, messages: list) -> int:
        """Rough token estimate, ~4 chars/token."""
        total = 0
        for msg in messages or []:
            content = getattr(msg, "content", None) or str(msg)
            if not isinstance(content, str):
                content = str(content)
            total += len(content)
        return max(total // 4, 1)

    def _extract_usage(self, result: Any) -> tuple[int, int]:
        from huginn.security.rate_limiter import _extract_usage as _extract

        return _extract(result)

    def wrap_model_call(self, request, handler):
        _tid = get_thread_id() or "default"
        ok, reason = self._limiter.check_allowed(
            "agent", self._estimate_tokens(getattr(request, "messages", [])),
            thread_id=_tid,
        )
        if not ok:
            from huginn.security.rate_limiter import RateLimitExceeded

            raise RateLimitExceeded(reason, reason="limit_exceeded")
        result = handler(request)
        in_tok, out_tok = self._extract_usage(result)
        self._limiter.record_usage("agent", in_tok, out_tok, thread_id=_tid)
        return result

    async def awrap_model_call(self, request, handler):
        _tid = get_thread_id() or "default"
        ok, reason = self._limiter.check_allowed(
            "agent", self._estimate_tokens(getattr(request, "messages", [])),
            thread_id=_tid,
        )
        if not ok:
            from huginn.security.rate_limiter import RateLimitExceeded

            raise RateLimitExceeded(reason, reason="limit_exceeded")
        result = await handler(request)
        in_tok, out_tok = self._extract_usage(result)
        self._limiter.record_usage("agent", in_tok, out_tok, thread_id=_tid)
        return result

    def wrap_tool_call(self, request, handler):
        return handler(request)

    async def awrap_tool_call(self, request, handler):
        return await handler(request)
