"""Hook wrapping and tool-call callback management."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from typing import Any

from huginn.hooks import HookContext, HookManager
from huginn.utils.session_context import get_thread_id, get_user_message

logger = logging.getLogger(__name__)


class CallbackMixin:
    """Hook registration, tool wrapping, and approval callbacks."""

    def set_approval_callback(
        self, callback: Callable[[str, str], bool] | None
    ) -> None:
        """Set the interactive approval callback.

        Must be called before register_tools_from_registry so the
        callback is captured by each adapted tool.  Calling it after
        tools are already registered has no effect on those tools.
        """
        self._approval_callback = callback

    def set_style_learner(self, learner: Any | None) -> None:
        """Inject a StyleLearner for personalised communication style.

        Also registers to the personalization module's shared singleton
        so tools and routes see the same learner instance.  Pass None
        to disable.
        """
        self.style_learner = learner
        if learner is not None:
            from huginn.personalization import set_shared_style_learner

            set_shared_style_learner(learner)

    def register_hook(self, event: str, callback: Any) -> None:
        """Register a hook for a given event (PRE_TOOL_USE, etc.)."""
        self.hook_manager.register(event, callback)

    def _wrap_tool_with_hooks(self, tool: Any) -> Any:
        """Wrap a LangChain StructuredTool with pre/post hooks.

        Async path runs the full hook flow; sync path runs hooks only
        when there's no running event loop, otherwise forwards directly
        to avoid nested-loop errors.
        """
        from langchain_core.tools import StructuredTool

        original = tool
        tool_name = getattr(original, "name", "")
        hm = self.hook_manager

        async def _invoke_with_hooks(input_data: dict) -> Any:
            allowed, modified, pre_ctx = await hm.run_pre(
                tool_name,
                input_data,
                thread_id=get_thread_id() or getattr(self, "thread_id", None),
            )
            if not allowed:
                reason = pre_ctx.metadata.get("block_reason") if pre_ctx else None
                return {
                    "error": "blocked by pre_tool_use hook",
                    "block_reason": reason or "blocked by pre_tool_use hook",
                }
            if isinstance(modified, dict):
                input_data = modified
            # Scheduler admission: gate every tool call by cost_tier so
            # the heavy/light semaphores arbitrate concurrency.
            admission = None
            sched = getattr(self, "scheduler", None)
            if sched is not None:
                cost_tier, cost = self._scheduler_cost(tool_name, input_data)
                try:
                    admission = await sched.acquire(tool_name, cost_tier, cost)
                except Exception as exc:
                    return {
                        "error": "resource_exhausted",
                        "block_reason": str(exc),
                    }
            start = time.time()
            error: BaseException | None = None
            result: Any = None
            try:
                result = await original.ainvoke(input_data)
            except Exception as exc:
                error = exc
                if admission is not None and sched is not None:
                    try:
                        sched.release(admission)
                    except Exception:
                        logger.warning("scheduler release failed for %s", tool_name)
                duration_ms = (time.time() - start) * 1000
                await hm.run_post(
                    tool_name, input_data, result, error, duration_ms,
                    thread_id=get_thread_id() or getattr(self, "thread_id", None),
                    user_message=get_user_message() or getattr(self, "_current_user_message", None),
                )
                raise

            if admission is not None and sched is not None:
                try:
                    sched.release(admission)
                except Exception:
                    logger.warning("scheduler release failed for %s", tool_name)
            duration_ms = (time.time() - start) * 1000
            post_ctx = await hm.run_post(
                tool_name, input_data, result, None, duration_ms,
                thread_id=get_thread_id() or getattr(self, "thread_id", None),
                user_message=get_user_message() or getattr(self, "_current_user_message", None),
            )
            if post_ctx and post_ctx.metadata.get("blocked_by_hook"):
                block_reason = post_ctx.metadata.get("block_reason", "blocked by post_tool_use hook")
                return {"error": block_reason, "_hook_blocked": True}
            return result

        async def hooked_coroutine(**kwargs: Any) -> Any:
            return await _invoke_with_hooks(kwargs)

        def hooked_func(**kwargs: Any) -> Any:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                return asyncio.run(_invoke_with_hooks(kwargs))
            return original.invoke(kwargs)

        return StructuredTool.from_function(
            name=tool_name or "unnamed_tool",
            description=getattr(original, "description", "") or "",
            args_schema=getattr(original, "args_schema", None),
            coroutine=hooked_coroutine,
            func=hooked_func,
            return_direct=getattr(original, "return_direct", False),
        )

    def _scheduler_cost(
        self, tool_name: str, input_data: dict
    ) -> tuple[str, dict[str, float] | None]:
        """Best-effort (cost_tier, estimate_cost) for scheduler admission.

        Looks the tool up in the live ToolRegistry.  Unknown tools or
        estimate_cost failures return ("none", None) so the call is
        admitted without gating.
        """
        try:
            from huginn.tools.registry import ToolRegistry

            t = ToolRegistry.get(tool_name)
            if t is None:
                return "none", None
            cost_tier = t.cost_tier
            try:
                cost = t.estimate_cost(input_data)
            except Exception:
                cost = None
            return cost_tier, cost
        except Exception:
            return "none", None
