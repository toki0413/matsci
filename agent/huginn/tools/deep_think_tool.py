"""Deep-think tool — external scratchpad for reasoning.

oh-my-pi's externalThinking idea: providers hide native chain-of-thought, but
a plain tool call's arguments are returned to the developer over the API. This
tool lets the model write its analysis *before* acting, so that analysis lands
in ``session.reasoning_trace`` (the same channel native ``reasoning_content``
feeds) and can be consumed by the distillation / evolution loop downstream.

It is a *complement* to native reasoning, not a replacement: when the model
exposes ``reasoning_content`` we keep capturing that too; ``deep_think`` is the
explicit scratchpad that works even when the provider does not expose native
thinking.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from huginn.core_types import ToolContext, ToolResult
from huginn.tools.base import HuginnTool

logger = logging.getLogger(__name__)


class DeepThinkToolInput(BaseModel):
    analysis: str = Field(
        ...,
        description=(
            "The step-by-step analysis / reasoning you worked through before "
            "answering, modifying code, or calling other tools."
        ),
    )


class DeepThinkTool(HuginnTool):
    """Write the current analysis to the external scratchpad (reasoning trace)."""

    name = "deep_think"
    category = "meta"
    description = (
        "Use this BEFORE answering, modifying code, or calling other tools. "
        "Write down your step-by-step analysis and reasoning here. This is an "
        "external scratchpad: your analysis is recorded and returned to the "
        "developer, but not echoed back as part of your visible answer."
    )
    input_schema = DeepThinkToolInput
    destructive = False
    read_only = True

    def is_read_only(self, args: DeepThinkToolInput) -> bool:
        return True

    async def _execute(self, args: Any, context: ToolContext) -> ToolResult:
        if isinstance(args, dict):
            args = DeepThinkToolInput(**args)
        analysis = (args.analysis or "").strip()
        if not analysis:
            return ToolResult(
                data=None,
                success=False,
                error="deep_think requires a non-empty 'analysis'",
            )

        # Best-effort: record into the shared reasoning_trace channel so the
        # distillation / evolution loop can consume it. Fail-open when there is
        # no memory manager (e.g. some bench harnesses) — never block the loop.
        try:
            if context.memory_manager is not None:
                context.memory_manager.add_reasoning(analysis)
            else:
                logger.debug(
                    "deep_think called without memory_manager; analysis not recorded"
                )
        except Exception:
            logger.warning("deep_think failed to record reasoning", exc_info=True)

        return ToolResult(
            data={"recorded": True, "chars": len(analysis)},
            success=True,
        )
