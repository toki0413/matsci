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
from huginn.memory.reasoning import ReasoningPhase, ReasoningRecord
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
    # 结构化推理协议 (可选增强, 不填则 fallback 到 analysis 全文):
    # 让推理可被自校验/蒸馏/阶段化追溯消费. 模型可只填 analysis, 字段全可选.
    phase: str = Field(
        default="think",
        description=(
            "Reasoning stage: 'think' (hypothesis/claim), 'plan' (action plan), "
            "'pre_action' (quantitative prediction before acting), "
            "'reflect' (post-execution review)."
        ),
    )
    hypothesis: str = Field(
        default="",
        description="The core claim / hypothesis you are asserting.",
    )
    evidence: str = Field(
        default="",
        description="The evidence / derivation supporting the claim.",
    )
    estimate: str = Field(
        default="",
        description="A quantitative prediction (with units/range) to check later.",
    )
    uncertainty: str = Field(
        default="",
        description="Known uncertainties and boundary conditions of the claim.",
    )
    plan: str = Field(
        default="",
        description="The next concrete action(s) your reasoning leads to.",
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

        # 1) 扁平通道 (旧, 向后兼容) — 原生 reasoning_content 也走这里.
        try:
            if context.memory_manager is not None:
                context.memory_manager.add_reasoning(analysis)
        except Exception:
            logger.warning("deep_think failed to record reasoning_trace", exc_info=True)

        # 2) 结构化侧信道 (external thinking 深化) — 模型可选填结构化字段,
        #    不填时以 analysis 全文作为 claim.
        try:
            phase = (
                ReasoningPhase(args.phase)
                if args.phase in ReasoningPhase._value2member_map_
                else ReasoningPhase.THINK
            )
            record = ReasoningRecord(
                claim=args.hypothesis or analysis,
                phase=phase,
                evidence=args.evidence,
                estimate=args.estimate,
                uncertainty=args.uncertainty,
                plan=args.plan,
            )
            if context.memory_manager is not None:
                context.memory_manager.add_reasoning_record(record)
        except Exception:
            logger.warning("deep_think failed to record structured reasoning", exc_info=True)

        return ToolResult(
            data={"recorded": True, "chars": len(analysis)},
            success=True,
        )
