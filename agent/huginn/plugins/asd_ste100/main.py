"""ASD-STE100 plugin — controlled-language discipline for agent-facing text.

Brings the ASD-STE100 Simplified Technical English discipline (danyuchn/
asd-ste100-skill, MIT) into Huginn.  STE exists so a reader with no back-channel
cannot misread an instruction.  An agent that parses another agent's tool
description, error message, or inter-agent instruction has the same problem.

Two live surfaces (both wired into the real pipeline, not just declared):

1. A prompt segment — registered into ``huginn.plugins.prompt_segments``, the
   synchronous registry that ``build_prompt`` actually assembles from.  The
   segment injects the STE rules into the system prompt, scoped by mode.
2. A ``ste_lint`` tool — registered into ``ToolRegistry``, so the agent can
   check its own draft text against the deterministic structural rules before
   it sends the text to another agent or tool.
3. A ``comms_lint`` tool + an advisory audit on the event bus — the same
   discipline (one word one meaning, explicit envelope, stated contract)
   applied to *structured* communication: module events, workflow handoffs,
   and agent-to-agent messages.  See ``huginn/comms/contract.py``.

Modes (env ``HUGINN_STE_MODE``, default ``agents``):
  agents   — rules scoped to tool descriptions, error messages, status reports,
             and inter-agent instructions.  Does not constrain scientific prose.
  strict   — full STE structural discipline for all instruction-like output.
  flavored — structural rules for prose; lexical rules advisory only.
  off      — no injection (the linter tool stays available).

The feature flag ``asd_ste100`` (HUGINN_FEATURE_ASD_STE100) is the master switch.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from huginn.api.context import PluginContext
from huginn.api.star import Star
from huginn.core_types import ToolContext, ToolResult
from huginn.plugins.asd_ste100.ste_lint import summarize
from huginn.plugins.prompt_segments import (
    register_prompt_segment,
    unregister_prompt_segment,
)
from huginn.tools.base import HuginnTool
from huginn.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

SEGMENT_NAME = "asd_ste100"
TOOL_NAME = "ste_lint"
COMMS_TOOL_NAME = "comms_lint"
# Between writing (60) and thinking (100): the agent's voice rules come first,
# then the discipline for the strings it emits to other agents.
SEGMENT_PRIORITY = 65

_VALID_MODES = ("agents", "strict", "flavored", "off")

# ── Rule text ────────────────────────────────────────────────────────
# Adapted from danyuchn/asd-ste100-skill SKILL.md (MIT).  Trimmed so the
# always-on block stays small in the system prompt.

_STRUCTURAL = """\
- Use active voice and name the actor: "The tool writes the file." Not "The file is written."
- Write one instruction per sentence.
- Keep each sentence at or below 25 words (20 for instructions).
- Do not use semicolons. Write two sentences.
- Do not use phrasal verbs (spin up, reach out, dive into). Use start, contact, read.
- Describe an action with its verb: "analyze the log", not "perform an analysis of the log".
- Delete marketing adjectives (seamless, robust, cutting-edge, effortless). State the measurement that earns the claim.
- Keep every hedge (may, could, sometimes). Confidence is content. Never upgrade a hedge to a fact.
- Keep necessary technical terms. Define a term once when it is not common English."""

_COMMS = """\
- Keep one field name for one meaning everywhere: tool_name, not tool. session_id, not sid. source, not src.
- Put the parts a receiver needs in every message: type, timestamp, source.
- Name each event with a lowercase dotted namespace (tool.result, team.member.start). Do not invent a new name when a registered one fits.
- A success result carries no error. A failed result states the error. Never report both at once."""

_BLOCK_AGENTS = f"""\
## Agent-facing text (ASD-STE100)

The strings you send to other agents and tools must survive a parser that cannot ask a follow-up question. Apply these rules to tool descriptions, error messages, status reports, and inter-agent instructions. Do not apply them to scientific prose or creative writing.

{_STRUCTURAL}

Structured messages (events, workflow handoffs, agent-to-agent calls) follow the same discipline:

{_COMMS}"""

_BLOCK_STRICT = f"""\
## Output language (ASD-STE100 Strict)

Write all instructions, error messages, tool descriptions, and status reports in controlled English. One word has one meaning. One sentence has one instruction.

{_STRUCTURAL}
- Use simple tenses only: infinitive, imperative, simple present, simple past, simple future, and past participle as an adjective.
- Do not write present perfect ("we have received"). Write simple past ("we received"). Keep the compound form only when current relevance is the point, and flag it.
- Stack no more than three words in a noun phrase.
- Keep the subject, verb, and article explicit. Do not drop words to shorten the sentence.
- Limit a paragraph to one topic and six sentences.
- Use a numbered or bulleted list for three or more steps or conditions.
- Pick one word for one action and reuse it everywhere. Do not rotate check/verify/confirm.
- Apply the same one-word-one-meaning rule to structured messages:

{_COMMS}"""

_BLOCK_FLAVORED = f"""\
## Output language (ASD-STE100 flavored)

Keep the sentence discipline for explanatory text: short sentences, active voice, simple tenses, no phrasal verbs, no semicolons, no nominalization, no marketing adjectives. Treat the one-word-one-meaning rule as advice, not a lockdown, because prose needs range.

{_STRUCTURAL}"""

_BLOCKS = {
    "agents": _BLOCK_AGENTS,
    "strict": _BLOCK_STRICT,
    "flavored": _BLOCK_FLAVORED,
}


def current_mode() -> str:
    """The active STE mode, read from ``HUGINN_STE_MODE`` each call."""
    mode = os.environ.get("HUGINN_STE_MODE", "agents").strip().lower()
    return mode if mode in _VALID_MODES else "agents"


def set_mode(mode: str) -> str:
    """Set the STE mode for this process. Returns the mode actually applied."""
    mode = (mode or "").strip().lower()
    if mode not in _VALID_MODES:
        raise ValueError(f"unknown STE mode {mode!r}; use one of {_VALID_MODES}")
    os.environ["HUGINN_STE_MODE"] = mode
    return mode


def _enabled() -> bool:
    """Master switch. Fail-closed: a broken flag layer disables injection."""
    try:
        from huginn.feature_flags import FeatureFlags

        return FeatureFlags.shared().is_enabled("asd_ste100")
    except Exception:
        logger.debug("asd_ste100 feature flag unavailable; injection disabled", exc_info=True)
        return False


def ste_prompt_segment(mode: str, phase: str, metacog_state: str, system_prompt: str | None) -> str:
    """Prompt segment: return the STE rules block for the active mode.

    Signature matches ``prompt_segments.PromptSegmentFn``.  Returns an empty
    string when injection is off, so ``assemble_prompt_segments`` skips it.
    """
    ste_mode = current_mode()
    if ste_mode == "off" or not _enabled():
        return ""
    return _BLOCKS.get(ste_mode, _BLOCK_AGENTS)


# ── Tool ─────────────────────────────────────────────────────────────

class SteLintInput(BaseModel):
    """Input for the STE structural linter."""

    text: str | None = Field(
        default=None,
        description="The English text to check. Give this or path.",
    )
    path: str | None = Field(
        default=None,
        description="Path to a text file to check. Use only when text is not given.",
    )


class SteLintTool(HuginnTool):
    """Check English text against the ASD-STE100 structural rules.

    The check is deterministic and uses no model.  It reports sentence length,
    semicolons, phrasal verbs, nominalization, marketing adjectives, synonym
    rotation, and dangling list conjunctions.  Passive voice and compound
    tenses are advisory.  Hedges (may, could) are never flagged.
    """

    name = TOOL_NAME
    description = (
        "Check English text against the ASD-STE100 structural rules. The check "
        "is deterministic and uses no model. Use it on tool descriptions, error "
        "messages, status reports, and inter-agent messages before you send "
        "them. It reports hard violations and advisory findings. It never flags "
        "hedges (may, could), because confidence is content."
    )
    category = "meta"
    read_only = True
    destructive = False
    input_schema = SteLintInput

    async def _execute(self, args: SteLintInput, context: ToolContext) -> ToolResult:
        text = args.text
        filename = "<text>"
        if not text and args.path:
            filename = args.path
            try:
                text = Path(args.path).read_text(encoding="utf-8")
            except OSError as exc:
                return ToolResult(data=None, success=False, error=f"cannot read {args.path}: {exc}")
        if not text:
            return ToolResult(
                data=None,
                success=False,
                error="give text or path to check",
            )

        result = summarize(text, filename=filename)
        hard = [v for v in result["violations"] if v["level"] == "advisory-free"]
        lines = [f"{filename}: {result['hard_count']} hard, "
                 f"{result['count'] - result['hard_count']} advisory "
                 f"in {result['words']} words"]
        for v in result["violations"][:20]:
            lines.append(f"  line {v['line']}:{v['col']} [{v['level']}] {v['rule']}: "
                         f"{v['match']} — {v['message']}")
        if len(result["violations"]) > 20:
            lines.append(f"  ... and {len(result['violations']) - 20} more")
        if not result["violations"]:
            lines.append("  no structural violations")
        result["summary"] = "\n".join(lines)
        result["hard_violations"] = hard
        return ToolResult(data=result, success=True)


class CommsLintInput(BaseModel):
    """Input for the structured-communication contract linter."""

    payload: dict[str, Any] = Field(
        description=(
            "The structured message to check. An AgentEvent ({type,timestamp,"
            "source,data}), a SubagentResult ({summary,success,error}), an "
            "AgentMessage ({role,content}), or a workflow event "
            "({workflow_name,stage_name})."
        ),
    )
    surface: str | None = Field(
        default=None,
        description="event | workflow | message | result. Omit to infer from the fields.",
    )


class CommsLintTool(HuginnTool):
    """Check a structured message against the communication contract.

    Deterministic, no model.  Applies the ASD-STE100 discipline to *structured*
    communication: envelope completeness, event-name vocabulary, declared
    payload fields, field-name aliases, and result self-consistency.  Reports
    hard and advisory violations.  See ``huginn/comms/contract.py``.
    """

    name = COMMS_TOOL_NAME
    description = (
        "Check a structured message against the communication contract. The "
        "check is deterministic and uses no model. Use it on events, workflow "
        "handoffs, and agent-to-agent messages before you publish or send them. "
        "It checks the envelope (type/timestamp/source), the event-name "
        "vocabulary, declared payload fields, field-name aliases such as tool "
        "for tool_name, and result self-consistency. It reports hard and "
        "advisory violations."
    )
    category = "meta"
    read_only = True
    destructive = False
    input_schema = CommsLintInput

    async def _execute(self, args: CommsLintInput, context: ToolContext) -> ToolResult:
        from huginn.comms.contract import lint

        report = lint(args.payload, surface=(args.surface or None))
        data = report.to_dict()
        lines = [f"{report.surface}: {report.hard_count} hard, "
                 f"{report.advisory_count} advisory"]
        for v in report.violations[:20]:
            lines.append(f"  [{v.severity}] {v.rule} ({v.field}): {v.message}")
        if not report.violations:
            lines.append("  contract satisfied")
        data["summary"] = "\n".join(lines)
        return ToolResult(data=data, success=True)


# ── Plugin ───────────────────────────────────────────────────────────

class AsdSte100Star(Star):
    """Mounts the STE prompt segment and the lint tool for this process."""

    name = SEGMENT_NAME
    author = "Huginn Integration (port of danyuchn/asd-ste100-skill, MIT)"
    version = "1.1.0"
    description = "ASD-STE100 controlled-language discipline for agent-facing text and structured communication"
    priority = 60

    def __init__(self, context: PluginContext | None = None) -> None:
        super().__init__(context)
        self._tool: SteLintTool | None = None
        self._comms_tool: CommsLintTool | None = None

    async def on_load(self) -> None:
        register_prompt_segment(SEGMENT_NAME, ste_prompt_segment, priority=SEGMENT_PRIORITY)
        self._tool = SteLintTool()
        ToolRegistry.register(self._tool)
        self._comms_tool = CommsLintTool()
        ToolRegistry.register(self._comms_tool)
        self.logger.info(
            "asd_ste100 mounted (mode=%s, enabled=%s)", current_mode(), _enabled()
        )

    async def on_unload(self) -> None:
        unregister_prompt_segment(SEGMENT_NAME)
        ToolRegistry.unregister(TOOL_NAME)
        ToolRegistry.unregister(COMMS_TOOL_NAME)
        self._tool = None
        self._comms_tool = None
        self.logger.info("asd_ste100 unmounted")


__all__ = [
    "AsdSte100Star",
    "SteLintTool",
    "SteLintInput",
    "CommsLintTool",
    "CommsLintInput",
    "SEGMENT_NAME",
    "TOOL_NAME",
    "COMMS_TOOL_NAME",
    "current_mode",
    "set_mode",
    "ste_prompt_segment",
]
