"""Research phase state machine for structured scientific workflows.

Defines a set of research phases that the agent cycles through, each with
its own system-prompt prefix and tool filter. This provides coarse-grained
control flow on top of the fine-grained ReAct loop managed by LangGraph.

Phase flow (typical):
    LITERATURE → HYPOTHESIS → PLANNING → EXECUTION → VALIDATION → REPORTING
                                                                ↑     │
                                                                └─────┘
                                                              (iterate)

The agent can transition between phases explicitly (via the phase_tool)
or the LLM can request a transition by including a special marker in its
response.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ResearchPhase(str, Enum):
    """Coarse-grained research workflow phases."""

    LITERATURE = "literature"
    HYPOTHESIS = "hypothesis"
    PLANNING = "planning"
    EXECUTION = "execution"
    VALIDATION = "validation"
    REPORTING = "reporting"
    # Free-form mode — no phase constraints (default for backwards compat)
    OPEN = "open"


@dataclass
class BudgetSpec:
    """Per-phase tool-call budget. 打通 phase → chat() 的 budget 通道."""
    max_calls: int
    recursion_limit: int
    max_tool_output_tokens: int | None = None


# Phase-aware budget: 每 phase 独立预算, EXECUTION 给最多 (写代码+训练),
# LITERATURE 最少 (只读论文). 总和 ~530 calls.
PHASE_BUDGETS: dict[ResearchPhase, BudgetSpec] = {
    ResearchPhase.LITERATURE:  BudgetSpec(max_calls=50,  recursion_limit=300),
    ResearchPhase.HYPOTHESIS:  BudgetSpec(max_calls=30,  recursion_limit=200),
    ResearchPhase.PLANNING:    BudgetSpec(max_calls=30,  recursion_limit=200),
    ResearchPhase.EXECUTION:   BudgetSpec(max_calls=300, recursion_limit=1600),
    ResearchPhase.VALIDATION:  BudgetSpec(max_calls=100, recursion_limit=550),
    ResearchPhase.REPORTING:   BudgetSpec(max_calls=20,  recursion_limit=150),
    ResearchPhase.OPEN:        BudgetSpec(max_calls=500, recursion_limit=2600),
}


# Phase-specific system prompt prefixes. These are prepended to the main
# system prompt when the agent is in that phase.
PHASE_PROMPTS: dict[ResearchPhase, str] = {
    ResearchPhase.LITERATURE: (
        "## Current phase: Literature Review\n"
        "Focus on: searching databases, reading papers, identifying gaps, "
        "understanding prior work. Prefer read-only tools. Do not start "
        "calculations yet — gather information first."
    ),
    ResearchPhase.HYPOTHESIS: (
        "## Current phase: Hypothesis Formation\n"
        "Focus on: formulating testable hypotheses based on literature. "
        "Define clear predictions, expected outcomes, and what would "
        "falsify each hypothesis. Use symbolic_math_tool for theoretical "
        "derivations."
    ),
    ResearchPhase.PLANNING: (
        "## Current phase: Experiment Planning\n"
        "Focus on: selecting computational methods, defining convergence "
        "criteria, choosing parameters (ENCUT, K-points, basis sets, "
        "timesteps). Create a clear plan before executing. Use "
        "structure_tool and potential_tool to prepare inputs."
    ),
    ResearchPhase.EXECUTION: (
        "## Current phase: Execution\n"
        "Focus on: running simulations and calculations. Submit jobs, "
        "monitor progress, collect raw results. Use vasp_tool, "
        "lammps_tool, and orchestrate_tool. Do not analyze yet — "
        "gather data first."
    ),
    ResearchPhase.VALIDATION: (
        "## Current phase: Validation & Analysis\n"
        "Focus on: checking convergence, analyzing errors, comparing "
        "with experiments, computing derived properties. Use "
        "evaluation_tool, symbolic_math_tool, and validate_tool. "
        "If results are unreliable, transition back to PLANNING."
    ),
    ResearchPhase.REPORTING: (
        "## Current phase: Reporting\n"
        "Focus on: summarizing findings, generating plots, writing "
        "structured reports. Use report_tool, visualize_tool, and "
        "diff_tool. Connect results back to the original hypotheses."
    ),
    ResearchPhase.OPEN: "",  # No prefix — full autonomy
}

# Tool filters per phase. When non-None, only the listed tools are exposed
# to the LLM during that phase. None means all registered tools are available.
# Core utility tools (file_read_tool, remember, recall, bash_tool) are always
# included via _CORE_TOOLS below.
_CORE_TOOLS: set[str] = {
    "file_read_tool",
    "file_write_tool",
    "file_edit_tool",
    "remember",
    "recall",
    "bash_tool",
    "git_tool",
    "unit_tool",
    "numerical_tool",
    "skill",
    "tool_search",
}

PHASE_TOOLS: dict[ResearchPhase, set[str] | None] = {}
"""Per-phase tool filters, derived from ToolProfile metadata after
registration. OPEN maps to None (all tools). Other phases map to
_CORE_TOOLS plus every tool whose profile.phases includes that phase
(or whose profile.phases is None, meaning all phases).

Populated by _rebuild_phase_tools(), called from register_all_tools().
Empty before registration — PhaseManager.tool_filter() returns None
for missing keys, which safely degrades to "all tools available"."""


def _rebuild_phase_tools() -> None:
    """Rebuild PHASE_TOOLS in place from ToolProfile metadata.

    Called at the end of register_all_tools() so the phase filters track
    the registered tools' declared phases instead of a hand-maintained dict.
    """
    from huginn.tools.registry import ToolRegistry

    new: dict[ResearchPhase, set[str] | None] = {ResearchPhase.OPEN: None}
    for phase in ResearchPhase:
        if phase is ResearchPhase.OPEN:
            continue
        tools = set(_CORE_TOOLS)
        for tool in ToolRegistry._tools.values():
            phases = tool.phases
            if phases is None or phase in phases:
                tools.add(tool.name)
        new[phase] = tools

    PHASE_TOOLS.clear()
    PHASE_TOOLS.update(new)


# Allowed phase transitions. Keys are source phases, values are sets of
# valid destination phases.
PHASE_TRANSITIONS: dict[ResearchPhase, set[ResearchPhase]] = {
    ResearchPhase.LITERATURE: {
        ResearchPhase.HYPOTHESIS,
        ResearchPhase.OPEN,
    },
    ResearchPhase.HYPOTHESIS: {
        ResearchPhase.PLANNING,
        ResearchPhase.LITERATURE,  # Back to lit review if needed
        ResearchPhase.OPEN,
    },
    ResearchPhase.PLANNING: {
        ResearchPhase.EXECUTION,
        ResearchPhase.HYPOTHESIS,  # Refine hypothesis
        ResearchPhase.OPEN,
    },
    ResearchPhase.EXECUTION: {
        ResearchPhase.VALIDATION,
        ResearchPhase.PLANNING,  # Replan if something goes wrong
        ResearchPhase.OPEN,
    },
    ResearchPhase.VALIDATION: {
        ResearchPhase.REPORTING,
        ResearchPhase.EXECUTION,  # Rerun with adjusted params
        ResearchPhase.PLANNING,  # Back to drawing board
        ResearchPhase.OPEN,
    },
    ResearchPhase.REPORTING: {
        ResearchPhase.LITERATURE,  # Start new cycle
        ResearchPhase.VALIDATION,  # Re-analyze
        ResearchPhase.OPEN,
    },
    ResearchPhase.OPEN: {p for p in ResearchPhase},  # Can go anywhere from OPEN
}


class PhaseManager:
    """Manages the current research phase and validates transitions."""

    def __init__(self, initial: ResearchPhase = ResearchPhase.OPEN) -> None:
        self._phase: ResearchPhase = initial
        self._history: list[ResearchPhase] = [initial]
        # budget channel: transition 后提议新 budget, Orchestrator 读它传给 chat()
        self.proposed_budget: BudgetSpec | None = PHASE_BUDGETS.get(initial)

    @property
    def phase(self) -> ResearchPhase:
        return self._phase

    @property
    def history(self) -> list[ResearchPhase]:
        return list(self._history)

    def can_transition(self, target: ResearchPhase) -> bool:
        """Check if transitioning to *target* is allowed from current phase."""
        if target == self._phase:
            return True
        allowed = PHASE_TRANSITIONS.get(self._phase, set())
        return target in allowed

    def transition(self, target: ResearchPhase) -> bool:
        """Attempt to transition to *target*. Returns True on success."""
        if not self.can_transition(target):
            return False
        if target != self._phase:
            self._phase = target
            self._history.append(target)
        # phase 转移时提议新 budget, chat() 通过 budget_override 接受
        self.proposed_budget = PHASE_BUDGETS.get(target)
        return True

    def prompt_prefix(self) -> str:
        """Return the system prompt prefix for the current phase."""
        return PHASE_PROMPTS.get(self._phase, "")

    def tool_filter(self) -> set[str] | None:
        """Return the tool filter for the current phase, or None for all."""
        return PHASE_TOOLS.get(self._phase)

    def reset(self, phase: ResearchPhase = ResearchPhase.OPEN) -> None:
        """Reset to *phase* and clear history."""
        self._phase = phase
        self._history = [phase]
        self.proposed_budget = PHASE_BUDGETS.get(phase)

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self._phase.value,
            "history": [p.value for p in self._history],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhaseManager:
        phase = ResearchPhase(data.get("phase", "open"))
        mgr = cls(initial=phase)
        mgr._history = [
            ResearchPhase(p) for p in data.get("history", [phase.value])
        ]
        return mgr


# ResearchPhase ↔ ResearchStage mapping
# Deli pipeline 的 9 个阶段映射到 ResearchPhase 的 7 个阶段
PHASE_STAGE_MAP: dict[ResearchPhase, list[str]] = {
    ResearchPhase.LITERATURE: ["topic_analysis", "literature_search", "gap_analysis"],
    ResearchPhase.HYPOTHESIS: [],
    ResearchPhase.PLANNING: ["outline"],
    ResearchPhase.EXECUTION: ["drafting"],
    ResearchPhase.VALIDATION: ["citation_verify"],
    ResearchPhase.REPORTING: ["peer_review", "revision", "final"],
}


def stage_to_phase(stage_name: str) -> ResearchPhase:
    """Deli stage name → corresponding ResearchPhase."""
    for phase, stages in PHASE_STAGE_MAP.items():
        if stage_name in stages:
            return phase
    return ResearchPhase.OPEN


# ── Autoloop phase adapter ────────────────────────────────────
# Maps autoloop's 7 string phases to ResearchPhase enum so both
# systems share the same transition graph and prompt table.
# ponytail: adapter, not refactor — one dict + two functions.

AUTOLOOP_TO_PHASE: dict[str, ResearchPhase] = {
    "perceive": ResearchPhase.LITERATURE,
    "hypothesize": ResearchPhase.HYPOTHESIS,
    "plan": ResearchPhase.PLANNING,
    "execute": ResearchPhase.EXECUTION,
    "validate": ResearchPhase.VALIDATION,
    "learn": ResearchPhase.VALIDATION,   # learn is post-validation reflection
    "report": ResearchPhase.REPORTING,
}

PHASE_TO_AUTOLOOP: dict[ResearchPhase, str] = {
    v: k for k, v in AUTOLOOP_TO_PHASE.items() if k != "learn"
}
PHASE_TO_AUTOLOOP[ResearchPhase.VALIDATION] = "validate"


def autoloop_to_phase(aloop_name: str) -> ResearchPhase:
    return AUTOLOOP_TO_PHASE.get(aloop_name, ResearchPhase.OPEN)


def phase_to_autoloop(phase: ResearchPhase) -> str:
    return PHASE_TO_AUTOLOOP.get(phase, "perceive")
