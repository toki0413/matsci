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
