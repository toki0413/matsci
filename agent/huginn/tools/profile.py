"""Tool dispatch profile — declarative metadata for scheduling decisions.

Each tool declares a ToolProfile to tell the dispatcher about its cost tier,
phase applicability, constraint scope, and light alternatives. This replaces
the old hardcoded HEAVY_TOOLS / LIGHT_TOOLS / PHASE_TOOLS / _TOOL_CONSTRAINT_SCOPES
dicts that had to be hand-maintained in four different files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from huginn.phases import ResearchPhase

CostTier = Literal["light", "heavy", "none"]


@dataclass(frozen=True)
class ToolProfile:
    """Declarative scheduling metadata for a tool.

    Attributes:
        cost_tier: "heavy" = burns hours of CPU/GPU, router gates it behind
            light-path attempts. "light" = returns in seconds, always allowed
            and recorded as a light-path attempt. "none" = neither, allowed
            by default (gated only by phase/budget/loop).
        phases: Which research phases expose this tool. None = all phases.
            Empty frozenset (default) = only OPEN phase, matching the old
            behavior for tools not listed in PHASE_TOOLS. Tools in
            _CORE_TOOLS are always available regardless of this field.
        constraint_scope: Domain scope for post-call constraint validation
            ("dft", "md", "cfd", "fea"). None = no constraint checking.
        light_alternatives: Tool names to try before this heavy tool. Only
            meaningful when cost_tier == "heavy".
        heavy_actions: When cost_tier == "heavy", restrict gating to these
            action names only (case-insensitive). None = all actions heavy.
            Example: ml_potential_tool sets {"train","fit","training"} so
            "predict" calls skip the heavy gate.
    """

    cost_tier: CostTier = "none"
    phases: frozenset[ResearchPhase] | None = frozenset()
    constraint_scope: str | None = None
    light_alternatives: tuple[str, ...] = ()
    heavy_actions: frozenset[str] | None = None
    degradation_chain: tuple[str, ...] = ()
    # Ordered fallback tools to try when THIS tool's circuit is open.
    # Follows the MGE quality hierarchy: HSE06 → PBE → ML surrogate → database.
    # The coordinator tries each automatically — no LLM improvisation needed.
    quality_tier: str = ""
    # Human-readable quality level for result tagging:
    # "dft_hse" > "dft_pbe" > "ml_surrogate" > "database" > "empirical"
