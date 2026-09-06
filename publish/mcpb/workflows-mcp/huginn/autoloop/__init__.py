"""Autoloop — autonomous closed-loop engine for Huginn.

Provides the main Perceive→Hypothesize→Plan→Execute→Validate→Learn→Report
loop that orchestrates exploration, coder, workflow, benchmark, and report
subsystems into a single cohesive agent.
"""

from __future__ import annotations

from huginn.autoloop.conjecture import ConjectureGenerator, get_conjecture_generator
from huginn.autoloop.engine import (
    AUTOLOOP_PHASES,
    AutoloopEngine,
)
from huginn.autoloop.types import (
    AutoloopResult,
    LoopPhase,
    load_autoloop_snapshot,
    objective_hash,
    save_autoloop_snapshot,
)

__all__ = [
    "AutoloopEngine",
    "AutoloopResult",
    "LoopPhase",
    "AUTOLOOP_PHASES",
    "ConjectureGenerator",
    "get_conjecture_generator",
    "objective_hash",
    "save_autoloop_snapshot",
    "load_autoloop_snapshot",
]
