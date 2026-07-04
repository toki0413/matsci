"""Autoloop — autonomous closed-loop engine for Huginn.

Provides the main Perceive→Hypothesize→Plan→Execute→Validate→Learn→Report
loop that orchestrates exploration, coder, workflow, benchmark, and report
subsystems into a single cohesive agent.
"""

from __future__ import annotations

from huginn.autoloop.engine import AutoloopEngine, AutoloopResult, LoopPhase
from huginn.autoloop.conjecture import ConjectureGenerator, get_conjecture_generator

__all__ = [
    "AutoloopEngine",
    "AutoloopResult",
    "LoopPhase",
    "ConjectureGenerator",
    "get_conjecture_generator",
]
