"""Cognitive Integration Layer — Layer 4 of Multi-Modal Perception.

Highest layer: cross-modal reasoning, temporal memory, uncertainty
quantification, and autonomous hypothesis generation.

Integrates outputs from L1-L3 into coherent cognitive state that drives
the Autoloop engine.

No external dependencies. Pure Python + numpy.

Usage:
    from huginn.perception.cognitive_integration import CognitiveIntegrator
    from huginn.perception import PerceptionSnapshot

    cog = CognitiveIntegrator()
    snapshot = PerceptionSnapshot(...)  # from L1-L2-L3
    state = cog.integrate(snapshot)
    
    # Query state
    if state.suggests_action():
        print(state.proposed_hypothesis)
        print(state.recommended_tools)
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from huginn.perception.semantic_alignment import SemanticAligner, SemanticConflict


@dataclass
class TemporalMemory:
    """A remembered perception event with decay."""
    timestamp: float
    modality: str
    summary: str
    embedding: np.ndarray
    importance: float = 1.0

    def decay_score(self, now: float, half_life: float = 3600.0) -> float:
        """Exponential decay of importance over time."""
        age = now - self.timestamp
        return self.importance * (0.5 ** (age / half_life))


@dataclass
class PerceptionSnapshot:
    """Integrated cognitive state from multi-modal perception."""
    timestamp: float
    workspace: str
    
    # Raw summaries
    file_summary: str = ""
    terminal_summary: str = ""
    browser_summary: str = ""
    simulation_summary: str = ""
    
    # Inferred state
    active_task: str | None = None
    errors_present: bool = False
    warnings_present: bool = False
    simulation_running: bool = False
    simulation_converged: bool | None = None
    
    # Conflicts
    conflicts: list[SemanticConflict] = field(default_factory=list)
    
    # Hypothesis
    proposed_hypothesis: str = ""
    confidence: float = 0.0
    
    # Recommendations
    recommended_tools: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    
    def suggests_action(self) -> bool:
        """Whether the current state suggests taking an action."""
        meaningful_actions = [a for a in self.recommended_actions if a != "wait_for_activity"]
        is_idle_hypothesis = self.proposed_hypothesis and "workspace is idle" in self.proposed_hypothesis
        return bool(
            self.errors_present or
            self.conflicts or
            (self.proposed_hypothesis and not is_idle_hypothesis) or
            meaningful_actions
        )


class CognitiveIntegrator:
    """Layer 4: Cross-modal reasoning and temporal memory integration."""

    def __init__(self, max_memory: int = 100) -> None:
        self.aligner = SemanticAligner()
        self.memory: deque[TemporalMemory] = deque(maxlen=max_memory)
        self._last_integration_time: float = 0.0

    def integrate(self, snapshot: Any) -> PerceptionSnapshot:
        """Integrate a perception snapshot into cognitive state."""
        now = time.time()
        self._last_integration_time = now

        # Extract from snapshot (duck typing — works with PerceptionSnapshot or dict)
        workspace = getattr(snapshot, "workspace", ".")
        if isinstance(workspace, str) and not workspace:
            workspace = "."
        if hasattr(snapshot, "get"):
            workspace = snapshot.get("workspace", workspace)
        file_events = getattr(snapshot, "file_events", [])
        if hasattr(snapshot, "get"):
            file_events = snapshot.get("file_events", file_events)
        terminal_status = getattr(snapshot, "terminal_status", None)
        if hasattr(snapshot, "get"):
            terminal_status = snapshot.get("terminal_status", terminal_status)
        browser_tabs = getattr(snapshot, "browser_tabs", [])
        if hasattr(snapshot, "get"):
            browser_tabs = snapshot.get("browser_tabs", browser_tabs)
        sim_updates = getattr(snapshot, "simulation_updates", [])
        if hasattr(snapshot, "get"):
            sim_updates = snapshot.get("simulation_updates", sim_updates)

        state = PerceptionSnapshot(timestamp=now, workspace=workspace)

        # ── File events summary ──
        if file_events:
            modified = [e.path for e in file_events if e.event_type == "modified"]
            created = [e.path for e in file_events if e.event_type == "created"]
            state.file_summary = f"{len(modified)} modified, {len(created)} created"
            if modified:
                # Infer active task from file names
                state.active_task = self._infer_task_from_files(modified)

        # ── Terminal summary ──
        if terminal_status:
            state.terminal_summary = terminal_status.last_output[:200]
            state.errors_present = terminal_status.error_detected
            state.warnings_present = terminal_status.warning_count > 0
            state.simulation_running = terminal_status.is_running

        # ── Browser summary ──
        if browser_tabs:
            active = [t for t in browser_tabs if t.active]
            state.browser_summary = f"{len(browser_tabs)} tabs, {len(active)} active"
            if active:
                urls = [t.url for t in active]
                if any("band" in u or "dos" in u or "structure" in u for u in urls):
                    state.active_task = state.active_task or "visual analysis"

        # ── Simulation summary ──
        if sim_updates:
            latest = sim_updates[-1]
            state.simulation_summary = f"{latest.simulator}: {latest.status}"
            state.simulation_running = latest.status == "running"
            state.simulation_converged = latest.status == "converged"
            if latest.status == "error":
                state.errors_present = True

        # ── Cross-modal conflict detection ──
        modalities = self._extract_modalities(state, snapshot)
        if len(modalities) >= 2:
            state.conflicts = self.aligner.detect_conflicts(modalities)

        # ── Temporal memory integration ──
        self._update_memory(now, state, snapshot)
        historical_context = self._retrieve_relevant_memory(now, state)

        # ── Hypothesis generation ──
        state.proposed_hypothesis = self._generate_hypothesis(state, historical_context)
        state.confidence = self._compute_confidence(state)

        # ── Recommendations ──
        state.recommended_tools = self._recommend_tools(state)
        state.recommended_actions = self._recommend_actions(state)

        return state

    def _extract_modalities(self, state: PerceptionSnapshot, snapshot: Any) -> list[tuple[str, str]]:
        """Extract (modality, text) pairs for conflict detection."""
        modalities = []
        if state.file_summary:
            modalities.append(("file", state.file_summary))
        if state.terminal_summary:
            modalities.append(("terminal", state.terminal_summary))
        if state.browser_summary:
            modalities.append(("browser", state.browser_summary))
        if state.simulation_summary:
            modalities.append(("simulation", state.simulation_summary))
        # Add code/doc content if available in snapshot
        context = getattr(snapshot, "to_context", lambda: {})()
        if isinstance(context, dict):
            for err in context.get("error_patterns", [])[:3]:
                modalities.append(("error", err))
        return modalities

    def _infer_task_from_files(self, files: list[str]) -> str | None:
        """Infer active task from modified file names."""
        text = " ".join(files).lower()
        if any(x in text for x in ["vasp", "incar", "kpoints", "poscar"]):
            return "dft simulation"
        if any(x in text for x in ["lammps", "input", "data", "in."]):
            return "md simulation"
        if any(x in text for x in ["conservation", "matrix", "equation", "pde"]):
            return "mathematical modeling"
        if any(x in text for x in ["test", "pytest", "spec"]):
            return "testing"
        if any(x in text for x in ["structure", "cif", "crystal", "atom"]):
            return "structure analysis"
        return "code development"

    def _update_memory(self, now: float, state: PerceptionSnapshot, snapshot: Any) -> None:
        """Store current state to temporal memory."""
        summary = f"{state.file_summary}; {state.terminal_summary[:50]}"
        vec = self.aligner.embed(summary)
        importance = 1.0
        if state.errors_present:
            importance = 2.0
        if state.conflicts:
            importance = 2.5
        self.memory.append(TemporalMemory(
            timestamp=now, modality="integrated",
            summary=summary, embedding=vec,
            importance=importance,
        ))

    def _retrieve_relevant_memory(self, now: float, state: PerceptionSnapshot, top_k: int = 3) -> list[TemporalMemory]:
        """Retrieve most relevant past memories."""
        if not self.memory:
            return []
        query_vec = self.aligner.embed(f"{state.file_summary} {state.terminal_summary}")
        scored = []
        for mem in self.memory:
            sim = self.aligner.similarity(query_vec, mem.embedding)
            decay = mem.decay_score(now)
            scored.append((sim * decay, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scored[:top_k]]

    def _generate_hypothesis(self, state: PerceptionSnapshot, memory: list[TemporalMemory]) -> str:
        """Generate a hypothesis from current state and memory."""
        if state.errors_present and state.simulation_running:
            return "Simulation may be failing due to convergence issues; check parameters"
        if state.conflicts:
            c = state.conflicts[0]
            return f"Conflict detected: {c.description} between {c.source_a} and {c.source_b}"
        if state.simulation_converged is False:
            return "Simulation did not converge; consider parameter tuning or alternative method"
        if state.simulation_converged is True and not state.errors_present:
            return "Simulation converged successfully; results ready for analysis"
        if state.active_task == "mathematical modeling" and state.warnings_present:
            return "Mathematical model has warnings; verify conservation laws"
        if not state.file_summary and not state.terminal_summary:
            return "No recent activity detected; workspace is idle"
        return f"Active task: {state.active_task or 'unknown'}; monitoring for changes"

    def _compute_confidence(self, state: PerceptionSnapshot) -> float:
        """Compute confidence score for the hypothesis."""
        score = 0.5
        if state.errors_present:
            score += 0.2  # Higher confidence when errors are clear
        if state.conflicts:
            score += 0.3
        if state.simulation_converged is not None:
            score += 0.1
        return min(score, 1.0)

    def _recommend_tools(self, state: PerceptionSnapshot) -> list[str]:
        """Recommend tools based on state."""
        tools = []
        if state.errors_present:
            tools.append("diagnose_tool")
            tools.append("code_tool")
        if state.simulation_running:
            tools.append("validate_tool")
        if state.conflicts:
            tools.append("bourbaki_tool")
        if state.active_task == "dft simulation":
            tools.append("vasp_tool")
        if state.active_task == "md simulation":
            tools.append("lammps_tool")
        if state.active_task == "mathematical modeling":
            tools.append("symbolic_math_tool")
        if not tools and state.file_summary:
            tools.append("file_read_tool")
        return tools

    def _recommend_actions(self, state: PerceptionSnapshot) -> list[str]:
        """Recommend high-level actions."""
        actions = []
        if state.errors_present:
            actions.append("investigate_errors")
        if state.conflicts:
            actions.append("resolve_conflicts")
        if state.simulation_converged is False:
            actions.append("tune_simulation_parameters")
        if state.simulation_converged is True:
            actions.append("extract_results")
        if state.warnings_present and not state.errors_present:
            actions.append("review_warnings")
        if not state.file_summary and not state.terminal_summary:
            actions.append("wait_for_activity")
        return actions
