"""Multi-modal perception layer for Huginn (updated with L3-L4).

Integrates filesystem, terminal, web, and simulator signals into a unified
perception stream that feeds the Autoloop engine.

Layers:
  L1: Signal Acquisition (filesystem_watcher, terminal_capture, webbridge_monitor, simulator_log_tailer)
  L2: Structured Understanding (feature extraction, AST parsing, 3D coordinate extraction)
  L3: Semantic Alignment (unified embedding space, cross-modal retrieval, conflict detection)
  L4: Cognitive Integration (cross-modal reasoning, temporal memory, hypothesis generation)

Usage:
    from huginn.perception import PerceptionLayer
    p = PerceptionLayer(workspace=".")
    p.start()
    snapshot = p.get_snapshot()
    
    # L3: semantic alignment
    conflicts = p.aligner.detect_conflicts([
        ("code", "conservation_matrix uses np.clip"),
        ("doc", "docstring claims NO band-aids"),
    ])
    
    # L4: cognitive integration
    state = p.integrate(snapshot)
    if state.suggests_action():
        print(state.proposed_hypothesis)
        print(state.recommended_tools)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from huginn.perception.filesystem_watcher import FilesystemWatcher, FileEvent
from huginn.perception.terminal_capture import TerminalCapture, TerminalStatus
from huginn.perception.webbridge_monitor import WebBridgeMonitor, BrowserSnapshot
from huginn.perception.simulator_log_tailer import SimulatorLogTailer, SimulationUpdate
from huginn.perception.semantic_alignment import SemanticAligner, SemanticConflict
from huginn.perception.cognitive_integration import CognitiveIntegrator, PerceptionSnapshot

# DocGraph pipeline modules (M1-M7)
from huginn.perception.doc_types import (
    BBox,
    DocumentElement,
    ElementType,
    EdgeType,
    GraphEdge,
    InformationPackage,
)
from huginn.perception.document_graph import DocumentGraph
from huginn.perception.pdf_parser import PDFElementExtractor
from huginn.perception.relation_predictor import RelationPredictor
from huginn.perception.cross_validator import CrossModalAdapter
from huginn.perception.info_pack import InfoPackAssembler
from huginn.perception.data_extractor import FigureDataExtractor
from huginn.perception.rag_bridge import RAGBridge


@dataclass
class PerceptionSnapshot:
    """Unified snapshot of all perceived state (L1-L2)."""
    timestamp: float
    workspace: str
    file_events: list[FileEvent] = field(default_factory=list)
    terminal_status: TerminalStatus | None = None
    browser_tabs: list[BrowserSnapshot] = field(default_factory=list)
    simulation_updates: list[SimulationUpdate] = field(default_factory=list)
    
    def has_activity(self) -> bool:
        return bool(
            self.file_events or
            self.simulation_updates or
            (self.terminal_status and self.terminal_status.error_detected)
        )
    
    def to_context(self) -> dict[str, Any]:
        """Convert to Autoloop context format."""
        return {
            "changed_files": [e.path for e in self.file_events if e.event_type in {"modified", "created"}],
            "git_diff": "",
            "error_patterns": self._extract_errors(),
            "browser_state": [(t.url, t.title) for t in self.browser_tabs],
            "simulation_status": {
                u.source: {"status": u.status, "progress": u.progress, "iteration": u.iteration}
                for u in self.simulation_updates
            },
            "terminal_active": self.terminal_status.is_running if self.terminal_status else False,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.timestamp)),
        }
    
    def _extract_errors(self) -> list[str]:
        errors = []
        if self.terminal_status and self.terminal_status.error_detected:
            errors.append(f"Terminal error: {self.terminal_status.last_output[:200]}")
        for u in self.simulation_updates:
            if u.status == "error":
                errors.append(f"Sim error: {u.source}: {u.message[:200]}")
        return errors


class PerceptionLayer:
    """Unified multi-modal perception layer (L1-L4)."""

    def __init__(self, workspace: str | Path) -> None:
        self.workspace = Path(workspace).resolve()
        # L1: Signal Acquisition
        self.fs = FilesystemWatcher(self.workspace)
        self.terminal = TerminalCapture()
        self.webbridge = WebBridgeMonitor()
        self.simulator = SimulatorLogTailer()
        # L3: Semantic Alignment
        self.aligner = SemanticAligner()
        # L4: Cognitive Integration
        self.integrator = CognitiveIntegrator()
        self._started = False

    def start(self) -> None:
        """Start all perception subsystems."""
        self.fs.start()
        self.simulator.start()
        self._started = True

    def stop(self) -> None:
        """Stop all perception subsystems."""
        self.fs.stop()
        self.simulator.stop()
        self._started = False

    def feed_terminal(self, data: bytes | str) -> None:
        """Feed terminal output into perception."""
        self.terminal.feed(data)

    def watch_simulator(self, log_path: str | Path, simulator_type: str | None = None) -> None:
        """Add a simulator log to watch."""
        self.simulator.watch(log_path, simulator_type)

    def get_snapshot(self) -> PerceptionSnapshot:
        """Capture a unified snapshot of current state (L1-L2)."""
        return PerceptionSnapshot(
            timestamp=time.time(),
            workspace=str(self.workspace),
            file_events=self.fs.get_events(),
            terminal_status=self.terminal.get_status(),
            browser_tabs=self.webbridge.list_tabs(),
            simulation_updates=list(self.simulator.updates()),
        )

    def get_cognitive_state(self) -> PerceptionSnapshot:
        """Capture full L1-L4 integrated cognitive state."""
        snapshot = self.get_snapshot()
        return self.integrator.integrate(snapshot)

    def detect_conflicts(self, extra_modalities: list[tuple[str, str]] | None = None) -> list[SemanticConflict]:
        """Detect semantic conflicts across modalities (L3)."""
        snapshot = self.get_snapshot()
        modalities = []
        if snapshot.file_events:
            modalities.append(("file", f"{len(snapshot.file_events)} file changes"))
        if snapshot.terminal_status:
            modalities.append(("terminal", snapshot.terminal_status.last_output[:200]))
        if snapshot.browser_tabs:
            modalities.append(("browser", f"{len(snapshot.browser_tabs)} tabs open"))
        if snapshot.simulation_updates:
            latest = snapshot.simulation_updates[-1]
            modalities.append(("simulation", f"{latest.simulator}: {latest.status}"))
        if extra_modalities:
            modalities.extend(extra_modalities)
        return self.aligner.detect_conflicts(modalities)
