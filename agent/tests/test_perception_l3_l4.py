"""Tests for multi-modal perception layers L3 and L4."""
import pytest
import numpy as np

from huginn.perception.semantic_alignment import SemanticAligner, SemanticConflict
from huginn.perception.cognitive_integration import CognitiveIntegrator, CognitiveState


class TestSemanticAligner:
    """Tests for Layer 3: Semantic Alignment."""

    def test_embed_vector(self):
        aligner = SemanticAligner(dim=64)
        vec = aligner.embed("crystal structure lattice")
        assert isinstance(vec, np.ndarray)
        assert vec.shape == (64,)
        assert np.linalg.norm(vec) > 0.99  # Approximately normalized

    def test_similarity_same_text(self):
        aligner = SemanticAligner(dim=64)
        vec = aligner.embed("dft simulation vasp")
        sim = aligner.similarity(vec, vec)
        assert sim == pytest.approx(1.0, abs=1e-5)

    def test_similarity_different_text(self):
        aligner = SemanticAligner(dim=64)
        vec_a = aligner.embed("crystal structure")
        vec_b = aligner.embed("python code bug")
        sim = aligner.similarity(vec_a, vec_b)
        assert sim < 0.5  # Should be low

    def test_detect_conflicts_clip_vs_docstring(self):
        aligner = SemanticAligner(dim=64)
        modalities = [
            ("code", "conservation_matrix uses np.clip to truncate values"),
            ("doc", "docstring claims NO band-aids are used in this code"),
        ]
        conflicts = aligner.detect_conflicts(modalities)
        assert len(conflicts) >= 1
        assert any("band-aids" in c.description.lower() for c in conflicts)

    def test_detect_conflicts_converged_vs_failed(self):
        aligner = SemanticAligner(dim=64)
        modalities = [
            ("log", "SCF run converged successfully"),
            ("status", "convergence failed for this system"),
        ]
        conflicts = aligner.detect_conflicts(modalities)
        assert len(conflicts) >= 1
        assert any("convergence" in c.description.lower() for c in conflicts)

    def test_cross_modal_retrieve(self):
        aligner = SemanticAligner(dim=64)
        corpus = [
            ("file", "modified conservation_matrix.py"),
            ("file", "added test cases for vasp"),
            ("terminal", "error in line 42 of structure.py"),
        ]
        results = aligner.cross_modal_retrieve("crystal structure error", corpus, top_k=2)
        assert len(results) == 2
        assert all(isinstance(r[2], float) for r in results)


class TestCognitiveIntegrator:
    """Tests for Layer 4: Cognitive Integration."""

    def test_integrate_empty(self):
        cog = CognitiveIntegrator()
        # Mock snapshot with no data
        snapshot = type("Snapshot", (), {
            "workspace": ".",
            "file_events": [],
            "terminal_status": None,
            "browser_tabs": [],
            "simulation_updates": [],
            "to_context": lambda self: {},
        })()
        state = cog.integrate(snapshot)
        assert isinstance(state, CognitiveState)
        assert state.workspace == "."
        assert not state.suggests_action()  # No activity

    def test_integrate_with_errors(self):
        cog = CognitiveIntegrator()
        from huginn.perception.terminal_capture import TerminalStatus
        snapshot = type("Snapshot", (), {
            "workspace": ".",
            "file_events": [],
            "terminal_status": TerminalStatus(error_detected=True, last_output="Error: division by zero", warning_count=1, is_running=False),
            "browser_tabs": [],
            "simulation_updates": [],
            "to_context": lambda self: {},
        })()
        state = cog.integrate(snapshot)
        assert state.errors_present is True
        assert state.suggests_action() is True
        assert "investigate_errors" in state.recommended_actions

    def test_integrate_simulation_converged(self):
        cog = CognitiveIntegrator()
        from huginn.perception.simulator_log_tailer import SimulationUpdate
        snapshot = type("Snapshot", (), {
            "workspace": ".",
            "file_events": [],
            "terminal_status": None,
            "browser_tabs": [],
            "simulation_updates": [SimulationUpdate(source="vasp.log", simulator="vasp", status="converged", message="SCF converged")],
            "to_context": lambda self: {},
        })()
        state = cog.integrate(snapshot)
        assert state.simulation_converged is True
        assert "extract_results" in state.recommended_actions

    def test_memory_decay(self):
        cog = CognitiveIntegrator()
        now = 1000.0
        mem = cog.memory
        # Add memory
        from huginn.perception.cognitive_integration import TemporalMemory
        mem.append(TemporalMemory(timestamp=now, modality="test", summary="test", embedding=np.zeros(64), importance=1.0))
        assert mem[0].decay_score(now + 3600, half_life=3600) == pytest.approx(0.5, abs=0.01)

    def test_infer_task_from_files(self):
        cog = CognitiveIntegrator()
        assert cog._infer_task_from_files(["INCAR", "KPOINTS"]) == "dft simulation"
        assert cog._infer_task_from_files(["in.lammps", "data.lmp"]) == "md simulation"
        assert cog._infer_task_from_files(["conservation_matrix.py"]) == "mathematical modeling"
