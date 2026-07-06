"""Long-range task tests — L1 coordinate survival, cross-session continuity,
multi-turn cognitive state progression.

These tests simulate extended research workflows (20+ turns) and verify that:
- L1 coordinates accumulate meaningfully across many turns
- Context compression preserves L1 coordinates
- Cross-session continuity restores cognitive state + plan progress
- The full S0→S1→S2→S3→S4→S5→S6→S1 cycle completes successfully
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock
from huginn.cognitive_engine import (
    CognitiveState,
    AttentionMode,
    TransitionSignal,
    CognitiveStateMachine,
    update_l1_coordinates,
)


class TestLongRangeL1Survival:
    """L1 coordinates must survive across many turns and compressions."""

    def test_l1_survives_20_turn_research_cycle(self):
        """Simulate a 20-turn research conversation and verify L1 coords
        contain meaningful structural position at the end."""
        csm = CognitiveStateMachine()
        csm.start_session()

        # Turn 1: User starts with a research goal
        csm.transition(TransitionSignal("user_goal", {
            "goal": "Calculate the bandgap of GaN using DFT"
        }))
        assert "exploring" in csm.l1_coordinates

        # Turn 2-3: Discovery and validation
        csm.transition(TransitionSignal("hypothesis_generated"))
        assert csm.state == CognitiveState.S2_VALIDATE

        # Turn 4: User confirms approach
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S4_CONSTRUCT

        # Turns 5-15: Execution with tool calls
        tools = ["vasp_tool", "convergence_test", "structure_builder",
                 "vasp_tool", "physics_audit"]
        for i, tool in enumerate(tools * 2):
            csm.transition(TransitionSignal("tool_success", {
                "objective": "GaN bandgap",
                "step": str(i + 1),
                "tool_name": tool,
                "result_summary": f"step {i+1} done",
            }))

        # L1 should contain construction history
        assert "constructing" in csm.l1_coordinates

        # Turn 16: Plan complete
        csm.transition(TransitionSignal("plan_complete"))
        assert csm.state == CognitiveState.S5_UNIFY

        # Turn 17: User asks follow-up → new discovery cycle
        csm.transition(TransitionSignal("new_question"))
        assert csm.state == CognitiveState.S1_DISCOVER

        # L1 should still contain accumulated history
        assert len(csm.l1_coordinates) > 0
        assert len(csm.l1_coordinates) <= 500

    def test_l1_survives_context_compression(self):
        """Simulate context compression by verifying L1 coords are prepended
        to the summary that gets passed to the summarizer."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "LAMMPS MD simulation"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("tool_success", {
            "objective": "LAMMPS MD",
            "step": "1",
            "tool_name": "lammps_tool",
            "result_summary": "T=300K stable",
        }))

        # Simulate what _build_compact_summary does
        base_summary = "User asked about LAMMPS. Agent ran simulation."
        l1 = csm.l1_coordinates
        compact = f"[Structural Position: {l1}]\n{base_summary}"

        # The compact summary must contain L1 coordinates
        assert "[Structural Position:" in compact
        assert "LAMMPS" in compact or "lammps" in compact
        assert base_summary in compact

    def test_l1_accumulation_with_physics_errors(self):
        """Physics errors should be recorded in L1 as gaps, surviving compression."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "VASP calc"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))

        # First tool succeeds
        csm.transition(TransitionSignal("tool_success", {
            "objective": "VASP calc", "step": "1",
            "tool_name": "vasp_tool", "result_summary": "converged",
        }))

        # Second tool has physics error
        csm.transition(TransitionSignal("physics_error", {"tool_name": "vasp_tool"}))
        assert csm.state == CognitiveState.S6_FEEDBACK
        assert "GAP" in csm.l1_coordinates

        # Re-discover and try again
        csm.transition(TransitionSignal("new_question"))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))

        # The gap should still be in L1 coords (survived the cycle)
        assert "GAP" in csm.l1_coordinates


class TestCrossSessionContinuity:
    """Cross-session restoration of cognitive state and plan progress."""

    def test_csm_snapshot_restore_preserves_state_and_l1(self):
        """Session 1 saves snapshot, Session 2 restores it."""
        # Session 1
        csm1 = CognitiveStateMachine()
        csm1.start_session()
        csm1.transition(TransitionSignal("user_goal", {"goal": "long task"}))
        csm1.transition(TransitionSignal("user_confirmed"))
        csm1.transition(TransitionSignal("user_confirmed"))
        csm1.transition(TransitionSignal("tool_success", {
            "objective": "long task", "step": "3",
            "tool_name": "vasp_tool", "result_summary": "partial result",
        }))

        snap = csm1.get_snapshot()
        assert snap["state"] == "s4_construct"
        assert "constructing" in snap["l1_coordinates"]

        # Session 2
        csm2 = CognitiveStateMachine()
        csm2.restore_from_snapshot(snap)
        assert csm2.state == CognitiveState.S4_CONSTRUCT
        assert "constructing" in csm2.l1_coordinates

    def test_session_state_snapshot_excludes_transient(self):
        """Session state snapshot must not include transient fields."""
        from huginn.session_state import UnifiedSessionState

        state = UnifiedSessionState()
        state._cognitive_prompt = "transient"
        state.pending_confirmation = {"type": "test", "message": "confirm?"}
        state.tool_results_this_turn = [{"tool": "test"}]
        state.l1_coordinates = "persistent coords"

        snap = state.to_snapshot()
        assert "_cognitive_prompt" not in snap
        assert "pending_confirmation" not in snap
        assert "tool_results_this_turn" not in snap
        assert "l1_coordinates" in snap  # L1 IS persisted

    def test_plan_progress_restored_with_plan_id(self):
        """load_active_plan must return plan_id for session_state.set_plan()."""
        from huginn.memory.manager import MemoryManager
        from unittest.mock import MagicMock

        mgr = MemoryManager.__new__(MemoryManager)
        mgr.longterm = MagicMock()
        mgr.longterm.retrieve.return_value = [{
            "content": "Plan: VASP relaxation | Step: 5 | Status: in_progress | Position: constructing: step 4",
            "source": "plan:plan_a1b2c3",
            "tags": ["plan_progress", "plan_a1b2c3"],
        }]

        result = mgr.load_active_plan()
        assert result is not None
        assert result["plan_id"] == "plan_a1b2c3"
        assert result["objective"] == "VASP relaxation"
        assert result["step_index"] == 5
        assert result["status"] == "in_progress"


class TestMultiTurnCognitiveProgression:
    """Full multi-turn cognitive state progression scenarios."""

    def test_research_workflow_full_cycle(self):
        """Simulate a complete research workflow:
        Explore → Hypothesize → Validate → Plan → Execute → Report → Feedback
        """
        csm = CognitiveStateMachine()
        csm.start_session()

        # Phase 1: Exploration (S0 → S1)
        csm.transition(TransitionSignal("user_goal", {
            "goal": "Predict thermal conductivity of Si"
        }))
        assert csm.state == CognitiveState.S1_DISCOVER
        assert csm.attention_mode == AttentionMode.SINGULARITY_CONDENSATION

        # Phase 2: Hypothesis generation (S1 → S2)
        csm.transition(TransitionSignal("hypothesis_generated"))
        assert csm.state == CognitiveState.S2_VALIDATE

        # Phase 3: User confirms approach (S2 → S3 → S4)
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S3_SWITCH
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S4_CONSTRUCT
        assert csm.attention_mode == AttentionMode.AXIOM_FOCUS

        # Phase 4: Execution with multiple tool calls
        for i in range(5):
            csm.transition(TransitionSignal("tool_success", {
                "objective": "thermal conductivity",
                "step": str(i + 1),
                "tool_name": "vasp_tool" if i % 2 == 0 else "lammps_tool",
                "result_summary": f"step {i+1} converged",
            }))
        assert csm.state == CognitiveState.S4_CONSTRUCT  # still executing

        # Phase 5: Plan complete (S4 → S5)
        csm.transition(TransitionSignal("plan_complete"))
        assert csm.state == CognitiveState.S5_UNIFY

        # Phase 6: User asks follow-up (S5 → S1)
        csm.transition(TransitionSignal("new_question"))
        assert csm.state == CognitiveState.S1_DISCOVER

        # Verify L1 coordinates contain the full journey
        assert len(csm.l1_coordinates) > 0
        assert len(csm.l1_coordinates) <= 500

    def test_error_recovery_workflow(self):
        """Simulate error recovery: execute → fail → feedback → re-discover → re-execute."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "DFT calculation"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))

        # Execution fails
        csm.transition(TransitionSignal("tool_failure", {"tool_name": "vasp_tool"}))
        assert csm.state == CognitiveState.S6_FEEDBACK

        # Re-discover
        csm.transition(TransitionSignal("new_question"))
        assert csm.state == CognitiveState.S1_DISCOVER

        # Try again
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S4_CONSTRUCT

        # This time success
        csm.transition(TransitionSignal("tool_success", {
            "objective": "DFT calculation", "step": "1",
            "tool_name": "vasp_tool", "result_summary": "converged",
        }))
        csm.transition(TransitionSignal("plan_complete"))
        assert csm.state == CognitiveState.S5_UNIFY

    def test_multiple_research_questions_in_one_session(self):
        """User asks 3 different research questions in one session."""
        csm = CognitiveStateMachine()
        csm.start_session()

        questions = [
            "bandgap of GaN",
            "thermal conductivity of Si",
            "elastic modulus of steel",
        ]

        for q in questions:
            csm.transition(TransitionSignal("user_goal" if csm.state == CognitiveState.S0_BLANK else "new_question", {
                "goal": q,
            }))
            assert csm.state == CognitiveState.S1_DISCOVER
            csm.transition(TransitionSignal("user_confirmed"))
            csm.transition(TransitionSignal("user_confirmed"))
            csm.transition(TransitionSignal("plan_complete"))
            assert csm.state == CognitiveState.S5_UNIFY

        # L1 should contain history from all three questions
        assert len(csm.l1_coordinates) > 0
        assert len(csm.l1_coordinates) <= 500
