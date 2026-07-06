"""Scientific research workflow tests — full materials science scenarios.

These tests simulate realistic materials science research workflows and verify
that the cognitive engine correctly drives the agent through discovery,
validation, execution, and unification phases.

Scenarios:
1. DFT bandgap calculation (VASP)
2. Molecular dynamics simulation (LAMMPS)
3. Multi-step workflow with error recovery
4. Cross-domain structure-property relationship
"""

import pytest
from unittest.mock import MagicMock, patch
from huginn.cognitive_engine import (
    CognitiveState,
    AttentionMode,
    TransitionSignal,
    CognitiveStateMachine,
)


class TestDFTBandgapWorkflow:
    """Full DFT bandgap calculation workflow."""

    def test_gaN_bandgap_calculation_full_cycle(self):
        """Simulate: user asks for GaN bandgap → explore → plan → execute → report."""
        csm = CognitiveStateMachine()
        csm.start_session()

        # S0 → S1: User states goal
        csm.transition(TransitionSignal("user_goal", {
            "goal": "Calculate the bandgap of GaN using DFT"
        }))
        assert csm.state == CognitiveState.S1_DISCOVER
        assert csm.is_discovery
        # L1 should record the exploration target
        assert "GaN" in csm.l1_coordinates or "bandgap" in csm.l1_coordinates

        # S1 → S2: Agent generates hypothesis (e.g., PBE vs HSE06)
        csm.transition(TransitionSignal("hypothesis_generated"))
        assert csm.state == CognitiveState.S2_VALIDATE

        # S2 → S3 → S4: User confirms PBE approach, agent starts execution
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S4_CONSTRUCT
        assert csm.is_construction

        # S4: Execute VASP calculation steps
        steps = [
            ("structure_builder", "GaN wurtzite structure built"),
            ("vasp_tool", "SCF converged, E=-3.5 eV"),
            ("vasp_tool", "Band structure computed"),
            ("physics_audit", "Bandgap = 1.7 eV (PBE), reasonable"),
        ]
        for i, (tool, result) in enumerate(steps):
            csm.transition(TransitionSignal("tool_success", {
                "objective": "GaN bandgap",
                "step": str(i + 1),
                "tool_name": tool,
                "result_summary": result,
            }))

        # S4 → S5: Plan complete
        csm.transition(TransitionSignal("plan_complete"))
        assert csm.state == CognitiveState.S5_UNIFY

        # L1 should contain the full execution history (truncated)
        assert len(csm.l1_coordinates) > 0
        assert len(csm.l1_coordinates) <= 500

    def test_dft_calculation_with_convergence_failure(self):
        """VASP calculation fails to converge → feedback → retry with higher ENCUT."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "DFT Si bandgap"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S4_CONSTRUCT

        # First attempt: convergence failure
        csm.transition(TransitionSignal("tool_failure", {"tool_name": "vasp_tool"}))
        assert csm.state == CognitiveState.S6_FEEDBACK
        assert "GAP" in csm.l1_coordinates

        # Re-discover with adjusted parameters
        csm.transition(TransitionSignal("new_question"))
        assert csm.state == CognitiveState.S1_DISCOVER

        # Re-execute with higher ENCUT
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("tool_success", {
            "objective": "DFT Si bandgap",
            "step": "1",
            "tool_name": "vasp_tool",
            "result_summary": "Converged with ENCUT=520",
        }))


class TestMolecularDynamicsWorkflow:
    """LAMMPS MD simulation workflow."""

    def test_md_simulation_with_physics_audit(self):
        """MD simulation completes but physics audit flags temperature spike."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "MD simulation of Cu at 300K"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S4_CONSTRUCT

        # MD simulation "succeeds" but physics audit finds issue
        csm.transition(TransitionSignal("physics_error", {"tool_name": "lammps_tool"}))
        assert csm.state == CognitiveState.S6_FEEDBACK
        assert "GAP" in csm.l1_coordinates
        assert "lammps" in csm.l1_coordinates.lower()

        # User asks to retry with smaller timestep
        csm.transition(TransitionSignal("new_question"))
        assert csm.state == CognitiveState.S1_DISCOVER


class TestMultiStepWorkflowWithRecovery:
    """Multi-step workflow where some steps fail and need recovery."""

    def test_three_step_workflow_with_middle_failure(self):
        """3-step plan: step 1 succeeds, step 2 fails, step 2 retried, step 3 succeeds."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "Structure optimization + bandgap"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))

        # Step 1: Structure optimization — success
        csm.transition(TransitionSignal("tool_success", {
            "objective": "Structure optimization + bandgap",
            "step": "1",
            "tool_name": "vasp_tool",
            "result_summary": "Structure relaxed",
        }))

        # Step 2: SCF calculation — failure
        csm.transition(TransitionSignal("tool_failure", {"tool_name": "vasp_tool"}))
        assert csm.state == CognitiveState.S6_FEEDBACK

        # Recovery
        csm.transition(TransitionSignal("new_question"))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))

        # Step 2 retry: SCF — success
        csm.transition(TransitionSignal("tool_success", {
            "objective": "Structure optimization + bandgap",
            "step": "2",
            "tool_name": "vasp_tool",
            "result_summary": "SCF converged",
        }))

        # Step 3: Band structure — success
        csm.transition(TransitionSignal("tool_success", {
            "objective": "Structure optimization + bandgap",
            "step": "3",
            "tool_name": "vasp_tool",
            "result_summary": "Bandgap = 0.6 eV",
        }))

        # Plan complete
        csm.transition(TransitionSignal("plan_complete"))
        assert csm.state == CognitiveState.S5_UNIFY

    def test_cascading_failures_trigger_feedback_loop(self):
        """Multiple consecutive failures should keep triggering feedback."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "difficult calc"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))

        for _ in range(5):
            csm.transition(TransitionSignal("tool_failure", {"tool_name": "vasp_tool"}))
            assert csm.state == CognitiveState.S6_FEEDBACK
            csm.transition(TransitionSignal("new_question"))
            assert csm.state == CognitiveState.S1_DISCOVER
            csm.transition(TransitionSignal("user_confirmed"))
            csm.transition(TransitionSignal("user_confirmed"))
            assert csm.state == CognitiveState.S4_CONSTRUCT


class TestCrossDomainResonance:
    """Cross-domain research — structure-property relationships."""

    def test_structure_to_property_workflow(self):
        """User explores structure, then asks about property — L1 should connect them."""
        csm = CognitiveStateMachine()
        csm.start_session()

        # First question: structure
        csm.transition(TransitionSignal("user_goal", {"goal": "GaN crystal structure"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("tool_success", {
            "objective": "GaN crystal structure",
            "step": "1",
            "tool_name": "structure_builder",
            "result_summary": "Wurtzite, a=3.19Å",
        }))
        csm.transition(TransitionSignal("plan_complete"))

        # Second question: property (follow-up)
        csm.transition(TransitionSignal("new_question", {"goal": "GaN bandgap from structure"}))
        assert csm.state == CognitiveState.S1_DISCOVER

        # L1 should contain context from both questions
        assert len(csm.l1_coordinates) > 0
        # The structural coordinates serve as the "L1 position" that
        # connects the two research questions


class TestCognitivePromptRelevance:
    """Verify that cognitive prompts are relevant to the research phase."""

    def test_discovery_prompt_mentions_hypothesis(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "test"}))
        prompt = csm.get_attention_prompt()
        assert "hypothesis" in prompt.lower() or "discovery" in prompt.lower()

    def test_construction_prompt_mentions_verification(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "test"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        prompt = csm.get_attention_prompt()
        assert "verif" in prompt.lower() or "rigorous" in prompt.lower()

    def test_feedback_prompt_mentions_gap(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "test"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("tool_failure", {"tool_name": "test"}))
        prompt = csm.get_attention_prompt()
        assert "gap" in prompt.lower() or "error" in prompt.lower()

    def test_tool_preference_matches_research_phase(self):
        """Discovery phase should prefer exploration tools; construction should prefer computation."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "explore materials"}))

        discovery_pref = csm.get_tool_preference()
        assert "web_search" in discovery_pref["prefer"]
        assert "vasp" in discovery_pref["deprioritize"]

        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))

        construction_pref = csm.get_tool_preference()
        assert "vasp" in construction_pref["prefer"]
        assert "web_search" in construction_pref["deprioritize"]
