"""Tests for the cognitive engine — S0-S6 state machine + dual-mode attention."""

import pytest
from huginn.cognitive_engine import (
    CognitiveState,
    AttentionMode,
    TransitionSignal,
    CognitiveStateMachine,
    resolve_transition,
    get_attention_prompt,
    get_tool_preference,
    update_l1_coordinates,
    ALLOWED_TRANSITIONS,
    DISCOVERY_STATES,
    CONSTRUCTION_STATES,
)


class TestCognitiveStateEnum:
    def test_all_seven_states(self):
        assert len(CognitiveState) == 7

    def test_state_values(self):
        assert CognitiveState.S0_BLANK.value == "s0_blank"
        assert CognitiveState.S6_FEEDBACK.value == "s6_feedback"

    def test_string_enum(self):
        assert CognitiveState.S0_BLANK == "s0_blank"


class TestAttentionModeMapping:
    def test_discovery_states_use_singularity(self):
        for s in DISCOVERY_STATES:
            mode = AttentionMode.SINGULARITY_CONDENSATION
            from huginn.cognitive_engine import STATE_TO_ATTENTION
            assert STATE_TO_ATTENTION[s] == mode

    def test_construction_states_use_axiom_focus(self):
        from huginn.cognitive_engine import STATE_TO_ATTENTION
        for s in CONSTRUCTION_STATES:
            assert STATE_TO_ATTENTION[s] == AttentionMode.AXIOM_FOCUS

    def test_switch_states_use_mode_switch(self):
        from huginn.cognitive_engine import STATE_TO_ATTENTION
        assert STATE_TO_ATTENTION[CognitiveState.S3_SWITCH] == AttentionMode.MODE_SWITCH
        assert STATE_TO_ATTENTION[CognitiveState.S6_FEEDBACK] == AttentionMode.MODE_SWITCH


class TestAllowedTransitions:
    def test_s0_only_to_s1(self):
        assert ALLOWED_TRANSITIONS[CognitiveState.S0_BLANK] == {CognitiveState.S1_DISCOVER}

    def test_s1_can_validate_switch_or_feedback(self):
        allowed = ALLOWED_TRANSITIONS[CognitiveState.S1_DISCOVER]
        assert CognitiveState.S2_VALIDATE in allowed
        assert CognitiveState.S3_SWITCH in allowed
        assert CognitiveState.S6_FEEDBACK in allowed

    def test_s3_can_construct_or_back_to_discover(self):
        allowed = ALLOWED_TRANSITIONS[CognitiveState.S3_SWITCH]
        assert CognitiveState.S4_CONSTRUCT in allowed
        assert CognitiveState.S1_DISCOVER in allowed

    def test_s4_can_unify_or_feedback(self):
        allowed = ALLOWED_TRANSITIONS[CognitiveState.S4_CONSTRUCT]
        assert CognitiveState.S5_UNIFY in allowed
        assert CognitiveState.S6_FEEDBACK in allowed

    def test_s6_loops_back(self):
        allowed = ALLOWED_TRANSITIONS[CognitiveState.S6_FEEDBACK]
        assert CognitiveState.S0_BLANK in allowed
        assert CognitiveState.S1_DISCOVER in allowed


class TestResolveTransition:
    def test_session_start_resets_to_s0(self):
        sig = TransitionSignal("session_start")
        result = resolve_transition(CognitiveState.S4_CONSTRUCT, sig)
        assert result == CognitiveState.S0_BLANK

    def test_user_goal_from_s0_to_s1(self):
        sig = TransitionSignal("user_goal", {"goal": "calculate GaN bandgap"})
        result = resolve_transition(CognitiveState.S0_BLANK, sig)
        assert result == CognitiveState.S1_DISCOVER

    def test_user_goal_from_s6_to_s1(self):
        sig = TransitionSignal("user_goal", {"goal": "try different approach"})
        result = resolve_transition(CognitiveState.S6_FEEDBACK, sig)
        assert result == CognitiveState.S1_DISCOVER

    def test_new_question_returns_to_discover(self):
        sig = TransitionSignal("new_question")
        result = resolve_transition(CognitiveState.S5_UNIFY, sig)
        assert result == CognitiveState.S1_DISCOVER

    def test_user_confirmed_from_s2_to_s3(self):
        sig = TransitionSignal("user_confirmed")
        result = resolve_transition(CognitiveState.S2_VALIDATE, sig)
        assert result == CognitiveState.S3_SWITCH

    def test_user_confirmed_from_s3_to_s4(self):
        sig = TransitionSignal("user_confirmed")
        result = resolve_transition(CognitiveState.S3_SWITCH, sig)
        assert result == CognitiveState.S4_CONSTRUCT

    def test_user_rejected_back_to_discover(self):
        sig = TransitionSignal("user_rejected")
        result = resolve_transition(CognitiveState.S3_SWITCH, sig)
        assert result == CognitiveState.S1_DISCOVER

    def test_tool_failure_to_feedback(self):
        sig = TransitionSignal("tool_failure")
        result = resolve_transition(CognitiveState.S4_CONSTRUCT, sig)
        assert result == CognitiveState.S6_FEEDBACK

    def test_physics_error_to_feedback(self):
        sig = TransitionSignal("physics_error")
        result = resolve_transition(CognitiveState.S4_CONSTRUCT, sig)
        assert result == CognitiveState.S6_FEEDBACK

    def test_plan_complete_to_unify(self):
        sig = TransitionSignal("plan_complete")
        result = resolve_transition(CognitiveState.S4_CONSTRUCT, sig)
        assert result == CognitiveState.S5_UNIFY

    def test_invalid_transition_returns_none(self):
        sig = TransitionSignal("tool_failure")
        # Can't go from S0 to feedback directly
        result = resolve_transition(CognitiveState.S0_BLANK, sig)
        assert result is None

    def test_unknown_signal_returns_none(self):
        sig = TransitionSignal("totally_unknown_signal")
        result = resolve_transition(CognitiveState.S1_DISCOVER, sig)
        assert result is None


class TestAttentionPrompts:
    def test_discovery_prompt_content(self):
        prompt = get_attention_prompt(CognitiveState.S1_DISCOVER)
        assert "Discovery" in prompt
        assert "Ramanujan" in prompt
        assert "hypothesis" in prompt.lower()

    def test_construction_prompt_content(self):
        prompt = get_attention_prompt(CognitiveState.S4_CONSTRUCT)
        assert "Construction" in prompt
        assert "Bourbaki" in prompt
        assert "verif" in prompt.lower()   # matches "verified" / "verification"

    def test_switch_prompt_content(self):
        prompt = get_attention_prompt(CognitiveState.S3_SWITCH)
        assert "Switching" in prompt

    def test_feedback_prompt_content(self):
        prompt = get_attention_prompt(CognitiveState.S6_FEEDBACK)
        assert "gap" in prompt.lower() or "error" in prompt.lower()

    def test_all_states_have_prompts(self):
        for state in CognitiveState:
            prompt = get_attention_prompt(state)
            assert prompt  # non-empty
            assert "###" in prompt  # has section markers


class TestToolPreference:
    def test_discovery_prefers_exploration_tools(self):
        pref = get_tool_preference(CognitiveState.S1_DISCOVER)
        assert "web_search" in pref["prefer"]
        assert "literature" in pref["prefer"]
        assert "vasp" in pref["deprioritize"]

    def test_construction_prefers_computation_tools(self):
        pref = get_tool_preference(CognitiveState.S4_CONSTRUCT)
        assert "vasp" in pref["prefer"]
        assert "validate" in pref["prefer"]
        assert "web_search" in pref["deprioritize"]

    def test_switch_state_has_no_preference(self):
        pref = get_tool_preference(CognitiveState.S3_SWITCH)
        assert pref["prefer"] == []
        assert pref["deprioritize"] == []


class TestL1Coordinates:
    def test_empty_coords_stay_empty_on_irrelevant_event(self):
        result = update_l1_coordinates("", CognitiveState.S1_DISCOVER, "session_start")
        # session_start doesn't match any state-specific branch
        # but current_coords is empty so result should be empty
        assert result == ""

    def test_discovery_adds_exploration(self):
        result = update_l1_coordinates(
            "", CognitiveState.S1_DISCOVER, "user_goal",
            {"objective": "calculate GaN bandgap"},
        )
        assert "exploring: calculate GaN bandgap" in result

    def test_construction_adds_step(self):
        result = update_l1_coordinates(
            "", CognitiveState.S4_CONSTRUCT, "tool_success",
            {"objective": "GaN calc", "step": "2", "tool_name": "vasp_tool", "result_summary": "converged"},
        )
        assert "constructing: GaN calc, step 2" in result
        assert "vasp_tool" in result

    def test_feedback_records_gap(self):
        result = update_l1_coordinates(
            "exploring: GaN", CognitiveState.S6_FEEDBACK, "physics_error",
            {"tool_name": "vasp_tool"},
        )
        assert "GAP" in result
        assert "vasp_tool" in result
        assert "exploring: GaN" in result  # old coords preserved

    def test_truncation_when_too_long(self):
        long_base = "x" * 600
        result = update_l1_coordinates(
            long_base, CognitiveState.S1_DISCOVER, "user_goal",
            {"objective": "test"},
        )
        assert len(result) <= 500
        assert result.startswith("...")


class TestCognitiveStateMachine:
    def test_start_session_resets_to_s0(self):
        csm = CognitiveStateMachine()
        csm._state = CognitiveState.S4_CONSTRUCT
        csm._l1_coordinates = "old coords"
        csm.start_session()
        assert csm.state == CognitiveState.S0_BLANK
        assert csm.l1_coordinates == ""

    def test_full_lifecycle(self):
        """Test the full S0 → S1 → S2 → S3 → S4 → S5 → S6 → S1 cycle."""
        csm = CognitiveStateMachine()
        csm.start_session()

        # S0 → S1: user states a goal
        csm.transition(TransitionSignal("user_goal", {"goal": "calculate bandgap"}))
        assert csm.state == CognitiveState.S1_DISCOVER
        assert csm.is_discovery

        # S1 → S2: hypothesis generated
        csm.transition(TransitionSignal("hypothesis_generated"))
        assert csm.state == CognitiveState.S2_VALIDATE

        # S2 → S3: user confirms approach
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S3_SWITCH
        assert csm.attention_mode == AttentionMode.MODE_SWITCH

        # S3 → S4: user confirms again (or auto-confirms)
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S4_CONSTRUCT
        assert csm.is_construction
        assert csm.attention_mode == AttentionMode.AXIOM_FOCUS

        # S4 → S5: plan complete
        csm.transition(TransitionSignal("plan_complete"))
        assert csm.state == CognitiveState.S5_UNIFY

        # S5 → S1: user asks new question
        csm.transition(TransitionSignal("new_question"))
        assert csm.state == CognitiveState.S1_DISCOVER

    def test_failure_recovery_path(self):
        """S4 → S6 (feedback) → S1 (re-discover)"""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "test"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S4_CONSTRUCT

        # Tool fails
        csm.transition(TransitionSignal("tool_failure", {"tool_name": "vasp_tool"}))
        assert csm.state == CognitiveState.S6_FEEDBACK
        assert "GAP" in csm.l1_coordinates

        # Re-discover
        csm.transition(TransitionSignal("new_question"))
        assert csm.state == CognitiveState.S1_DISCOVER

    def test_physics_error_triggers_feedback(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "test"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S4_CONSTRUCT

        csm.transition(TransitionSignal("physics_error", {"tool_name": "vasp_tool"}))
        assert csm.state == CognitiveState.S6_FEEDBACK

    def test_l1_coordinates_accumulate(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "GaN bandgap"}))
        assert "exploring" in csm.l1_coordinates

        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("tool_success", {
            "objective": "GaN bandgap",
            "step": "1",
            "tool_name": "vasp_tool",
            "result_summary": "converged, E=-3.5eV",
        }))
        assert "constructing" in csm.l1_coordinates
        assert "vasp_tool" in csm.l1_coordinates

    def test_snapshot_and_restore(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "test"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm._l1_coordinates = "test coords"

        snap = csm.get_snapshot()
        assert snap["state"] == "s3_switch"
        assert snap["l1_coordinates"] == "test coords"

        csm2 = CognitiveStateMachine()
        csm2.restore_from_snapshot(snap)
        assert csm2.state == CognitiveState.S3_SWITCH
        assert csm2.l1_coordinates == "test coords"

    def test_confirmation_management(self):
        csm = CognitiveStateMachine()
        assert not csm.awaiting_confirmation
        csm.request_confirmation("plan")
        assert csm.awaiting_confirmation
        assert csm.confirmation_type == "plan"
        csm.clear_confirmation()
        assert not csm.awaiting_confirmation

    def test_attention_prompt_changes_with_state(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        assert "Discovery" in csm.get_attention_prompt()

        csm.transition(TransitionSignal("user_goal", {"goal": "test"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))
        assert "Construction" in csm.get_attention_prompt()

    def test_invalid_transition_stays_in_current_state(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        # Can't go from S0 to construct directly
        csm.transition(TransitionSignal("plan_complete"))
        assert csm.state == CognitiveState.S0_BLANK  # unchanged
