"""P0 tests for cognitive state machine transitions and attention-mode mapping."""

from __future__ import annotations

import pytest

from huginn.cognitive_engine import (
    ALLOWED_TRANSITIONS,
    AttentionMode,
    CognitiveState,
    CognitiveStateMachine,
    STATE_TO_ATTENTION,
    TransitionSignal,
    resolve_transition,
)


# ── helpers ──────────────────────────────────────────────────────────


def _walk(csm: CognitiveStateMachine, *signal_types: str) -> CognitiveState:
    """Feed signals in sequence, return final state."""
    for st in signal_types:
        csm.transition(TransitionSignal(st))
    return csm.state


def _to_s6() -> CognitiveStateMachine:
    """CSM walked all the way to S6_FEEDBACK."""
    csm = CognitiveStateMachine()
    csm.start_session()
    _walk(csm, "user_goal", "hypothesis_generated", "user_confirmed",
          "user_confirmed", "plan_complete", "tool_failure")
    return csm


# ── legal transitions along the main cycle ──────────────────────────


class TestLegalTransitions:
    def test_s0_to_s1(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "study Si"}))
        assert csm.state == CognitiveState.S1_DISCOVER

    def test_s1_to_s2(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        _walk(csm, "user_goal", "hypothesis_generated")
        assert csm.state == CognitiveState.S2_VALIDATE

    def test_s2_to_s3(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        _walk(csm, "user_goal", "hypothesis_generated", "user_confirmed")
        assert csm.state == CognitiveState.S3_SWITCH

    def test_s3_to_s4(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        _walk(csm, "user_goal", "hypothesis_generated", "user_confirmed",
              "user_confirmed")
        assert csm.state == CognitiveState.S4_CONSTRUCT

    def test_s4_to_s5(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        _walk(csm, "user_goal", "hypothesis_generated", "user_confirmed",
              "user_confirmed", "plan_complete")
        assert csm.state == CognitiveState.S5_UNIFY

    def test_s5_to_s6(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        _walk(csm, "user_goal", "hypothesis_generated", "user_confirmed",
              "user_confirmed", "plan_complete", "tool_failure")
        assert csm.state == CognitiveState.S6_FEEDBACK

    def test_s6_to_s1(self):
        csm = _to_s6()
        csm.transition(TransitionSignal("new_question"))
        assert csm.state == CognitiveState.S1_DISCOVER

    def test_full_cycle_returns_to_discover(self):
        csm = _to_s6()
        _walk(csm, "new_question")
        assert csm.state == CognitiveState.S1_DISCOVER

    def test_s6_to_s0(self):
        csm = _to_s6()
        csm.transition(TransitionSignal("session_start"))
        assert csm.state == CognitiveState.S0_BLANK

    def test_s1_to_s6_directly(self):
        """S1 can jump to feedback on tool failure."""
        csm = CognitiveStateMachine()
        csm.start_session()
        _walk(csm, "user_goal", "tool_failure")
        assert csm.state == CognitiveState.S6_FEEDBACK

    def test_s2_to_s1_validation_failed(self):
        """S2 can loop back to discover on rejection."""
        csm = CognitiveStateMachine()
        csm.start_session()
        _walk(csm, "user_goal", "hypothesis_generated", "user_rejected")
        assert csm.state == CognitiveState.S1_DISCOVER

    def test_s4_to_s6_on_physics_error(self):
        """S4 can go to feedback on physics error."""
        csm = CognitiveStateMachine()
        csm.start_session()
        _walk(csm, "user_goal", "hypothesis_generated", "user_confirmed",
              "user_confirmed", "physics_error")
        assert csm.state == CognitiveState.S6_FEEDBACK


# ── illegal transitions are rejected ─────────────────────────────────


class TestIllegalTransitions:
    def test_s0_hypothesis_generated_stays(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("hypothesis_generated"))
        assert csm.state == CognitiveState.S0_BLANK

    def test_s0_plan_complete_stays(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("plan_complete"))
        assert csm.state == CognitiveState.S0_BLANK

    def test_s0_user_confirmed_stays(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.state == CognitiveState.S0_BLANK

    def test_s1_plan_complete_stays(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        _walk(csm, "user_goal")
        csm.transition(TransitionSignal("plan_complete"))
        assert csm.state == CognitiveState.S1_DISCOVER

    def test_s3_hypothesis_generated_stays(self):
        csm = CognitiveStateMachine()
        csm.start_session()
        _walk(csm, "user_goal", "user_confirmed")
        csm.transition(TransitionSignal("hypothesis_generated"))
        assert csm.state == CognitiveState.S3_SWITCH

    def test_resolve_returns_none_for_illegal(self):
        """resolve_transition should return None, not raise."""
        result = resolve_transition(
            CognitiveState.S0_BLANK,
            TransitionSignal("plan_complete"),
        )
        assert result is None


# ── attention mode mapping ──────────────────────────────────────────


class TestAttentionModeMapping:
    @pytest.mark.parametrize("state,expected", [
        (CognitiveState.S0_BLANK, AttentionMode.SINGULARITY_CONDENSATION),
        (CognitiveState.S1_DISCOVER, AttentionMode.SINGULARITY_CONDENSATION),
        (CognitiveState.S2_VALIDATE, AttentionMode.SINGULARITY_CONDENSATION),
        (CognitiveState.S3_SWITCH, AttentionMode.MODE_SWITCH),
        (CognitiveState.S4_CONSTRUCT, AttentionMode.AXIOM_FOCUS),
        (CognitiveState.S5_UNIFY, AttentionMode.AXIOM_FOCUS),
        (CognitiveState.S6_FEEDBACK, AttentionMode.MODE_SWITCH),
    ])
    def test_state_to_attention_table(self, state, expected):
        assert STATE_TO_ATTENTION[state] == expected

    def test_csm_attention_mode_tracks_state(self):
        """Verify the CSM's attention_mode property follows state changes."""
        csm = CognitiveStateMachine()
        csm.start_session()
        assert csm.attention_mode == AttentionMode.SINGULARITY_CONDENSATION  # S0

        csm.transition(TransitionSignal("user_goal"))
        assert csm.attention_mode == AttentionMode.SINGULARITY_CONDENSATION  # S1

        csm.transition(TransitionSignal("hypothesis_generated"))
        assert csm.attention_mode == AttentionMode.SINGULARITY_CONDENSATION  # S2

        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.attention_mode == AttentionMode.MODE_SWITCH  # S3

        csm.transition(TransitionSignal("user_confirmed"))
        assert csm.attention_mode == AttentionMode.AXIOM_FOCUS  # S4

        csm.transition(TransitionSignal("plan_complete"))
        assert csm.attention_mode == AttentionMode.AXIOM_FOCUS  # S5

        csm.transition(TransitionSignal("tool_failure"))
        assert csm.attention_mode == AttentionMode.MODE_SWITCH  # S6


# ── ALLOWED_TRANSITIONS adjacency sanity ─────────────────────────────


class TestAllowedTransitionsAdjacency:
    def test_s0_only_s1(self):
        assert ALLOWED_TRANSITIONS[CognitiveState.S0_BLANK] == {CognitiveState.S1_DISCOVER}

    def test_s6_can_go_to_s0_or_s1(self):
        allowed = ALLOWED_TRANSITIONS[CognitiveState.S6_FEEDBACK]
        assert CognitiveState.S0_BLANK in allowed
        assert CognitiveState.S1_DISCOVER in allowed

    def test_every_state_has_at_least_one_outgoing(self):
        for state in CognitiveState:
            assert len(ALLOWED_TRANSITIONS.get(state, set())) >= 1
