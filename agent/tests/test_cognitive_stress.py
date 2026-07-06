"""Stress tests for the cognitive engine — rapid transitions, edge cases, concurrency.

Tests that the state machine remains stable under:
- Rapid signal bursts (many transitions in quick succession)
- Invalid signal sequences (signals that don't match any transition)
- L1 coordinate accumulation over many turns (truncation, no overflow)
- Snapshot/restore under load
- Concurrent access (thread safety of CognitiveStateMachine)
"""

import pytest
import threading
import time
from huginn.cognitive_engine import (
    CognitiveState,
    AttentionMode,
    TransitionSignal,
    CognitiveStateMachine,
    update_l1_coordinates,
    resolve_transition,
    ALLOWED_TRANSITIONS,
)


class TestRapidTransitions:
    """Rapid-fire transitions — state machine must stay consistent."""

    def test_rapid_discovery_construction_cycles(self):
        """50 full S0→S1→S4→S6→S1 cycles back-to-back."""
        csm = CognitiveStateMachine()
        csm.start_session()

        for i in range(50):
            # S0/S6 → S1
            csm.transition(TransitionSignal("user_goal", {"goal": f"task_{i}"}))
            assert csm.state in (CognitiveState.S1_DISCOVER, CognitiveState.S0_BLANK)

            # S1 → S3 → S4
            csm.transition(TransitionSignal("user_confirmed"))
            csm.transition(TransitionSignal("user_confirmed"))
            assert csm.state == CognitiveState.S4_CONSTRUCT

            # S4 → S6 (failure)
            csm.transition(TransitionSignal("tool_failure", {"tool_name": "vasp"}))
            assert csm.state == CognitiveState.S6_FEEDBACK

        # After 50 cycles, L1 coords should be truncated (≤500 chars)
        assert len(csm.l1_coordinates) <= 500

    def test_burst_invalid_signals(self):
        """100 invalid signals must not change state."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "test"}))
        original_state = csm.state

        for _ in range(100):
            csm.transition(TransitionSignal("totally_invalid_signal"))

        assert csm.state == original_state

    def test_alternating_confirm_reject(self):
        """Rapid confirm/reject cycles must not corrupt state."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "test"}))

        for _ in range(20):
            csm.transition(TransitionSignal("user_confirmed"))
            csm.transition(TransitionSignal("user_rejected"))

        # Should end back in S1_DISCOVER
        assert csm.state == CognitiveState.S1_DISCOVER


class TestL1CoordinateAccumulation:
    """L1 coordinates under heavy accumulation."""

    def test_l1_truncation_under_massive_accumulation(self):
        """1000 tool results should not exceed 500 chars."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "massive task"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))

        for i in range(1000):
            csm.transition(TransitionSignal("tool_success", {
                "objective": "massive task",
                "step": str(i),
                "tool_name": f"tool_{i}",
                "result_summary": f"result_{i}" * 10,
            }))

        assert len(csm.l1_coordinates) <= 500

    def test_l1_preserves_recent_context_after_truncation(self):
        """After truncation, the most recent coordinates should be visible."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "task"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))

        # Generate enough data to trigger truncation
        for i in range(50):
            csm.transition(TransitionSignal("tool_success", {
                "objective": "task",
                "step": str(i),
                "tool_name": f"tool_{i}",
                "result_summary": f"latest_result_{i}",
            }))

        # The latest result should be in the coordinates
        assert "tool_49" in csm.l1_coordinates or "latest" in csm.l1_coordinates

    def test_l1_empty_string_handled(self):
        """Empty L1 coordinates must not cause issues."""
        csm = CognitiveStateMachine()
        csm.start_session()
        assert csm.l1_coordinates == ""
        prompt = csm.get_attention_prompt()
        assert prompt  # still returns a valid prompt


class TestSnapshotRestoreUnderLoad:
    """Snapshot/restore consistency."""

    def test_rapid_snapshot_restore_cycle(self):
        """20 snapshot→restore cycles must preserve state."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "test"}))
        csm.transition(TransitionSignal("user_confirmed"))

        for _ in range(20):
            snap = csm.get_snapshot()
            csm2 = CognitiveStateMachine()
            csm2.restore_from_snapshot(snap)
            assert csm2.state == csm.state
            assert csm2.l1_coordinates == csm.l1_coordinates

    def test_restore_corrupt_snapshot(self):
        """Corrupt snapshot data must not crash — falls back to S0."""
        csm = CognitiveStateMachine()
        csm.restore_from_snapshot({"state": "totally_invalid", "l1_coordinates": 12345})
        assert csm.state == CognitiveState.S0_BLANK

    def test_restore_empty_snapshot(self):
        csm = CognitiveStateMachine()
        csm.restore_from_snapshot({})
        assert csm.state == CognitiveState.S0_BLANK
        assert csm.l1_coordinates == ""


class TestThreadSafety:
    """CognitiveStateMachine is used from the agent's chat() loop.

    While not truly concurrent (Python GIL), we test that rapid
    interleaved access from multiple threads doesn't corrupt state.
    """

    def test_concurrent_transitions_same_csm(self):
        """Multiple threads transitioning the same CSM must not crash."""
        csm = CognitiveStateMachine()
        csm.start_session()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    csm.transition(TransitionSignal("user_goal", {"goal": "t"}))
                    csm.transition(TransitionSignal("user_confirmed"))
                    csm.transition(TransitionSignal("user_confirmed"))
                    csm.transition(TransitionSignal("tool_failure", {"tool_name": "x"}))
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert not errors, f"Thread errors: {errors}"
        # State should be valid (one of the 7)
        assert csm.state in CognitiveState

    def test_concurrent_l1_updates(self):
        """Concurrent L1 coordinate updates must not crash."""
        csm = CognitiveStateMachine()
        csm.start_session()
        csm.transition(TransitionSignal("user_goal", {"goal": "t"}))
        csm.transition(TransitionSignal("user_confirmed"))
        csm.transition(TransitionSignal("user_confirmed"))

        def worker():
            for i in range(100):
                csm.transition(TransitionSignal("tool_success", {
                    "objective": "t",
                    "step": str(i),
                    "tool_name": f"tool_{i}",
                    "result_summary": "ok",
                }))

        threads = [threading.Thread(target=worker) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(csm.l1_coordinates) <= 500


class TestEdgeCases:
    """Edge cases that could trip up the state machine."""

    def test_start_session_multiple_times(self):
        """Calling start_session() multiple times must be safe."""
        csm = CognitiveStateMachine()
        for _ in range(10):
            csm.start_session()
        assert csm.state == CognitiveState.S0_BLANK

    def test_transition_before_start(self):
        """Transition before start_session should still work (starts at S0)."""
        csm = CognitiveStateMachine()
        # CSM initializes at S0 by default
        csm.transition(TransitionSignal("user_goal", {"goal": "test"}))
        assert csm.state == CognitiveState.S1_DISCOVER

    def test_all_states_have_allowed_transitions(self):
        """Every state must have an entry in ALLOWED_TRANSITIONS."""
        for state in CognitiveState:
            assert state in ALLOWED_TRANSITIONS, f"Missing transitions for {state}"

    def test_all_states_have_attention_mode(self):
        """Every state must map to an attention mode."""
        from huginn.cognitive_engine import STATE_TO_ATTENTION
        for state in CognitiveState:
            assert state in STATE_TO_ATTENTION, f"Missing attention mode for {state}"

    def test_all_states_have_prompt(self):
        """Every state must have a non-empty attention prompt."""
        from huginn.cognitive_engine import STATE_PROMPTS
        for state in CognitiveState:
            assert state in STATE_PROMPTS, f"Missing prompt for {state}"
            assert STATE_PROMPTS[state], f"Empty prompt for {state}"

    def test_no_deadend_states(self):
        """No state should have zero allowed transitions (except maybe none)."""
        for state, allowed in ALLOWED_TRANSITIONS.items():
            # Every state should be able to transition somewhere
            assert len(allowed) > 0, f"Dead-end state: {state}"
