"""Tests for UnifiedSessionState — the loop-engineering session state object.

Pins the defaults, the plan lifecycle, the cognitive-mode switching protocol,
and the snapshot round-trip that backs cross-session continuity.
"""

from __future__ import annotations

from huginn.session_state import (
    CognitiveMode,
    SessionPhase,
    UnifiedSessionState,
)


def test_default_state():
    # A fresh session starts in discovery / exploration with no plan attached.
    state = UnifiedSessionState()

    assert state.cognitive_mode == CognitiveMode.DISCOVER
    assert state.phase == SessionPhase.EXPLORE
    assert state.active_plan_id is None
    assert state.active_plan_objective == ""
    assert state.active_plan_step_index == 0
    assert state.l1_coordinates == ""


def test_set_plan_switches_to_construct():
    state = UnifiedSessionState()

    state.set_plan("plan-1", "compute GaN band gap", n_steps=3)

    assert state.active_plan_id == "plan-1"
    assert state.active_plan_objective == "compute GaN band gap"
    assert state.active_plan_step_index == 0
    # Confirming a plan means we leave exploration and start building.
    assert state.cognitive_mode == CognitiveMode.CONSTRUCT
    assert state.phase == SessionPhase.EXECUTE


def test_clear_plan_returns_to_discover():
    state = UnifiedSessionState()
    state.set_plan("plan-1", "compute GaN band gap", n_steps=3)

    state.clear_plan()

    assert state.active_plan_id is None
    assert state.active_plan_objective == ""
    assert state.active_plan_step_index == 0
    # Dropping a plan puts us back on the discovery chain.
    assert state.cognitive_mode == CognitiveMode.DISCOVER


def test_advance_step_updates_coordinates():
    state = UnifiedSessionState()
    state.set_plan("plan-1", "GaN band structure", n_steps=3)

    state.advance_step()

    assert state.active_plan_step_index == 1
    # L1 coordinates are the compressed structural position — they must
    # reflect both the objective and the new step number.
    assert "GaN band structure" in state.l1_coordinates
    assert "step 2" in state.l1_coordinates


def test_request_confirmation():
    state = UnifiedSessionState()

    state.request_confirmation(
        "new_plan", "Approve this plan?", data={"steps": 3}
    )

    assert state.pending_confirmation is not None
    assert state.pending_confirmation["type"] == "new_plan"
    assert state.pending_confirmation["message"] == "Approve this plan?"
    assert state.pending_confirmation["data"] == {"steps": 3}


def test_clear_confirmation():
    state = UnifiedSessionState()
    state.request_confirmation("new_plan", "Approve this plan?")

    state.clear_confirmation()

    assert state.pending_confirmation is None


def test_snapshot_round_trip():
    state = UnifiedSessionState()
    state.session_id = "sess-1"
    state.persona_name = "matsci"
    state.cognitive_mode = CognitiveMode.CONSTRUCT
    state.phase = SessionPhase.EXECUTE
    state.l1_coordinates = "GaN SCF, step 1"
    state.active_plan_id = "plan-7"
    state.active_plan_objective = "compute DOS"
    state.active_plan_step_index = 2
    state.turns_count = 5
    state.user_goals_history = ["goal A", "goal B"]

    snap = state.to_snapshot()
    restored = UnifiedSessionState.from_snapshot(snap)

    assert restored.session_id == "sess-1"
    assert restored.persona_name == "matsci"
    assert restored.cognitive_mode == CognitiveMode.CONSTRUCT
    assert restored.phase == SessionPhase.EXECUTE
    assert restored.l1_coordinates == "GaN SCF, step 1"
    assert restored.active_plan_id == "plan-7"
    assert restored.active_plan_objective == "compute DOS"
    assert restored.active_plan_step_index == 2
    assert restored.turns_count == 5
    assert restored.user_goals_history == ["goal A", "goal B"]


def test_snapshot_excludes_transient():
    state = UnifiedSessionState()
    state.request_confirmation("new_plan", "msg", data={"a": 1})
    state.add_tool_result({"foo": "bar"})

    snap = state.to_snapshot()

    # Working state is per-turn, never persisted across sessions.
    assert "pending_confirmation" not in snap
    assert "tool_results_this_turn" not in snap


def test_switch_cognitive_mode():
    state = UnifiedSessionState()

    state.switch_cognitive_mode(CognitiveMode.CONSTRUCT, "entering construction")
    assert state.cognitive_mode == CognitiveMode.CONSTRUCT

    state.switch_cognitive_mode(CognitiveMode.DISCOVER, "back to discovery")
    assert state.cognitive_mode == CognitiveMode.DISCOVER


def test_user_goals_history():
    state = UnifiedSessionState()
    state.user_goals_history.append("find band gap of GaN")
    state.user_goals_history.append("relax the structure")

    snap = state.to_snapshot()

    assert "find band gap of GaN" in snap["user_goals_history"]
    assert "relax the structure" in snap["user_goals_history"]

    restored = UnifiedSessionState.from_snapshot(snap)
    assert restored.user_goals_history == [
        "find band gap of GaN",
        "relax the structure",
    ]
