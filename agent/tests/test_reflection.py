"""Tests for TaskReflector — the post-tool reflection step.

Locks the rule-based decision table: success / failure / physics errors /
warnings each take a distinct branch, plan advancement flips the evolve
signal, and construct-mode failures suggest a switch back to discovery.
No LLM, no I/O — pure logic.
"""

from __future__ import annotations

from huginn.task_reflector import ReflectionResult, TaskReflector
from huginn.session_state import CognitiveMode, UnifiedSessionState


def test_success_no_audit():
    reflector = TaskReflector()

    result = reflector.reflect("vasp_tool", {"success": True})

    assert result.tool_succeeded is True
    assert result.has_physics_errors is False
    assert result.has_physics_warnings is False
    assert result.should_evolve is False
    assert result.needs_user_input is False


def test_success_with_physics_errors():
    reflector = TaskReflector()
    tool_result = {
        "success": True,
        "physics_audit": {
            "has_errors": True,
            "has_warnings": False,
            "findings": [
                {"severity": "error", "message": "Energy not bounded below"}
            ],
        },
    }

    result = reflector.reflect("vasp_tool", tool_result)

    assert result.tool_succeeded is True
    assert result.has_physics_errors is True
    # Physics errors count as a failure worth evolving from.
    assert result.should_evolve is True
    assert result.evolve_signal == "failure"
    assert result.should_switch_mode is True
    assert result.suggested_mode == "discover"
    assert result.needs_user_input is True
    assert result.confirm_type == "replan"


def test_success_with_physics_warnings():
    reflector = TaskReflector()
    tool_result = {
        "success": True,
        "physics_audit": {
            "has_errors": False,
            "has_warnings": True,
            "findings": [
                {"severity": "warning", "message": "k-spacing too coarse"}
            ],
        },
    }

    result = reflector.reflect("vasp_tool", tool_result)

    assert result.has_physics_warnings is True
    assert result.has_physics_errors is False
    # Warnings surface to the user but don't trigger evolution.
    assert result.needs_user_input is True
    assert result.confirm_type == "continue"
    assert result.should_evolve is False
    assert "k-spacing too coarse" in result.message


def test_tool_failure():
    reflector = TaskReflector()

    result = reflector.reflect(
        "vasp_tool", {"success": False, "error": "SCF did not converge"}
    )

    assert result.tool_succeeded is False
    assert result.should_evolve is True
    assert result.evolve_signal == "failure"
    assert result.needs_user_input is True
    assert result.confirm_type == "replan"


def test_plan_step_completed():
    reflector = TaskReflector()
    state = UnifiedSessionState()
    state.active_plan_id = "plan-1"

    result = reflector.reflect(
        "vasp_tool", {"success": True}, session_state=state
    )

    # A successful tool while a plan is active marks the step done and
    # emits a *success* evolve signal (distinct from the failure signal).
    assert result.plan_step_completed is True
    assert result.should_evolve is True
    assert result.evolve_signal == "success"
    assert result.suggested_mode == "construct"


def test_construct_to_discover_on_failure():
    reflector = TaskReflector()
    state = UnifiedSessionState()
    state.cognitive_mode = CognitiveMode.CONSTRUCT

    result = reflector.reflect(
        "vasp_tool",
        {"success": False, "error": "job crashed"},
        session_state=state,
    )

    # A failure mid-construction should push us back onto the discovery
    # chain to look for alternatives.
    assert result.should_switch_mode is True
    assert result.suggested_mode == "discover"
    assert result.evolve_signal == "failure"


def test_failure_message_includes_physics():
    reflector = TaskReflector()
    tool_result = {
        "success": False,
        "error": "SCF failed",
        "physics_audit": {
            "has_errors": True,
            "findings": [
                {"severity": "error", "message": "Energy not conserved"}
            ],
        },
    }

    result = reflector.reflect("vasp_tool", tool_result)

    assert "Energy not conserved" in result.message
    assert "Physics concerns" in result.message
    assert "SCF failed" in result.message


def test_success_message_formats_key_results():
    # Guards the float-formatting happy path in _build_success_message —
    # a None energy must be skipped, a real energy must be rendered to 4 dp.
    reflector = TaskReflector()
    tool_result = {
        "success": True,
        "parsed": {
            "energy": -12.5,
            "band_gap": 1.23,
            "converged": True,
            "force": None,  # None values are ignored, not formatted
        },
    }

    result = reflector.reflect("vasp_tool", tool_result)

    assert "energy=-12.5000 eV" in result.message
    assert "band_gap=1.23 eV" in result.message
    assert "converged=True" in result.message
    assert "force" not in result.message


def test_reflection_result_to_dict_round_trip():
    result = ReflectionResult(
        tool_succeeded=True,
        plan_step_completed=True,
        should_evolve=True,
        evolve_signal="success",
        suggested_mode="construct",
    )

    d = result.to_dict()

    assert d["tool_succeeded"] is True
    assert d["plan_step_completed"] is True
    assert d["should_evolve"] is True
    assert d["evolve_signal"] == "success"
    assert d["suggested_mode"] == "construct"
    # Every dataclass field shows up in the dict.
    assert set(d.keys()) == {
        "tool_succeeded",
        "has_physics_errors",
        "has_physics_warnings",
        "plan_step_completed",
        "should_evolve",
        "evolve_signal",
        "should_switch_mode",
        "suggested_mode",
        "message",
        "needs_user_input",
        "confirm_type",
    }
