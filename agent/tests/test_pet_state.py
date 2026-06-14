"""Tests for the smarter pet state machine."""

from __future__ import annotations

import time

from huginn.pet import PetEventBus, PetMood


def test_active_task_tracking():
    bus = PetEventBus()
    bus.publish(PetMood.WORKING, "Running vasp_tool…", {"tool": "vasp_tool"})
    assert bus.state.active_tasks == 1
    bus.publish(PetMood.SUCCESS, "vasp_tool done", {"tool": "vasp_tool"})
    assert bus.state.active_tasks == 0


def test_team_task_lifecycle():
    bus = PetEventBus()
    bus.publish(PetMood.WORKING, "t1 running", {"task_id": "t1", "status": "running"})
    bus.publish(PetMood.WORKING, "t2 running", {"task_id": "t2", "status": "running"})
    assert bus.state.active_tasks == 2
    bus.publish(PetMood.SUCCESS, "t1 done", {"task_id": "t1", "status": "done"})
    assert bus.state.active_tasks == 1
    bus.publish(PetMood.ERROR, "t2 error", {"task_id": "t2", "status": "error"})
    assert bus.state.active_tasks == 0


def test_state_includes_recent_events():
    bus = PetEventBus()
    bus.publish(PetMood.THINKING, "Thinking…")
    bus.publish(PetMood.SUCCESS, "Done")
    state = bus.state.to_dict()
    assert state["active_tasks"] == 0
    assert len(state["recent_events"]) == 2
    assert state["recent_events"][-1]["mood"] == "success"


def test_idle_seconds_increase():
    bus = PetEventBus()
    bus.publish(PetMood.IDLE, "Ready")
    time.sleep(0.05)
    assert bus.state.to_dict()["idle_seconds"] > 0
