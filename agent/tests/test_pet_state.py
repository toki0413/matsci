"""Tests for the smarter pet state machine."""

from __future__ import annotations

import time

from huginn.config import HuginnConfig
from huginn.pet import RAVEN_NAME, PetEventBus, PetMood


def test_default_pet_is_raven():
    bus = PetEventBus()
    state = bus.state.to_dict()
    assert state["name"] == RAVEN_NAME
    assert "avatar" in state
    assert len(state["avatar"]) > 100
    assert "@" in state["avatar"]


def test_pet_avatar_image_exists():
    from huginn.pet import RAVEN_IMAGE_PATH

    assert RAVEN_IMAGE_PATH.exists()


def test_configure_avatar():
    bus = PetEventBus()
    bus.configure(avatar="custom-avatar")
    assert bus.state.to_dict()["avatar"] == "custom-avatar"


def test_config_default_pet_name_is_raven():
    cfg = HuginnConfig()
    assert cfg.pet_name == RAVEN_NAME


def test_config_env_override_pet_name(monkeypatch):
    monkeypatch.setenv("HUGINN_PET_NAME", "Muninn")
    cfg = HuginnConfig.from_env()
    assert cfg.pet_name == "Muninn"


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
