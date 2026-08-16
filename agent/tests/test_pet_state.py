"""Tests for the smarter pet state machine."""

from __future__ import annotations

import json
import time

from huginn.config import HuginnConfig
from huginn.pet import (
    ACCESSORY_REGISTRY,
    HUNGER_DECAY_PER_MIN,
    MOOD_DECAY_PER_MIN,
    RAVEN_NAME,
    XP_PER_SUCCESS,
    PetEventBus,
    PetMood,
    PetState,
    _xp_for_level,
)


def test_default_pet_is_raven():
    bus = PetEventBus()
    state = bus.state.to_dict()
    assert state["name"] == RAVEN_NAME
    assert "avatar" in state
    assert len(state["avatar"]) > 100
    # avatar 是渲染自图片的 ASCII, 渲染失败时退回 fallback 图 — 两者都含
    # 非空白字符即可, 不强行要求 '@' (取决于 PIL 是否可用).
    assert any(ch not in " \n" for ch in state["avatar"])


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


# ── Gamification: feed / stroke / vitals ────────────────────────────


def test_feed_increases_hunger_capped_at_100():
    bus = PetEventBus()
    bus.state.hunger = 90
    bus.feed(amount=25)
    assert bus.state.hunger == 100  # capped


def test_feed_increases_hunger_within_range():
    bus = PetEventBus()
    bus.state.hunger = 50
    bus.feed(amount=25)
    assert bus.state.hunger == 75


def test_pet_stroke_increases_happiness_capped_at_100():
    bus = PetEventBus()
    bus.state.happiness = 95
    bus.pet_stroke(amount=15)
    assert bus.state.happiness == 100  # capped


def test_feed_and_stroke_publish_happy_mood():
    bus = PetEventBus()
    bus.feed()
    bus.pet_stroke()
    moods = [e["mood"] for e in bus.state.recent_events]
    assert moods == ["happy", "happy"]


def test_default_vitals():
    bus = PetEventBus()
    assert bus.state.hunger == 80
    assert bus.state.happiness == 80


# ── Decay ───────────────────────────────────────────────────────────


def test_decay_reduces_hunger_and_happiness(monkeypatch):
    import huginn.pet as pet_mod

    now = 1_000_000.0
    monkeypatch.setattr(pet_mod.time, "time", lambda: now)
    bus = PetEventBus()
    state = bus.state
    state._last_decay = now - 60.0  # exactly 1 minute elapsed
    state._apply_decay()
    assert state.hunger == 80 - HUNGER_DECAY_PER_MIN
    assert state.happiness == 80 - MOOD_DECAY_PER_MIN


def test_decay_skipped_below_threshold(monkeypatch):
    bus = PetEventBus()
    state = bus.state
    state._last_decay = time.time() - 0.1  # < 0.5 min, skip
    state._apply_decay()
    assert state.hunger == 80
    assert state.happiness == 80


def test_decay_never_goes_below_zero():
    bus = PetEventBus()
    state = bus.state
    state._last_decay = time.time() - 60.0 * 100  # huge elapsed
    state._apply_decay()
    assert state.hunger >= 0
    assert state.happiness >= 0


# ── XP / level-up ───────────────────────────────────────────────────


def test_xp_formula():
    assert _xp_for_level(1) == 100
    assert _xp_for_level(2) == int(100 * 1.15)


def test_success_awards_xp():
    bus = PetEventBus()
    bus.publish(PetMood.SUCCESS, "done")
    assert bus.state.experience == XP_PER_SUCCESS


def test_level_up_loop(monkeypatch):
    bus = PetEventBus()
    state = bus.state
    # Push directly past several thresholds to exercise the while loop.
    state.experience = _xp_for_level(1) + _xp_for_level(2) + 10
    state.level = 1
    state._award_xp_on_success(
        type(
            "E",
            (),
            {"timestamp": time.time(), "mood": PetMood.SUCCESS, "message": "x", "details": {}},
        )()
    )
    assert state.level >= 3
    assert state.experience >= 0


# ── Accessories ─────────────────────────────────────────────────────


def test_accessory_registry_has_level_gates():
    assert "crown" in ACCESSORY_REGISTRY
    assert ACCESSORY_REGISTRY["glasses"]["min_level"] == 3


def test_toggle_accessory_respects_level():
    bus = PetEventBus()
    bus.state.level = 1
    bus.toggle_accessory("crown")  # needs level 5
    assert bus.state.accessories == []


def test_toggle_accessory_equip_and_remove():
    bus = PetEventBus()
    bus.state.level = 10
    bus.toggle_accessory("crown")
    assert bus.state.accessories == ["crown"]
    bus.toggle_accessory("crown")
    assert bus.state.accessories == []


# ── Persistence ─────────────────────────────────────────────────────


def test_save_load_roundtrip(tmp_path):
    bus = PetEventBus()
    bus.state.experience = 123
    bus.state.level = 4
    bus.state.hunger = 55
    bus.state.happiness = 42
    bus.state.accessories = ["crown"]
    bus.state.name = "TestRaven"
    path = tmp_path / "pet_state.json"
    bus.state.save(path)
    assert path.exists()

    fresh = PetState()
    fresh.load(path)
    assert fresh.experience == 123
    assert fresh.level == 4
    assert fresh.hunger == 55
    assert fresh.happiness == 42
    assert fresh.accessories == ["crown"]
    assert fresh.name == "TestRaven"


def test_save_does_not_persist_runtime_fields(tmp_path):
    bus = PetEventBus()
    bus.publish(PetMood.WORKING, "running")
    path = tmp_path / "pet_state.json"
    bus.state.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert "mood" not in payload
    assert "active_tasks" not in payload


def test_load_missing_file_is_noop(tmp_path):
    fresh = PetState()
    fresh.hunger = 10
    fresh.load(tmp_path / "nope.json")  # should not raise / not change defaults
    assert fresh.hunger == 10


# ── Reset ───────────────────────────────────────────────────────────


def test_reset_progress():
    bus = PetEventBus()
    bus.state.experience = 500
    bus.state.level = 8
    bus.state.hunger = 10
    bus.state.happiness = 5
    bus.state.accessories = ["crown"]
    bus.reset_progress()
    assert bus.state.experience == 0
    assert bus.state.level == 1
    assert bus.state.hunger == 80
    assert bus.state.happiness == 80
    assert bus.state.accessories == []


# ── Queue / publish ─────────────────────────────────────────────────


def test_queue_receives_events_and_unsubscribe():
    import asyncio

    bus = PetEventBus()
    q, unsub = asyncio.run(bus.queue())
    bus.publish(PetMood.SUCCESS, "hi")
    ev = q.get_nowait()
    assert ev.mood == PetMood.SUCCESS
    assert ev.message == "hi"
    unsub()
    assert bus._queues == []


def test_queue_full_does_not_raise():
    import asyncio

    bus = PetEventBus()
    q, _unsub = asyncio.run(bus.queue())
    for _ in range(300):  # maxsize=256
        bus.publish(PetMood.IDLE, "spam")
    # Publish beyond capacity must not raise despite QueueFull.
    bus.publish(PetMood.IDLE, "overflow")
    assert q.qsize() <= 256
