"""Tests for the persona emotional trajectory tracker."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from huginn.persona_emotion import EmotionEvent, EmotionState, EmotionTracker


def test_state_clamps_values() -> None:
    state = EmotionState.from_dict({"valence": 2.0, "trust": -0.5, "fatigue": 1.5})
    assert state.valence == pytest.approx(1.0)
    assert state.trust == pytest.approx(0.0)
    assert state.fatigue == pytest.approx(1.0)


def test_tracker_updates_from_positive_message(tmp_path: Path) -> None:
    tracker = EmotionTracker("test", workspace=tmp_path)
    state = tracker.update_from_message("thank you, amazing work!")
    assert state.valence > 0.0
    assert state.affection > 0.2
    assert state.arousal > 0.0


def test_tracker_updates_from_negative_message(tmp_path: Path) -> None:
    tracker = EmotionTracker("test", workspace=tmp_path)
    state = tracker.update_from_message("this is wrong and useless")
    assert state.valence < 0.0
    assert state.trust < 0.5


def test_tracker_updates_from_task_success(tmp_path: Path) -> None:
    tracker = EmotionTracker("test", workspace=tmp_path)
    state = tracker.update_from_message("it worked, fixed perfectly")
    assert state.valence > 0.1
    assert state.interest > 0.5


def test_tracker_saves_and_loads(tmp_path: Path) -> None:
    tracker = EmotionTracker("test", workspace=tmp_path)
    tracker.update_from_message("great job")
    tracker.save()

    tracker2 = EmotionTracker("test", workspace=tmp_path)
    assert tracker2.current_state().valence > 0.0
    assert len(tracker2.current_state().events) == 1


def test_decay_toward_baseline(tmp_path: Path) -> None:
    tracker = EmotionTracker("test", workspace=tmp_path)
    tracker.apply_event(
        "praise",
        deltas={"valence": 0.8, "arousal": 0.8, "fatigue": 0.8},
    )
    high_state = tracker.current_state()
    assert high_state.valence > 0.5

    # Simulate 24 hours passing by back-dating the last timestamp.
    tracker._state.timestamp = (datetime.now(UTC) - timedelta(hours=24)).isoformat()
    decayed = tracker.current_state()
    assert decayed.valence < high_state.valence
    assert decayed.fatigue < high_state.fatigue


def test_context_prompt_includes_mood(tmp_path: Path) -> None:
    tracker = EmotionTracker("test", workspace=tmp_path)
    tracker.apply_event(
        "praise",
        deltas={"valence": 0.8, "arousal": 0.6, "affection": 0.7},
    )
    prompt = tracker.context_prompt()
    assert "Current inner state" in prompt
    assert "do not mention" in prompt


def test_trajectory_returns_recent_events(tmp_path: Path) -> None:
    tracker = EmotionTracker("test", workspace=tmp_path)
    tracker.update_from_message("hello")
    tracker.update_from_message("nice")
    traj = tracker.trajectory(limit=2)
    assert len(traj) == 2


def test_emotion_event_round_trip() -> None:
    event = EmotionEvent(
        timestamp=datetime.now(UTC).isoformat(),
        source="user",
        type="praise",
        deltas={"valence": 0.1},
        note="positive",
    )
    data = event.to_dict()
    restored = EmotionEvent.from_dict(data)
    assert restored.type == "praise"
    assert restored.deltas["valence"] == pytest.approx(0.1)
