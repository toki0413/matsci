"""MemoryMaintainer 梦境接线测试：睡眠门控、落库、无记忆跳过."""
from __future__ import annotations

from huginn.memory.dream import SleepClock
from huginn.memory.dream_store import DreamStore
from huginn.memory.maintainer import MemoryMaintainer


class _FakeMM:
    def maintenance(self, **kwargs):
        return {"pruned": 0, "deduplicated": 0, "expired": 0}


def _clock(window: tuple[str, str, str]) -> SleepClock:
    return SleepClock(
        persona_id="p1",
        timezone="UTC",
        weekly_windows=(window,),
    )


def test_dream_runs_only_during_sleep_and_persists(tmp_path):
    store = DreamStore(tmp_path / "dream.db")
    m = MemoryMaintainer(
        memory_manager=_FakeMM(),
        dream_enabled=True,
        sleep_clock=_clock(("mon", "00:00", "23:59")),
        dream_store=store,
        dream_memories_fn=lambda: [{"memory_id": "m1", "degree": 0}],
        now_fn=lambda: "2026-10-19T12:00:00+00:00",
    )
    result = m.run_once()
    assert result["dream"]["dreamed"] == 1
    assert store.list_dreams(persona_id="p1")


def test_dream_skips_when_active(tmp_path):
    store = DreamStore(tmp_path / "dream.db")
    m = MemoryMaintainer(
        memory_manager=_FakeMM(),
        dream_enabled=True,
        sleep_clock=_clock(("mon", "22:00", "23:00")),
        dream_store=store,
        dream_memories_fn=lambda: [{"memory_id": "m1", "degree": 0}],
        now_fn=lambda: "2026-10-19T12:00:00+00:00",
    )
    result = m.run_once()
    assert result["dream"]["dreamed"] == 0
    assert result["dream"]["state"] == "active"
    assert not store.list_dreams(persona_id="default")


def test_dream_seed_tenant_isolation(tmp_path, monkeypatch):
    """I22: 梦境种子池跨租户零泄漏；只选当前 tenant 的种子."""
    monkeypatch.setenv("HUGINN_TENANT_ID", "t1")
    store = DreamStore(tmp_path / "dream.db")
    m = MemoryMaintainer(
        memory_manager=_FakeMM(),
        dream_enabled=True,
        sleep_clock=_clock(("mon", "00:00", "23:59")),
        dream_store=store,
        dream_memories_fn=lambda: [
            {"memory_id": "t1m", "degree": 0, "tenant_id": "t1"},
            {"memory_id": "t2m", "degree": 0, "tenant_id": "t2"},
        ],
        now_fn=lambda: "2026-10-19T12:00:00+00:00",
    )
    result = m.run_once()
    assert result["dream"]["dreamed"] == 1
    payload = store.list_dreams(persona_id="p1")[0]["payload"]
    seeds = " ".join(payload["seed_event_ids"])
    assert "t1m" in seeds
    assert "t2m" not in seeds, "跨租户种子不得进入梦境"


def test_dream_skips_without_memories(tmp_path):
    store = DreamStore(tmp_path / "dream.db")
    m = MemoryMaintainer(
        memory_manager=_FakeMM(),
        dream_enabled=True,
        sleep_clock=_clock(("mon", "00:00", "23:59")),
        dream_store=store,
        dream_memories_fn=lambda: [],
        now_fn=lambda: "2026-10-19T12:00:00+00:00",
    )
    result = m.run_once()
    assert result["dream"]["dreamed"] == 0
    assert result["dream"]["reason"] == "no_memories"
