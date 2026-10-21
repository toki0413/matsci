"""I26/I27: PreparedContext 只是候选 + TTL；空闲环不产生用户可见动作."""
from __future__ import annotations

from huginn.metacog.proactive_prep import (
    IdleJob,
    IdleJobQueue,
    PreparedContext,
    PreparedContextStore,
)


def _prep(**overrides) -> PreparedContext:
    data = {
        "prep_id": "prep1",
        "tenant_id": "t1",
        "persona_id": "p1",
        "predicted_need": "next query about band gap",
        "query_hint": "band gap",
        "ttl_s": 60,
        "created_at": "2026-01-01T00:00:00+00:00",
    }
    data.update(overrides)
    return PreparedContext(**data)


def test_candidate_and_ttl(tmp_path):
    store = PreparedContextStore(tmp_path / "prep.db")
    store.save(_prep())
    candidates = store.candidates("t1", now="2026-01-01T00:00:10+00:00")
    assert len(candidates) == 1
    assert candidates[0]["status"] == "candidate"

    assert store.consume(
        "prep1", tenant_id="t1", now="2026-01-01T00:00:30+00:00"
    ) is True
    assert store.consume(
        "prep1", tenant_id="t1", now="2026-01-01T00:00:40+00:00"
    ) is False, "consumed 不得再次注入"

    store.save(
        _prep(prep_id="prep2", ttl_s=5)
    )
    assert store.candidates("t1", now="2026-01-01T01:00:00+00:00") == []
    assert store.get("prep2", tenant_id="t1")["status"] == "expired"

    # 跨租户不可见
    assert store.get("prep1", tenant_id="t2") is None


def test_no_visible_action(tmp_path):
    store = PreparedContextStore(tmp_path / "prep.db")
    queue = IdleJobQueue()
    job = IdleJob(
        job_id="j1", tenant_id="t1", persona_id="p1",
        job_type="prepare_context", budget_ms=100, sleep_state="sleep",
    )
    assert queue.submit(job) == "j1"

    def _executor(job: IdleJob) -> None:
        from datetime import UTC, datetime

        store.save(
            _prep(
                prep_id=f"prep:{job.job_id}",
                predicted_need="prepared during idle",
                created_at=datetime.now(UTC).isoformat(),
            )
        )

    counts = queue.run_due("t1", _executor)
    assert counts["done"] == 1
    assert queue.visible_actions == 0, "空闲环不得产生用户可见动作"
    preps = store.candidates("t1")
    assert preps and preps[0]["status"] == "candidate"
    assert not hasattr(store, "inject"), "候选不得自动注入"
