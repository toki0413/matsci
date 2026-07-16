"""Tests for the side conversation channel (W2 A6).

Locks the behaviour added in A6:
- SideQuestion dataclass: id, question, answer, timestamps, is_answered
- SideChannel: submit / drain / respond / get / list_* / clear / counts (thread-safe)
- Shared singleton: get_shared_side_channel / set_shared_side_channel
- FastAPI /side routes: POST submit, GET list, GET pending, GET answered,
  GET single, DELETE clear
- AutoloopEngine._drain_side_questions: answers pending via model, returns count,
  disabled flag, empty channel, per-question failure isolation

All LLM paths are stubbed; no real model calls.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from huginn.autoloop.engine import AutoloopEngine
from huginn.side_conversation import (
    SideChannel,
    SideQuestion,
    get_shared_side_channel,
    set_shared_side_channel,
)


# ── shared fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def fresh_channel() -> SideChannel:
    """Inject a fresh SideChannel into the shared slot, yield it, then reset."""
    ch = SideChannel()
    set_shared_side_channel(ch)
    yield ch
    set_shared_side_channel(None)


@pytest.fixture(autouse=True)
def _reset_shared_channel():
    """Make sure no test leaks a channel into the next."""
    yield
    set_shared_side_channel(None)


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> AutoloopEngine:
    """Engine with heavy sub-components stubbed (same shape as test_autoloop_budget)."""
    monkeypatch.setattr("huginn.autoloop.engine.get_model", lambda s: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.MemoryManager", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.autoloop.engine.ProjectKnowledgeGraph", lambda *a, **kw: MagicMock()
    )
    monkeypatch.setattr("huginn.autoloop.engine.BenchmarkRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr("huginn.autoloop.engine.CoderRunner", lambda *a, **kw: MagicMock())
    monkeypatch.setattr(
        "huginn.agents.speculator.on_turn_start",
        lambda *a, **kw: {"hint": "", "predictions": []},
        raising=False,
    )
    # ponytail: KB 冷启动跑 ONNX embedding > 120s, KG 写 ~/.huginn 污染 home
    monkeypatch.setattr("huginn.autoloop.engine.AutoloopEngine._get_kb", lambda self: None)
    monkeypatch.setattr("huginn.autoloop.conjecture.get_kg", lambda *a, **kw: None)
    eng = AutoloopEngine(workspace=tmp_path)
    eng.progress_tracker = _DummyTracker()
    return eng


class _DummyTracker:
    def start_task(self, *a, **kw) -> None: ...
    def update(self, *a, **kw) -> None: ...
    def complete(self, *a, **kw) -> None: ...
    def fail(self, *a, **kw) -> None: ...


def _patch_phases(engine: AutoloopEngine, plan_mode: str = "coder") -> None:
    """Replace every phase method with a canned return."""
    engine._perceive = lambda: {"changed_files": ["x.py"], "timestamp": "t"}  # type: ignore[assignment]
    engine._hypothesize = AsyncMock(return_value="h")  # type: ignore[assignment]
    engine._plan = AsyncMock(return_value={"mode": plan_mode, "description": "d"})  # type: ignore[assignment]
    engine._execute = AsyncMock(return_value={"mode": plan_mode, "status": "ok"})  # type: ignore[assignment]
    engine._validate = AsyncMock(return_value={"tests_passed": True})  # type: ignore[assignment]
    engine._learn = AsyncMock(return_value=None)  # type: ignore[assignment]
    engine._report = AsyncMock(return_value=str(engine.workspace / "r.md"))  # type: ignore[assignment]


@pytest.fixture
def no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _noop(*a, **kw):  # noqa: ANN202
        return None

    monkeypatch.setattr(asyncio, "sleep", _noop)


# ── SideQuestion dataclass ───────────────────────────────────────────────────


class TestSideQuestion:
    def test_unanswered_by_default(self):
        sq = SideQuestion(id="side_1", question="hi?")
        assert sq.answer is None
        assert sq.answered_at is None
        assert sq.is_answered is False

    def test_answered_flag_flips_when_answer_set(self):
        sq = SideQuestion(id="side_1", question="hi?")
        sq.answer = "hello"
        sq.answered_at = "2026-07-01T00:00:00Z"
        assert sq.is_answered is True

    def test_to_dict_round_trip(self):
        sq = SideQuestion(id="side_1", question="hi?", metadata={"src": "cli"})
        d = sq.to_dict()
        assert d["id"] == "side_1"
        assert d["question"] == "hi?"
        assert d["answer"] is None
        assert d["is_answered"] is False
        assert "created_at" in d
        # metadata is not part of the public dict (only used internally)
        assert "metadata" not in d


# ── SideChannel CRUD ─────────────────────────────────────────────────────────


class TestSideChannelCRUD:
    def test_submit_returns_question_with_id(self, fresh_channel: SideChannel):
        sq = fresh_channel.submit("what is 1+1?")
        assert sq.id.startswith("side_")
        assert sq.question == "what is 1+1?"
        assert sq.is_answered is False
        assert fresh_channel.n_pending == 1

    def test_submit_with_metadata(self, fresh_channel: SideChannel):
        sq = fresh_channel.submit("q", metadata={"user": "alice"})
        assert sq.metadata == {"user": "alice"}

    def test_drain_returns_pending_snapshot(self, fresh_channel: SideChannel):
        fresh_channel.submit("q1")
        fresh_channel.submit("q2")
        pending = fresh_channel.drain()
        assert len(pending) == 2
        # non-destructive: still pending after drain
        assert fresh_channel.n_pending == 2

    def test_drain_empty_returns_empty_list(self, fresh_channel: SideChannel):
        assert fresh_channel.drain() == []

    def test_respond_moves_pending_to_answered(self, fresh_channel: SideChannel):
        sq = fresh_channel.submit("q1")
        result = fresh_channel.respond(sq.id, "a1")
        assert result is not None
        assert result.answer == "a1"
        assert result.is_answered is True
        assert result.answered_at is not None
        assert fresh_channel.n_pending == 0
        assert fresh_channel.n_answered == 1

    def test_respond_unknown_id_returns_none(self, fresh_channel: SideChannel):
        assert fresh_channel.respond("side_nope", "a") is None

    def test_get_returns_pending(self, fresh_channel: SideChannel):
        sq = fresh_channel.submit("q1")
        got = fresh_channel.get(sq.id)
        assert got is not None
        assert got.question == "q1"
        assert got.is_answered is False

    def test_get_returns_answered(self, fresh_channel: SideChannel):
        sq = fresh_channel.submit("q1")
        fresh_channel.respond(sq.id, "a1")
        got = fresh_channel.get(sq.id)
        assert got is not None
        assert got.is_answered is True
        assert got.answer == "a1"

    def test_get_unknown_returns_none(self, fresh_channel: SideChannel):
        assert fresh_channel.get("side_nope") is None

    def test_list_all_combines_pending_and_answered(self, fresh_channel: SideChannel):
        sq1 = fresh_channel.submit("q1")
        fresh_channel.submit("q2")
        fresh_channel.respond(sq1.id, "a1")
        all_items = fresh_channel.list_all()
        assert len(all_items) == 2
        assert fresh_channel.n_pending == 1
        assert fresh_channel.n_answered == 1

    def test_list_pending_only(self, fresh_channel: SideChannel):
        sq1 = fresh_channel.submit("q1")
        fresh_channel.submit("q2")
        fresh_channel.respond(sq1.id, "a1")
        pending = fresh_channel.list_pending()
        assert len(pending) == 1
        assert pending[0].question == "q2"

    def test_list_answered_only(self, fresh_channel: SideChannel):
        sq1 = fresh_channel.submit("q1")
        fresh_channel.submit("q2")
        fresh_channel.respond(sq1.id, "a1")
        answered = fresh_channel.list_answered()
        assert len(answered) == 1
        assert answered[0].answer == "a1"

    def test_clear_wipes_both_queues(self, fresh_channel: SideChannel):
        sq1 = fresh_channel.submit("q1")
        fresh_channel.submit("q2")
        fresh_channel.respond(sq1.id, "a1")
        fresh_channel.clear()
        assert fresh_channel.n_pending == 0
        assert fresh_channel.n_answered == 0
        assert fresh_channel.list_all() == []


# ── shared singleton ─────────────────────────────────────────────────────────


class TestSharedSingleton:
    def test_get_creates_singleton_lazily(self):
        set_shared_side_channel(None)
        ch = get_shared_side_channel()
        assert ch is not None
        # same instance on second call
        assert get_shared_side_channel() is ch

    def test_set_injects_custom_channel(self):
        custom = SideChannel()
        set_shared_side_channel(custom)
        assert get_shared_side_channel() is custom

    def test_set_none_resets_to_fresh_lazy(self):
        set_shared_side_channel(None)
        ch1 = get_shared_side_channel()
        set_shared_side_channel(None)
        ch2 = get_shared_side_channel()
        # after reset, a new instance is created on next get
        assert ch1 is not ch2


# ── FastAPI routes ───────────────────────────────────────────────────────────


@pytest.fixture
def client(fresh_channel: SideChannel):
    """FastAPI TestClient with the /side routes mounted."""
    from fastapi import FastAPI

    from huginn.routes.side import router as side_router

    app = FastAPI()
    app.include_router(side_router)
    return TestClient(app)


class TestSideRoutes:
    def test_post_submit_returns_question(self, client: TestClient):
        resp = client.post("/side", json={"question": "hello?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["success"] is True
        assert body["question"]["question"] == "hello?"
        assert body["question"]["is_answered"] is False
        assert body["question"]["id"].startswith("side_")

    def test_post_submit_rejects_empty_question(self, client: TestClient):
        resp = client.post("/side", json={"question": ""})
        body = resp.json()
        assert body["success"] is False
        assert "required" in body["error"]

    def test_post_submit_rejects_missing_question(self, client: TestClient):
        resp = client.post("/side", json={})
        body = resp.json()
        assert body["success"] is False

    def test_post_submit_with_metadata(self, client: TestClient):
        resp = client.post("/side", json={"question": "q", "metadata": {"k": "v"}})
        body = resp.json()
        assert body["success"] is True

    def test_post_submit_rejects_bad_metadata_type(self, client: TestClient):
        resp = client.post("/side", json={"question": "q", "metadata": "not-a-dict"})
        body = resp.json()
        assert body["success"] is False

    def test_get_list_returns_all(self, client: TestClient, fresh_channel: SideChannel):
        fresh_channel.submit("q1")
        fresh_channel.submit("q2")
        resp = client.get("/side")
        body = resp.json()
        assert body["success"] is True
        assert body["count"] == 2
        assert body["n_pending"] == 2
        assert body["n_answered"] == 0

    def test_get_pending_only(self, client: TestClient, fresh_channel: SideChannel):
        sq = fresh_channel.submit("q1")
        fresh_channel.submit("q2")
        fresh_channel.respond(sq.id, "a1")
        resp = client.get("/side/pending")
        body = resp.json()
        assert body["count"] == 1
        assert body["questions"][0]["question"] == "q2"

    def test_get_answered_only(self, client: TestClient, fresh_channel: SideChannel):
        sq = fresh_channel.submit("q1")
        fresh_channel.submit("q2")
        fresh_channel.respond(sq.id, "a1")
        resp = client.get("/side/answered")
        body = resp.json()
        assert body["count"] == 1
        assert body["questions"][0]["answer"] == "a1"

    def test_get_single_pending(self, client: TestClient, fresh_channel: SideChannel):
        sq = fresh_channel.submit("q1")
        resp = client.get(f"/side/{sq.id}")
        body = resp.json()
        assert body["success"] is True
        assert body["question"]["is_answered"] is False

    def test_get_single_answered(self, client: TestClient, fresh_channel: SideChannel):
        sq = fresh_channel.submit("q1")
        fresh_channel.respond(sq.id, "a1")
        resp = client.get(f"/side/{sq.id}")
        body = resp.json()
        assert body["success"] is True
        assert body["question"]["answer"] == "a1"
        assert body["question"]["is_answered"] is True

    def test_get_single_not_found(self, client: TestClient):
        resp = client.get("/side/side_nope")
        body = resp.json()
        assert body["success"] is False
        assert "not found" in body["error"]

    def test_delete_clears_all(self, client: TestClient, fresh_channel: SideChannel):
        fresh_channel.submit("q1")
        fresh_channel.submit("q2")
        resp = client.delete("/side")
        body = resp.json()
        assert body["success"] is True
        assert fresh_channel.n_pending == 0

    def test_full_lifecycle_via_http(self, client: TestClient):
        # submit
        resp = client.post("/side", json={"question": "what is water?"})
        qid = resp.json()["question"]["id"]
        # list shows it pending
        assert client.get("/side").json()["n_pending"] == 1
        # (agent would respond via channel; simulate by grabbing shared channel)
        ch = get_shared_side_channel()
        ch.respond(qid, "H2O")
        # poll single → answered
        single = client.get(f"/side/{qid}").json()
        assert single["question"]["is_answered"] is True
        assert single["question"]["answer"] == "H2O"
        # answered list shows it
        assert client.get("/side/answered").json()["count"] == 1
        assert client.get("/side/pending").json()["count"] == 0


# ── engine drain integration ─────────────────────────────────────────────────


class TestEngineDrainSideQuestions:
    def test_disabled_returns_zero(self, engine: AutoloopEngine, fresh_channel: SideChannel):
        engine._side_channel_enabled = False
        fresh_channel.submit("q1")
        result = asyncio.run(engine._drain_side_questions())
        assert result == 0
        # question stays pending
        assert fresh_channel.n_pending == 1

    def test_empty_channel_returns_zero(self, engine: AutoloopEngine, fresh_channel: SideChannel):
        result = asyncio.run(engine._drain_side_questions())
        assert result == 0

    def test_drains_and_answers_pending(self, engine: AutoloopEngine, fresh_channel: SideChannel):
        # stub model.ainvoke to return a canned answer
        canned = MagicMock()
        canned.content = "42"
        engine.model.ainvoke = AsyncMock(return_value=canned)

        sq1 = fresh_channel.submit("what is the answer?")
        sq2 = fresh_channel.submit("another question")
        result = asyncio.run(engine._drain_side_questions())
        assert result == 2
        assert fresh_channel.n_pending == 0
        assert fresh_channel.n_answered == 2
        got1 = fresh_channel.get(sq1.id)
        assert got1 is not None
        assert got1.answer == "42"
        got2 = fresh_channel.get(sq2.id)
        assert got2 is not None
        assert got2.answer == "42"

    def test_per_question_failure_isolation(self, engine: AutoloopEngine, fresh_channel: SideChannel):
        # first call raises, second succeeds
        good = MagicMock()
        good.content = "ok"
        engine.model.ainvoke = AsyncMock(
            side_effect=[RuntimeError("boom"), good]
        )
        sq1 = fresh_channel.submit("q1")
        sq2 = fresh_channel.submit("q2")
        result = asyncio.run(engine._drain_side_questions())
        # one answered, one failed (stays pending)
        assert result == 1
        assert fresh_channel.n_answered == 1
        assert fresh_channel.n_pending == 1
        # sq1 failed first → still pending; sq2 answered
        failed = fresh_channel.get(sq1.id)
        assert failed is not None
        assert failed.is_answered is False
        answered = fresh_channel.get(sq2.id)
        assert answered is not None
        assert answered.answer == "ok"

    def test_empty_answer_does_not_count(self, engine: AutoloopEngine, fresh_channel: SideChannel):
        canned = MagicMock()
        canned.content = "   "  # whitespace only → strips to empty
        engine.model.ainvoke = AsyncMock(return_value=canned)
        fresh_channel.submit("q1")
        result = asyncio.run(engine._drain_side_questions())
        assert result == 0
        # empty answer not posted → stays pending
        assert fresh_channel.n_pending == 1

    def test_drain_fires_on_idle_in_run(
        self, engine: AutoloopEngine, fresh_channel: SideChannel, no_sleep
    ):
        """run() with a perceive that returns None (idle) should trigger drain."""
        # make perceive return None → idle path → drain fires
        engine._perceive = lambda: None  # type: ignore[assignment]
        _patch_phases(engine)
        # override perceive AFTER _patch_phases (which sets it to a dict)
        engine._perceive = lambda: None  # type: ignore[assignment]

        canned = MagicMock()
        canned.content = "idle answer"
        engine.model.ainvoke = AsyncMock(return_value=canned)

        fresh_channel.submit("side q?")
        asyncio.run(engine.run(objective="o", max_iterations=1))
        # drain fired during idle → question answered
        assert fresh_channel.n_answered == 1
        assert fresh_channel.n_pending == 0
