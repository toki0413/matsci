"""侧边任务契约门 (红蓝双轨).

蓝队: 从某条消息 fork 时只继承该点之前的 user/assistant 上下文; 种子只注入
      一次; 转正后 thread_id 不变, 任务卡片带上历史条数与产物; 过期侧边任务
      自动清理而转正任务保留; 丢弃时 thread 注册 + checkpointer 行一起删.
红队: 父线程缺失 / 索引越界 / 重复转正 / 空回答必须如实失败, 不得把
      "没继承到上下文" 伪装成成功.
"""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from huginn.side_tasks import (
    STATUS_PROMOTED,
    STATUS_SIDE,
    SideTaskStore,
    extract_artifacts,
    purge_thread_checkpoints,
    set_shared_side_task_store,
)

_PARENT_MESSAGES = [
    {"role": "user", "content": "主任务: 计算 GaN 带隙"},
    {"role": "assistant", "content": "先做收敛测试"},
    {"role": "tool", "content": "ENCUT=520 converged"},
    {"role": "user", "content": "顺便查一下 k 点"},
    {"role": "assistant", "content": "k 点用 4x4"},
]


@pytest.fixture
def store(tmp_path):
    st = SideTaskStore(tmp_path / "side_tasks.json")
    set_shared_side_task_store(st)
    yield st
    set_shared_side_task_store(None)


# ── fork 语义 ──────────────────────────────────────────────


class TestFork:
    def test_blue_cuts_at_index_and_filters_tool_messages(self, store):
        task = store.create(
            parent_thread_id="main",
            messages=_PARENT_MESSAGES,
            from_message_index=1,  # user(0) + assistant(1)
        )
        assert [m["content"] for m in task.seed_messages] == [
            "主任务: 计算 GaN 带隙",
            "先做收敛测试",
        ]
        assert all(m["role"] in ("user", "assistant") for m in task.seed_messages)
        assert task.parent_thread_id == "main"
        assert task.status == STATUS_SIDE

    def test_blue_no_index_inherits_full_history(self, store):
        task = store.create(parent_thread_id="main", messages=_PARENT_MESSAGES)
        assert len(task.seed_messages) == 4  # 3 user/assistant + 1 extra pair
        assert task.title == "主任务: 计算 GaN 带隙"

    def test_blue_inherit_context_false_gives_empty_seed(self, store):
        task = store.create(
            parent_thread_id="main",
            messages=_PARENT_MESSAGES,
            inherit_context=False,
        )
        assert task.seed_messages == []
        assert task.title == "Side task"

    def test_red_index_out_of_range_raises(self, store):
        with pytest.raises(ValueError, match="out of range"):
            store.create(
                parent_thread_id="main",
                messages=_PARENT_MESSAGES,
                from_message_index=99,
            )

    def test_red_missing_parent_raises(self, store):
        with pytest.raises(ValueError, match="parent_thread_id"):
            store.create(parent_thread_id="", messages=_PARENT_MESSAGES)

    def test_blue_persists_across_store_instances(self, store, tmp_path):
        task = store.create(parent_thread_id="main", messages=_PARENT_MESSAGES)
        reloaded = SideTaskStore(tmp_path / "side_tasks.json")
        got = reloaded.get(task.id)
        assert got is not None
        assert got.seed_messages == task.seed_messages
        assert got.parent_thread_id == "main"


# ── fork 种子只注入一次 ────────────────────────────────────


class TestSeed:
    def test_blue_seed_peeked_then_consumed_once(self, store):
        task = store.create(parent_thread_id="main", messages=_PARENT_MESSAGES)
        assert len(store.peek_seed(task.thread_id)) == 4
        assert store.consume_seed(task.thread_id) is True
        assert store.peek_seed(task.thread_id) == []
        assert store.consume_seed(task.thread_id) is False

    def test_red_unknown_thread_has_no_seed(self, store):
        assert store.peek_seed("side_nope") == []

    def test_red_promoted_task_no_longer_seeds(self, store):
        task = store.create(parent_thread_id="main", messages=_PARENT_MESSAGES)
        store.promote(task.id)
        assert store.peek_seed(task.thread_id) == []


# ── 转正 ──────────────────────────────────────────────────


class TestPromote:
    def test_blue_promote_builds_task_card_with_history_and_artifacts(self, store):
        task = store.create(parent_thread_id="main", messages=_PARENT_MESSAGES)
        store.append_turn(task.id, "user", "侧边问题: 4x4 够吗?")
        store.append_turn(task.id, "assistant", "对 GaN 够用")
        store.add_artifacts(task.id, [{"path": "out/band.png", "tool": "file_write_tool"}])

        promoted = store.promote(task.id, title="GaN k 点核查")
        assert promoted is not None
        assert promoted.status == STATUS_PROMOTED
        assert promoted.thread_id == task.id  # 原地转正, 历史不搬家
        card = promoted.task_card
        assert card["task_id"] == task.id
        assert card["title"] == "GaN k 点核查"
        assert card["goal"] == "主任务: 计算 GaN 带隙"
        assert card["history_count"] == 4 + 2
        assert card["artifacts"] == [{"path": "out/band.png", "tool": "file_write_tool"}]
        assert card["source_thread_id"] == "main"

    def test_red_promote_twice_returns_none(self, store):
        task = store.create(parent_thread_id="main", messages=_PARENT_MESSAGES)
        assert store.promote(task.id) is not None
        assert store.promote(task.id) is None

    def test_red_promote_unknown_returns_none(self, store):
        assert store.promote("side_nope") is None


# ── TTL / 零残留 ──────────────────────────────────────────


class TestCleanup:
    def test_blue_sweep_expires_idle_side_but_keeps_promoted(self, store):
        side = store.create(parent_thread_id="main", messages=_PARENT_MESSAGES)
        promoted = store.create(parent_thread_id="main", messages=_PARENT_MESSAGES)
        store.promote(promoted.id)
        # 把两者都改成很久以前
        old = "2020-01-01T00:00:00+00:00"
        with store._lock:
            store._tasks[side.id].updated_at = old
            store._tasks[promoted.id].updated_at = old

        removed = store.sweep(ttl_s=60)
        assert [t.id for t in removed] == [side.id]
        assert store.get(side.id) is None
        assert store.get(promoted.id) is not None

    def test_red_discard_unknown_returns_none(self, store):
        assert store.discard("side_nope") is None

    def test_blue_purge_thread_checkpoints_deletes_both_tables(self, tmp_path):
        db = tmp_path / "ckpt.sqlite"
        conn = sqlite3.connect(db)
        conn.execute("CREATE TABLE checkpoints (thread_id TEXT)")
        conn.execute("CREATE TABLE writes (thread_id TEXT)")
        conn.executemany(
            "INSERT INTO checkpoints VALUES (?)", [("side_a",), ("main",)]
        )
        conn.executemany("INSERT INTO writes VALUES (?)", [("side_a",), ("main",)])
        conn.commit()
        conn.close()

        deleted = purge_thread_checkpoints("side_a", db_path=db)
        assert deleted == 2
        conn = sqlite3.connect(db)
        left = {
            row[0]
            for table in ("checkpoints", "writes")
            for row in conn.execute(f"SELECT thread_id FROM {table}")
        }
        conn.close()
        assert left == {"main"}

    def test_red_purge_missing_db_is_noop(self, tmp_path):
        assert purge_thread_checkpoints("side_a", db_path=tmp_path / "nope.sqlite") == 0


# ── 产物抽取 ──────────────────────────────────────────────


class TestArtifacts:
    def test_blue_only_write_tools_and_dedup(self):
        msgs = [
            SimpleNamespace(tool_calls=[
                {"name": "file_write_tool", "args": {"path": "a.png"}},
                {"name": "file_read_tool", "args": {"path": "readme.md"}},
            ]),
            SimpleNamespace(tool_calls=[
                {"name": "multi_edit_tool", "args": {"file_path": "a.png"}},
                {"name": "code_tool", "args": {"output": "b.json"}},
            ]),
        ]
        arts = extract_artifacts(msgs)
        assert arts == [
            {"path": "a.png", "tool": "file_write_tool"},
            {"path": "b.json", "tool": "code_tool"},
        ]

    def test_red_non_write_tool_yields_nothing(self):
        msgs = [SimpleNamespace(tool_calls=[
            {"name": "grep_tool", "args": {"path": "x"}},
        ])]
        assert extract_artifacts(msgs) == []


# ── HTTP 路由 ─────────────────────────────────────────────


@pytest.fixture
def side_env(store, monkeypatch):
    """Patch server_core thread registry + parent loader for route tests."""
    from huginn import server_core

    threads = {"main": {"id": "main", "label": "Main", "created_at": "", "last_active": ""}}
    monkeypatch.setattr(server_core, "_threads", threads)
    monkeypatch.setattr(
        "huginn.routes.side._load_parent_messages",
        lambda tid: _async(_PARENT_MESSAGES),
    )
    return threads


async def _async(value):
    return value


@pytest.fixture
def client(store, side_env):
    from fastapi import FastAPI

    from huginn.routes.side import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


class TestSideTaskRoutes:
    def test_blue_fork_registers_side_thread(self, client, store, side_env):
        resp = client.post(
            "/side/tasks",
            json={"parent_thread_id": "main", "from_message_index": 1},
        )
        body = resp.json()
        assert body["success"] is True
        task = body["task"]
        assert body["inherited_messages"] == 2
        assert side_env[task["thread_id"]]["kind"] == "side"

    def test_red_fork_missing_parent(self, client):
        body = client.post(
            "/side/tasks", json={"parent_thread_id": "ghost"}
        ).json()
        assert body["success"] is False
        assert "not found" in body["error"]

    def test_red_fork_bad_index(self, client):
        body = client.post(
            "/side/tasks",
            json={"parent_thread_id": "main", "from_message_index": 99},
        ).json()
        assert body["success"] is False
        assert "out of range" in body["error"]

    def test_blue_message_round_trip_persists_turns(self, client, store, side_env, monkeypatch):
        task_id = client.post(
            "/side/tasks", json={"parent_thread_id": "main"}
        ).json()["task"]["id"]

        async def fake_run(thread_id, content):
            return {
                "success": True,
                "answer": "4x4 足够",
                "artifacts": [{"path": "out/k.png", "tool": "file_write_tool"}],
            }

        monkeypatch.setattr("huginn.routes.side._run_side_agent", fake_run)
        body = client.post(
            f"/side/tasks/{task_id}/messages", json={"content": "4x4 够吗?"}
        ).json()
        assert body["success"] is True
        assert body["answer"] == "4x4 足够"
        got = store.get(task_id)
        assert [m["role"] for m in got.messages] == ["user", "assistant"]
        assert got.artifacts[0]["path"] == "out/k.png"

    def test_red_empty_answer_does_not_persist_turns(self, client, store, side_env, monkeypatch):
        task_id = client.post(
            "/side/tasks", json={"parent_thread_id": "main"}
        ).json()["task"]["id"]

        async def empty_run(thread_id, content):
            return {"success": False, "error": "side agent returned an empty answer"}

        monkeypatch.setattr("huginn.routes.side._run_side_agent", empty_run)
        body = client.post(
            f"/side/tasks/{task_id}/messages", json={"content": "q"}
        ).json()
        assert body["success"] is False
        assert store.get(task_id).messages == []

    def test_blue_promote_registers_task_thread_and_card(self, client, store, side_env):
        task_id = client.post(
            "/side/tasks", json={"parent_thread_id": "main"}
        ).json()["task"]["id"]
        body = client.post(f"/side/tasks/{task_id}/promote", json={"title": "正式任务"}).json()
        assert body["success"] is True
        assert body["thread_id"] == task_id
        assert body["task_card"]["title"] == "正式任务"
        assert side_env[task_id]["kind"] == "task"
        assert side_env[task_id]["task_card"]["task_id"] == task_id

    def test_red_promote_twice(self, client, store, side_env):
        task_id = client.post(
            "/side/tasks", json={"parent_thread_id": "main"}
        ).json()["task"]["id"]
        assert client.post(f"/side/tasks/{task_id}/promote", json={}).json()["success"] is True
        again = client.post(f"/side/tasks/{task_id}/promote", json={}).json()
        assert again["success"] is False
        assert "already promoted" in again["error"]

    def test_blue_discard_removes_thread_and_task(self, client, store, side_env):
        task_id = client.post(
            "/side/tasks", json={"parent_thread_id": "main"}
        ).json()["task"]["id"]
        body = client.delete(f"/side/tasks/{task_id}").json()
        assert body["success"] is True
        assert store.get(task_id) is None
        assert task_id not in side_env

    def test_blue_list_filters_by_status(self, client, store, side_env):
        a = client.post("/side/tasks", json={"parent_thread_id": "main"}).json()["task"]["id"]
        b = client.post("/side/tasks", json={"parent_thread_id": "main"}).json()["task"]["id"]
        client.post(f"/side/tasks/{b}/promote", json={})
        side_only = client.get("/side/tasks", params={"status": "side"}).json()
        assert [t["id"] for t in side_only["tasks"]] == [a]
