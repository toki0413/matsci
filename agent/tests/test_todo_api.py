"""/todos API 端点 + todo 持久化自检.

不依赖 LLM / 外部服务. 验证: 空桶读取、PUT 写入后 GET 读回、落盘后
进程内重载数据不丢、鉴权保护存在.
"""
from __future__ import annotations

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    """TestClient with a valid API key (dev mode doesn't exempt testserver)."""
    from fastapi.testclient import TestClient

    import huginn.server as server_module

    monkeypatch.setenv("HUGINN_DEV_MODE", "1")
    monkeypatch.setenv("HUGINN_API_KEY", "test-todo-key")
    with TestClient(server_module.app) as c:
        yield c


@pytest.fixture
def auth():
    return {"X-HUGINN-API-KEY": "test-todo-key"}


# 测试模块级 store, 隔离 HUGINN_CACHE_DIR 避免污染用户目录


@pytest.fixture
def todo_env(tmp_path, monkeypatch):
    import huginn.tools.todo_tool as tt

    monkeypatch.setenv("HUGINN_CACHE_DIR", str(tmp_path))
    # 重置模块级状态, 让 _load/_save 重新绑定到 tmp 路径
    tt._store_path = None
    tt._TODO_STORE.clear()
    return tt


def test_get_todos_empty(todo_env):
    assert todo_env.get_todos("no-such-session") == []


def test_set_get_roundtrip(todo_env):
    todo_env.set_todos("s1", [
        {"content": "a", "status": "pending"},
        {"content": "b", "status": "completed"},
    ])
    todos = todo_env.get_todos("s1")
    assert len(todos) == 2
    assert todos[1]["status"] == "completed"
    # 其它桶隔离
    assert todo_env.get_todos("s2") == []


def test_persistence_across_reload(todo_env):
    """写入落盘后, 重置模块状态重新加载, 数据仍在 (模拟进程重启)."""
    todo_env.set_todos("persist", [{"content": "keep me", "status": "in_progress"}])
    # 模拟重启: 清内存 + 重置路径, 下次访问从文件读
    todo_env._TODO_STORE.clear()
    todo_env._store_path = None
    todos = todo_env.get_todos("persist")
    assert len(todos) == 1
    assert todos[0]["content"] == "keep me"


def test_todos_api_endpoints(todo_env, client, auth):
    # 先清空该桶, 保证断言干净
    client.put("/todos", headers=auth, params={"session_id": "api-s"}, json={"todos": []})
    r = client.get("/todos", headers=auth, params={"session_id": "api-s"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 0
    assert body["todos"] == []

    r = client.put(
        "/todos",
        headers=auth,
        params={"session_id": "api-s"},
        json={"todos": [{"content": "x", "status": "pending"}]},
    )
    assert r.status_code == 200
    assert r.json()["total"] == 1

    r = client.get("/todos", headers=auth, params={"session_id": "api-s"})
    assert r.json()["todos"][0]["content"] == "x"

    # 勾选完成
    r = client.put(
        "/todos",
        headers=auth,
        params={"session_id": "api-s"},
        json={"todos": [{"content": "x", "status": "completed"}]},
    )
    assert r.json()["completed"] == 1


def test_todos_api_requires_auth(todo_env, monkeypatch):
    """无 dev mode 且无 key 时 /todos 应被鉴权拦截."""
    from fastapi.testclient import TestClient

    import huginn.server as server_module

    monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
    monkeypatch.delenv("HUGINN_ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("HUGINN_API_KEY", raising=False)
    with TestClient(server_module.app) as c:
        r = c.get("/todos")
        assert r.status_code in (401, 403)
