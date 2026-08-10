"""错误处理边界测试 — 空 body、超长字段、深嵌套、重复字段、错误 Content-Type."""
from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _isolated_auth_env(monkeypatch):
    """Isolate auth env so we don't pollute other modules in the same worker."""
    monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
    monkeypatch.setenv("HUGINN_API_KEY", "err-key-0123456789abcdef")
    monkeypatch.setenv("HUGINN_JWT_SECRET", "err-jwt-secret")
    monkeypatch.setenv("HUGINN_RATE_LIMIT_PER_MINUTE", "0")
    yield


async def _noop():
    pass


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    ws = tmp_path_factory.mktemp("err_ws")
    os.environ["HUGINN_WORKSPACE"] = str(ws)
    import huginn.server as sm
    sm._init_mcp_tools = _noop
    sm._shutdown_mcp = _noop
    from huginn.config import HuginnConfig
    from huginn.server_context import create_server_context, set_server_context
    ctx = create_server_context(HuginnConfig(provider="ollama", model="test", workspace=str(ws)))
    set_server_context(ctx)
    sm._context = ctx
    from huginn.security.auth import get_user_store
    from huginn.security.rbac import Role, User
    store = get_user_store()
    store._users["err-user"] = User(user_id="err-user", username="err", role=Role.ADMIN, active=True, api_key_hash="", created_at=0)
    from fastapi.testclient import TestClient
    with TestClient(sm.app) as c:
        yield c


@pytest.fixture
def admin_token(app_client):
    from huginn.security.auth import create_token, get_user_store
    return create_token(get_user_store()._users["err-user"], expires_in=3600)


def _bearer(t): return {"Authorization": f"Bearer {t}"}


class TestEmptyAndMissingBody:
    """空 body / 缺失字段."""

    def test_post_memory_empty_body(self, app_client, admin_token):
        """POST /memory 无 body."""
        resp = app_client.post("/memory", headers=_bearer(admin_token))
        assert resp.status_code != 500, f"crashed on empty body: {resp.status_code}"

    def test_post_memory_empty_json(self, app_client, admin_token):
        """POST /memory 空 JSON."""
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={})
        assert resp.status_code != 500, f"crashed on empty json: {resp.status_code}"

    def test_post_memory_missing_required_field(self, app_client, admin_token):
        """缺 content 字段."""
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={"category": "fact"})
        assert resp.status_code != 500

    def test_post_threads_empty_body(self, app_client, admin_token):
        resp = app_client.post("/threads", headers=_bearer(admin_token))
        assert resp.status_code != 500


class TestOversizedFields:
    """超长字段."""

    def test_memory_content_100kb(self, app_client, admin_token):
        """100KB content."""
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": "x" * 100000, "category": "fact",
        })
        assert resp.status_code != 500

    def test_memory_tags_1000_items(self, app_client, admin_token):
        """1000 个 tag."""
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": "many tags", "category": "fact", "tags": [f"tag-{i}" for i in range(1000)],
        })
        assert resp.status_code != 500

    def test_memory_search_query_10kb(self, app_client, admin_token):
        """10KB query."""
        resp = app_client.post("/memory/search", headers=_bearer(admin_token), json={
            "query": "q" * 10000, "top_k": 5,
        })
        assert resp.status_code != 500

    def test_thread_title_10kb(self, app_client, admin_token):
        """10KB 标题."""
        resp = app_client.post("/threads", headers=_bearer(admin_token), json={
            "title": "T" * 10000,
        })
        assert resp.status_code != 500


class TestDeepNesting:
    """深嵌套 JSON."""

    def test_deeply_nested_json(self, app_client, admin_token):
        """100 层嵌套."""
        # 构造 {"a": {"a": {"a": ... 100 层 ...}}}
        nested = "value"
        for _ in range(100):
            nested = {"a": nested}
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": "test", "category": "fact", "metadata": nested,
        })
        assert resp.status_code != 500

    def test_array_of_arrays(self, app_client, admin_token):
        """嵌套数组."""
        arr = list(range(100))
        nested_arr = [arr] * 100
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": "test", "category": "fact", "metadata": {"data": nested_arr},
        })
        assert resp.status_code != 500


class TestWrongContentTypes:
    """错误 Content-Type."""

    def test_post_with_text_plain(self, app_client, admin_token):
        """用 text/plain 发 JSON body."""
        resp = app_client.post("/memory", headers={
            **_bearer(admin_token), "Content-Type": "text/plain",
        }, data='{"content": "test", "category": "fact"}')
        assert resp.status_code != 500

    def test_post_with_form_urlencoded(self, app_client, admin_token):
        """用 form-urlencoded."""
        resp = app_client.post("/memory", headers={
            **_bearer(admin_token), "Content-Type": "application/x-www-form-urlencoded",
        }, data="content=test&category=fact")
        assert resp.status_code != 500

    def test_post_with_no_content_type(self, app_client, admin_token):
        """无 Content-Type."""
        resp = app_client.post("/memory", headers=_bearer(admin_token),
                               data='{"content": "test", "category": "fact"}')
        assert resp.status_code != 500


class TestTypeConfusion:
    """类型混淆."""

    def test_string_instead_of_int(self, app_client, admin_token):
        """importance 传字符串."""
        resp = app_client.post("/memory", headers=_bearer(admin_token), json={
            "content": "test", "category": "fact", "importance": "not a number",
        })
        assert resp.status_code != 500

    def test_array_instead_of_object(self, app_client, admin_token):
        """传数组而不是对象."""
        resp = app_client.post("/memory", headers=_bearer(admin_token), json=[1, 2, 3])
        assert resp.status_code != 500

    def test_string_instead_of_object(self, app_client, admin_token):
        """传字符串而不是对象."""
        resp = app_client.post("/memory", headers=_bearer(admin_token), json="just a string")
        assert resp.status_code != 500

    def test_number_instead_of_object(self, app_client, admin_token):
        """传数字."""
        resp = app_client.post("/memory", headers=_bearer(admin_token), json=42)
        assert resp.status_code != 500

    def test_null_body(self, app_client, admin_token):
        """传 null."""
        resp = app_client.post("/memory", headers=_bearer(admin_token), json=None)
        assert resp.status_code != 500

    def test_boolean_body(self, app_client, admin_token):
        """传 boolean."""
        resp = app_client.post("/memory", headers=_bearer(admin_token), json=True)
        assert resp.status_code != 500


class TestInvalidPathParams:
    """无效路径参数."""

    def test_memory_id_with_spaces(self, app_client, admin_token):
        """memory_id 含空格."""
        resp = app_client.delete("/memory/space in id", headers=_bearer(admin_token))
        assert resp.status_code != 500

    def test_memory_id_with_special_chars(self, app_client, admin_token):
        """memory_id 含特殊字符."""
        resp = app_client.delete("/memory/!@#$%^&*()", headers=_bearer(admin_token))
        assert resp.status_code != 500

    def test_thread_id_with_sql_injection(self, app_client, admin_token):
        """thread_id SQL 注入."""
        resp = app_client.get("/threads/'; DROP TABLE threads;--", headers=_bearer(admin_token))
        assert resp.status_code != 500

    def test_path_traversal_in_id(self, app_client, admin_token):
        """路径遍历在 ID 中."""
        resp = app_client.get("/threads/../../etc/passwd", headers=_bearer(admin_token))
        assert resp.status_code != 500


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-cov"]))
