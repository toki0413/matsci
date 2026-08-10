"""会话超时/清理测试 — session 过期、JWT 过期、revocation list prune."""
from __future__ import annotations

import asyncio
import os
import sys
import time

import pytest

os.environ.pop("HUGINN_DEV_MODE", None)
os.environ["HUGINN_API_KEY"] = "sess-key-0123456789abcdef"
os.environ["HUGINN_JWT_SECRET"] = "sess-jwt-secret"
os.environ["HUGINN_RATE_LIMIT_PER_MINUTE"] = "0"


async def _noop():
    pass


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    ws = tmp_path_factory.mktemp("sess_ws")
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
    store._users["sess-user"] = User(user_id="sess-user", username="sess", role=Role.ADMIN, active=True, api_key_hash="", created_at=0)
    from fastapi.testclient import TestClient
    with TestClient(sm.app) as c:
        yield c


@pytest.fixture
def admin_token(app_client):
    from huginn.security.auth import create_token, get_user_store
    return create_token(get_user_store()._users["sess-user"], expires_in=3600)


def _bearer(t): return {"Authorization": f"Bearer {t}"}


class TestJWTExpiry:
    """JWT 过期机制."""

    def test_expired_jwt_rejected(self, app_client):
        """过期的 JWT 被拒绝."""
        from huginn.security.auth import create_token, get_user_store
        from huginn.security.rbac import Role, User
        store = get_user_store()
        # 创建一个已过期的 token (expires_in=-1, 即 1 秒前过期)
        token = create_token(store._users["sess-user"], expires_in=-1)
        resp = app_client.get("/memory/stats", headers=_bearer(token))
        assert resp.status_code == 401, f"expired JWT accepted: {resp.status_code}"

    def test_valid_jwt_accepted(self, app_client, admin_token):
        """有效 JWT 被接受."""
        resp = app_client.get("/memory/stats", headers=_bearer(admin_token))
        assert resp.status_code == 200

    def test_jwt_with_zero_expiry(self, app_client):
        """expires_in=0 的 JWT."""
        from huginn.security.auth import create_token, get_user_store
        store = get_user_store()
        token = create_token(store._users["sess-user"], expires_in=0)
        resp = app_client.get("/memory/stats", headers=_bearer(token))
        # 0 秒过期可能立即过期或刚好有效
        assert resp.status_code in (200, 401), f"unexpected: {resp.status_code}"


class TestTokenRevocation:
    """Token 吊销."""

    def test_revoke_token_blocks_access(self, app_client, admin_token):
        """吊销后 token 被拒绝."""
        from huginn.security.rbac import TokenRevocationList, jwt_decode
        # 解析 token 的 jti
        from huginn.security.auth import _jwt_secret
        claims = jwt_decode(admin_token, _jwt_secret())
        jti = claims.get("jti")
        assert jti, "no jti in token"

        # 吊销
        TokenRevocationList.shared().revoke(jti, exp=claims.get("exp"))

        # 使用吊销的 token
        resp = app_client.get("/memory/stats", headers=_bearer(admin_token))
        assert resp.status_code == 401, f"revoked token accepted: {resp.status_code}"

    def test_revocation_list_auto_prune(self):
        """吊销列表自动清理过期条目."""
        from huginn.security.rbac import TokenRevocationList
        rl = TokenRevocationList()

        # 添加一个已过期的条目
        rl.revoke("expired-jti", exp=time.time() - 3600)
        # 添加一个有效的条目
        rl.revoke("valid-jti", exp=time.time() + 3600)

        # 触发 prune
        count = rl.count()
        # 过期条目应被清理
        assert count == 1, f"expected 1 after prune, got {count}"
        assert rl.is_revoked("valid-jti")
        assert not rl.is_revoked("expired-jti")

    def test_clear_revocation_list(self):
        """clear 方法清空."""
        from huginn.security.rbac import TokenRevocationList
        rl = TokenRevocationList()
        rl.revoke("jti-1", exp=time.time() + 3600)
        rl.revoke("jti-2", exp=time.time() + 3600)
        assert rl.count() == 2
        rl.clear()
        assert rl.count() == 0


class TestSessionManager:
    """Session 管理器."""

    def test_create_and_get_session(self):
        from huginn.security.rbac import SessionManager
        sm = SessionManager(max_idle=3600)
        session = sm.create("user-1")
        assert session.user_id == "user-1"

        retrieved = sm.get(session.session_id)
        assert retrieved is not None
        assert retrieved.user_id == "user-1"

    def test_destroy_session(self):
        from huginn.security.rbac import SessionManager
        sm = SessionManager()
        session = sm.create("user-1")
        sm.destroy(session.session_id)
        assert sm.get(session.session_id) is None

    def test_expired_session_auto_cleanup(self):
        """过期 session 自动清理."""
        from huginn.security.rbac import SessionManager
        sm = SessionManager(max_idle=0.1)  # 0.1 秒超时
        session = sm.create("user-1")
        assert sm.get(session.session_id) is not None

        time.sleep(0.2)  # 等待过期
        assert sm.get(session.session_id) is None  # 过期后返回 None

    def test_destroy_user_sessions(self):
        """销毁用户所有 session."""
        from huginn.security.rbac import SessionManager
        sm = SessionManager()
        s1 = sm.create("user-1")
        s2 = sm.create("user-1")
        s3 = sm.create("user-2")

        count = sm.destroy_user_sessions("user-1")
        assert count == 2
        assert sm.get(s1.session_id) is None
        assert sm.get(s2.session_id) is None
        assert sm.get(s3.session_id) is not None

    def test_cleanup_expired(self):
        """批量清理过期 session."""
        from huginn.security.rbac import SessionManager
        sm = SessionManager(max_idle=0.1)
        sm.create("user-1")
        sm.create("user-2")
        time.sleep(0.2)
        cleaned = sm.cleanup_expired()
        assert cleaned == 2
        assert sm.active_count() == 0


class TestUserStore:
    """User store."""

    def test_get_nonexistent_user(self):
        from huginn.security.auth import get_user_store
        store = get_user_store()
        user = store.get_user("nonexistent-id")
        assert user is None

    def test_get_user_by_api_key_nonexistent(self):
        from huginn.security.auth import get_user_store
        store = get_user_store()
        user = store.get_user_by_api_key("nonexistent-key")
        assert user is None

    def test_inactive_user_rejected(self, app_client):
        """非 active 用户被拒绝."""
        from huginn.security.auth import create_token, get_user_store
        from huginn.security.rbac import Role, User
        store = get_user_store()
        store._users["inactive-user"] = User(
            user_id="inactive-user", username="inactive", role=Role.ADMIN,
            active=False, api_key_hash="", created_at=0,
        )
        token = create_token(store._users["inactive-user"], expires_in=3600)
        resp = app_client.get("/memory/stats", headers=_bearer(token))
        assert resp.status_code == 401, f"inactive user accepted: {resp.status_code}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-cov"]))
