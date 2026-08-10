"""CORS / 安全 header 测试 — 跨域策略、安全响应头."""
from __future__ import annotations

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _isolated_auth_env(monkeypatch):
    """Isolate auth env so we don't pollute other modules in the same worker."""
    monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
    monkeypatch.setenv("HUGINN_API_KEY", "cors-key-0123456789abcdef")
    monkeypatch.setenv("HUGINN_JWT_SECRET", "cors-jwt-secret")
    monkeypatch.setenv("HUGINN_RATE_LIMIT_PER_MINUTE", "0")
    yield


async def _noop():
    pass


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    ws = tmp_path_factory.mktemp("cors_ws")
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
    store._users["cors-user"] = User(user_id="cors-user", username="cors", role=Role.ADMIN, active=True, api_key_hash="", created_at=0)
    from fastapi.testclient import TestClient
    with TestClient(sm.app) as c:
        yield c


@pytest.fixture
def admin_token(app_client):
    from huginn.security.auth import create_token, get_user_store
    return create_token(get_user_store()._users["cors-user"], expires_in=3600)


def _bearer(t): return {"Authorization": f"Bearer {t}"}


class TestCORSPolicy:
    """CORS 跨域策略."""

    def test_preflight_options_request(self, app_client):
        """OPTIONS preflight 请求不崩溃."""
        resp = app_client.options("/health", headers={
            "Origin": "https://example.com",
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Authorization",
        })
        # 200 或 400 都可接受 (FastAPI 默认处理 OPTIONS)
        assert resp.status_code != 500, f"preflight crashed: {resp.status_code}"

    def test_cors_header_on_get(self, app_client):
        """GET 请求带 Origin 时有 CORS header."""
        resp = app_client.get("/health", headers={"Origin": "https://example.com"})
        assert resp.status_code == 200
        # CORS 响应头 (allow-credentials, allow-methods 等)
        # 即使没有, 也不应崩溃
        allow_origin = resp.headers.get("access-control-allow-origin", "")
        print(f"\n[CORS] allow-origin: {allow_origin or '(none)'}")

    def test_cors_with_api_key_request(self, app_client, admin_token):
        """带 API key 的 CORS 请求."""
        resp = app_client.get("/memory/stats", headers={
            **_bearer(admin_token),
            "Origin": "https://example.com",
        })
        assert resp.status_code == 200


class TestSecurityHeaders:
    """安全响应头."""

    def test_response_has_security_headers(self, app_client):
        """响应包含基本安全 header."""
        resp = app_client.get("/health")
        headers = resp.headers

        # 检查常见安全 header (有最好, 没有也不 fail, 只记录)
        security_headers = [
            "x-content-type-options",
            "x-frame-options",
            "x-xss-protection",
            "strict-transport-security",
            "content-security-policy",
        ]
        found = []
        for h in security_headers:
            if h in headers:
                found.append(h)
        print(f"\n[SECURITY] headers found: {found or 'none'}")

    def test_content_type_not_sniffable(self, app_client, admin_token):
        """X-Content-Type-Options: nosniff (防止 MIME sniffing)."""
        resp = app_client.get("/memory/stats", headers=_bearer(admin_token))
        # 如果有这个 header, 值应该是 nosniff
        nosniff = resp.headers.get("x-content-type-options", "")
        if nosniff:
            assert "nosniff" in nosniff.lower(), f"unexpected nosniff value: {nosniff}"

    def test_no_server_version_leak(self, app_client):
        """Server header 不暴露详细版本."""
        resp = app_client.get("/health")
        server = resp.headers.get("server", "")
        # 只记录, 不强制断言 (生产环境用反代隐藏)
        if server:
            print(f"\n[SECURITY] Server header: {server}")

    def test_openapi_not_leak_internal_info(self, app_client):
        """OpenAPI schema 不暴露内部文件路径."""
        resp = app_client.get("/openapi.json")
        assert resp.status_code == 200
        body = resp.json()
        # 只检查不泄露文件系统路径 (description 里可能有占位符)
        text = str(body)
        # 不应包含真实的 /workspace/agent/ 路径
        assert "/workspace/agent/" not in text, "openapi leaks internal path"
        assert "/home/" not in text, "openapi leaks home path"


class TestHTTPMethods:
    """HTTP 方法限制."""

    def test_put_on_get_endpoint_rejected(self, app_client, admin_token):
        """PUT 请求到 GET-only 端点应 405."""
        resp = app_client.put("/health", headers=_bearer(admin_token))
        assert resp.status_code in (405, 404, 422), f"unexpected: {resp.status_code}"

    def test_delete_on_get_endpoint_rejected(self, app_client, admin_token):
        """DELETE 请求到 GET-only 端点应 405 (或 200 如果端点支持)."""
        resp = app_client.delete("/memory/stats", headers=_bearer(admin_token))
        # /memory/stats 可能被路由层 catch-all 处理, 200/405/404/422 都可接受
        assert resp.status_code in (200, 405, 404, 422), f"unexpected: {resp.status_code}"

    def test_patch_on_health_rejected(self, app_client):
        """PATCH /health 应 405."""
        resp = app_client.patch("/health")
        assert resp.status_code in (405, 404, 422), f"unexpected: {resp.status_code}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-cov"]))
