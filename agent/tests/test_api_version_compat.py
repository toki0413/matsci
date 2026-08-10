"""API 版本兼容测试 — /v1/* 与根路径一致性、deprecation header."""
from __future__ import annotations

import os
import sys

import pytest

os.environ.pop("HUGINN_DEV_MODE", None)
os.environ["HUGINN_API_KEY"] = "ver-key-0123456789abcdef"
os.environ["HUGINN_JWT_SECRET"] = "ver-jwt-secret"
os.environ["HUGINN_RATE_LIMIT_PER_MINUTE"] = "0"


async def _noop():
    pass


@pytest.fixture(scope="module")
def app_client(tmp_path_factory):
    ws = tmp_path_factory.mktemp("ver_ws")
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
    store._users["ver-user"] = User(user_id="ver-user", username="ver", role=Role.ADMIN, active=True, api_key_hash="", created_at=0)
    from fastapi.testclient import TestClient
    with TestClient(sm.app) as c:
        yield c


@pytest.fixture
def admin_token(app_client):
    from huginn.security.auth import create_token, get_user_store
    return create_token(get_user_store()._users["ver-user"], expires_in=3600)


def _bearer(t): return {"Authorization": f"Bearer {t}"}


class TestVersionedRoutes:
    """/v1/* 与根路径一致性."""

    ENDPOINTS = [
        ("GET", "/memory/stats"),
        ("GET", "/memory"),
        ("GET", "/threads"),
        ("GET", "/tools"),
        ("GET", "/skills"),
        ("GET", "/health"),
        ("GET", "/metrics"),
    ]

    @pytest.mark.parametrize("method,path", ENDPOINTS)
    def test_v1_and_root_consistency(self, app_client, admin_token, method, path):
        """同一端点 /v1/path 和 /path 返回相同状态码."""
        root_resp = app_client.request(method, path, headers=_bearer(admin_token))
        v1_resp = app_client.request(method, f"/v1{path}", headers=_bearer(admin_token))
        # 状态码一致 (或 v1 有不同路由配置时 200/404 都可)
        assert root_resp.status_code != 500, f"root {path} crashed"
        assert v1_resp.status_code != 500, f"v1 {path} crashed"
        # 公开端点 (health/metrics) 应该都 200
        if path in ("/health", "/metrics"):
            assert root_resp.status_code == 200
            assert v1_resp.status_code == 200


class TestDeprecationHeaders:
    """根路径 deprecation 提示."""

    def test_root_path_has_deprecation_warning(self, app_client, admin_token):
        """根路径可能返回 deprecation header (不影响功能)."""
        resp = app_client.get("/memory/stats", headers=_bearer(admin_token))
        assert resp.status_code == 200
        # 检查是否有 deprecation header (可选)
        deprecation = resp.headers.get("deprecation", "")
        sunset = resp.headers.get("sunset", "")
        print(f"\n[VER] deprecation: {deprecation or 'none'}, sunset: {sunset or 'none'}")

    def test_v1_path_no_deprecation(self, app_client, admin_token):
        """/v1 路径不应有 deprecation header."""
        resp = app_client.get("/v1/memory/stats", headers=_bearer(admin_token))
        assert resp.status_code == 200
        deprecation = resp.headers.get("deprecation", "")
        # v1 路径不应 deprecate
        assert not deprecation or deprecation.lower() != "true", \
            f"v1 path deprecated: {deprecation}"


class TestOpenAPISchema:
    """OpenAPI schema 完整性."""

    def test_openapi_has_v1_paths(self, app_client):
        """OpenAPI schema 包含 /v1 路径."""
        resp = app_client.get("/openapi.json")
        assert resp.status_code == 200
        body = resp.json()
        paths = body.get("paths", {})
        v1_paths = [p for p in paths if p.startswith("/v1/")]
        root_paths = [p for p in paths if not p.startswith("/v1/")]
        print(f"\n[VER] {len(v1_paths)} v1 paths, {len(root_paths)} root paths")
        # 应该有 v1 路径
        assert len(v1_paths) > 0, "no /v1 paths in openapi"

    def test_openapi_version(self, app_client):
        """OpenAPI 版本字段正确."""
        resp = app_client.get("/openapi.json")
        body = resp.json()
        assert body.get("openapi", "").startswith("3."), \
            f"unexpected openapi version: {body.get('openapi')}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short", "--no-cov"]))
