"""Security regression tests — path traversal + missing auth.

conftest.py sets HUGINN_DEV_MODE=1 (line 15) which bypasses all auth.
We use a fixture to clear it so auth dependencies actually enforce.
"""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def _app():
    from huginn.routes.viewer3d import router as v3d_router
    from huginn.routes.bot import router as bot_router
    app = FastAPI()
    app.include_router(v3d_router)
    app.include_router(bot_router)
    return app


@pytest.fixture
def enforced_auth(monkeypatch):
    """Clear HUGINN_DEV_MODE so require_api_key / require_admin_key run."""
    monkeypatch.delenv("HUGINN_DEV_MODE", raising=False)
    yield


# ── viewer3d auth ──────────────────────────────────────────────────


def test_viewer3d_load_requires_auth(enforced_auth, monkeypatch):
    """POST /viewer3d/load without API key header → 401."""
    monkeypatch.setenv("HUGINN_API_KEY", "secret")
    c = TestClient(_app())
    resp = c.post("/viewer3d/load", json={"content": "test"})
    assert resp.status_code == 401


def test_viewer3d_load_with_auth(enforced_auth, monkeypatch):
    """POST /viewer3d/load with correct API key → 200."""
    monkeypatch.setenv("HUGINN_API_KEY", "test-key")
    c = TestClient(_app())
    resp = c.post(
        "/viewer3d/load",
        json={"content": "2\n\nH 0 0 0\nH 0 0 1"},
        headers={"X-HUGINN-API-KEY": "test-key"},
    )
    assert resp.status_code == 200


def test_viewer3d_load_path_traversal_blocked(enforced_auth, monkeypatch):
    """POST /viewer3d/load with /etc/passwd → blocked."""
    monkeypatch.setenv("HUGINN_API_KEY", "test-key")
    c = TestClient(_app())
    resp = c.post(
        "/viewer3d/load",
        json={"file_path": "/etc/passwd"},
        headers={"X-HUGINN-API-KEY": "test-key"},
    )
    data = resp.json()
    assert "error" in data
    assert "workspace" in data["error"].lower()


def test_viewer3d_trajectory_requires_auth(enforced_auth, monkeypatch):
    """POST /viewer3d/trajectory without API key → 401."""
    monkeypatch.setenv("HUGINN_API_KEY", "secret")
    c = TestClient(_app())
    resp = c.post("/viewer3d/trajectory", json={"content": "test"})
    assert resp.status_code == 401


# ── bot auth ───────────────────────────────────────────────────────


def test_bot_status_requires_auth(enforced_auth, monkeypatch):
    """GET /bot/status without admin key → 403."""
    monkeypatch.setenv("HUGINN_ADMIN_API_KEY", "secret-admin")
    c = TestClient(_app())
    resp = c.get("/bot/status")
    assert resp.status_code == 403


def test_bot_start_requires_auth(enforced_auth, monkeypatch):
    """POST /bot/start without admin key → 403."""
    monkeypatch.setenv("HUGINN_ADMIN_API_KEY", "secret-admin")
    c = TestClient(_app())
    resp = c.post("/bot/start")
    assert resp.status_code == 403


def test_bot_config_requires_auth(enforced_auth, monkeypatch):
    """GET /bot/config without admin key → 403."""
    monkeypatch.setenv("HUGINN_ADMIN_API_KEY", "secret-admin")
    c = TestClient(_app())
    resp = c.get("/bot/config")
    assert resp.status_code == 403


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
