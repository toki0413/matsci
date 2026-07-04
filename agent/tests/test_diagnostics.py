"""Tests for the unified diagnostics endpoint."""

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Minimal FastAPI app with just diagnostics routes."""
    from fastapi import FastAPI
    from huginn.routes.diagnostics import router

    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_diagnostics_returns_full_report(client):
    r = client.get("/diagnostics")
    assert r.status_code == 200
    data = r.json()
    assert "timestamp" in data
    assert "system" in data
    assert "tools" in data
    assert "circuit_breakers" in data
    assert "telemetry" in data
    assert "plugins" in data
    assert "verdict" in data


def test_diagnostics_verdict_is_valid(client):
    r = client.get("/diagnostics")
    data = r.json()
    assert data["verdict"] in ("healthy", "degraded", "unhealthy")


def test_diagnostics_tools_endpoint(client):
    r = client.get("/diagnostics/tools")
    assert r.status_code == 200
    assert "tools" in r.json()


def test_diagnostics_circuit_endpoint(client):
    r = client.get("/diagnostics/circuit")
    assert r.status_code == 200
    assert "circuit_breakers" in r.json()


def test_diagnostics_trace_endpoint(client):
    r = client.get("/diagnostics/trace")
    assert r.status_code == 200
    assert "trace" in r.json()
    assert isinstance(r.json()["trace"], list)


def test_diagnostics_includes_pid(client):
    r = client.get("/diagnostics")
    data = r.json()
    assert "pid" in data
    assert data["pid"] > 0


def test_diagnostics_thread_count(client):
    r = client.get("/diagnostics")
    data = r.json()
    assert "thread_count" in data
    assert data["thread_count"] > 0
