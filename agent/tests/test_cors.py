"""Tests for CORS tightening."""

from __future__ import annotations

import os

from huginn.server import _get_cors_origins


def test_default_cors_origins_are_local():
    """Default origins should be local-only."""
    # Ensure env override is not present
    os.environ.pop("HUGINN_CORS_ORIGINS", None)
    origins = _get_cors_origins()
    assert "http://localhost:3000" in origins
    assert "http://localhost:1420" in origins
    assert "tauri://localhost" in origins
    assert "*" not in origins


def test_env_override(monkeypatch):
    monkeypatch.setenv("HUGINN_CORS_ORIGINS", "https://app.example.com, https://other.example.com")
    origins = _get_cors_origins()
    assert origins == ["https://app.example.com", "https://other.example.com"]
