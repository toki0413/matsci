"""Tests for API authentication dependencies."""

from __future__ import annotations

import pytest
from fastapi import HTTPException, Request

from huginn.security.auth import require_admin_key, require_api_key, secrets_match


def _make_request(
    path: str = "/tools", headers: dict[str, str] | None = None
) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [
            (k.lower().encode("latin-1"), v.encode("latin-1"))
            for k, v in (headers or {}).items()
        ],
    }
    return Request(scope)


class TestSecretsMatch:
    def test_equal_strings(self):
        assert secrets_match("abc", "abc") is True

    def test_different_strings(self):
        assert secrets_match("abc", "def") is False

    def test_different_lengths(self):
        assert secrets_match("abc", "abcd") is False


class TestRequireApiKey:
    def test_public_path_allows_without_key(self, monkeypatch):
        monkeypatch.delenv("HUGINN_API_KEY", raising=False)
        req = _make_request("/health")
        assert require_api_key(req, None) == ""

    def test_dev_mode_allows_without_key(self, monkeypatch):
        monkeypatch.delenv("HUGINN_API_KEY", raising=False)
        monkeypatch.setenv("HUGINN_DEV_MODE", "1")
        req = _make_request("/tools")
        assert require_api_key(req, None) == ""

    def test_valid_key(self, monkeypatch):
        monkeypatch.setenv("HUGINN_API_KEY", "secret")
        req = _make_request("/tools", {"X-HUGINN-API-KEY": "secret"})
        assert require_api_key(req, None) == "secret"

    def test_invalid_key(self, monkeypatch):
        monkeypatch.setenv("HUGINN_API_KEY", "secret")
        req = _make_request("/tools", {"X-HUGINN-API-KEY": "wrong"})
        with pytest.raises(HTTPException) as exc:
            require_api_key(req, None)
        assert exc.value.status_code == 401


class TestRequireAdminKey:
    def test_dev_mode_allows_without_key(self, monkeypatch):
        monkeypatch.delenv("HUGINN_API_KEY", raising=False)
        monkeypatch.delenv("HUGINN_ADMIN_API_KEY", raising=False)
        monkeypatch.setenv("HUGINN_DEV_MODE", "1")
        req = _make_request("/config")
        assert require_admin_key(req, None) == ""

    def test_admin_key_required(self, monkeypatch):
        monkeypatch.setenv("HUGINN_ADMIN_API_KEY", "admin-secret")
        req = _make_request("/config", {"X-HUGINN-ADMIN-API-KEY": "admin-secret"})
        assert require_admin_key(req, None) == "admin-secret"

    def test_invalid_admin_key(self, monkeypatch):
        monkeypatch.setenv("HUGINN_ADMIN_API_KEY", "admin-secret")
        req = _make_request("/config", {"X-HUGINN-ADMIN-API-KEY": "wrong"})
        with pytest.raises(HTTPException) as exc:
            require_admin_key(req, None)
        assert exc.value.status_code == 401
