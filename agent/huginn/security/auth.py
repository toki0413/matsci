"""Authentication/authorization dependencies for the Huginn API server.

Production deployments must set ``HUGINN_API_KEY``. Administrative endpoints
(for runtime configuration) additionally require ``HUGINN_ADMIN_API_KEY``.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Security, WebSocket, status
from fastapi.security import APIKeyHeader

API_KEY_HEADER = APIKeyHeader(name="X-HUGINN-API-KEY", auto_error=False)

# Paths that are intentionally public (health checks, OpenAPI docs).
_PUBLIC_PATHS: set[str] = {
    "/health",
    "/health/rust",
    "/docs",
    "/openapi.json",
    "/redoc",
}


def _public_path(request: Request) -> bool:
    return request.url.path in _PUBLIC_PATHS


def _configured_api_key() -> str | None:
    return os.environ.get("HUGINN_API_KEY") or None


def _configured_admin_key() -> str | None:
    return os.environ.get("HUGINN_ADMIN_API_KEY") or None


def _api_key_set() -> bool:
    return bool(_configured_api_key())


def require_api_key(
    request: Request = None,  # type: ignore[assignment]
    websocket: WebSocket = None,  # type: ignore[assignment]
) -> str:
    """Require a valid API key for the request.

    In dev mode (``HUGINN_API_KEY`` unset) this dependency allows any request.
    In production the key must match the configured value.

    Accepts either an HTTP ``Request`` or a ``WebSocket`` object so this
    dependency can be applied globally (including the WebSocket route).

    The parameters are intentionally annotated as non-optional with a ``None``
    default. FastAPI dependency injection introspects the signature and gets
    confused by ``Request | None``, but it correctly injects the active
    connection type when the class is left un-unioned.
    """
    conn = request or websocket
    if conn is None:
        return ""

    if _public_path(conn):
        return ""

    configured = _configured_api_key()
    if configured is None:
        # Development mode: auth is not enforced.
        return ""

    provided = conn.headers.get("X-HUGINN-API-KEY")
    if not provided or not secrets_match(provided, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
    return provided


def require_admin_key(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
) -> str:
    """Require the admin API key for sensitive endpoints.

    In dev mode (no admin and no regular API key configured) auth is not
    enforced.
    """
    configured = _configured_admin_key()
    if configured is None:
        # Fall back to the regular API key if no admin key is set.
        configured = _configured_api_key()

    if configured is None:
        # Development mode: auth is not enforced.
        return ""

    provided = request.headers.get("X-HUGINN-ADMIN-API-KEY") or api_key
    if not provided or not secrets_match(provided, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin API key",
        )
    return provided


def require_api_key_dependency() -> Callable[..., Any]:
    """Return a FastAPI dependency that validates the API key.

    This wrapper is convenient for adding auth to individual routers/endpoints.
    """
    return require_api_key


def secrets_match(a: str, b: str) -> bool:
    """Constant-time string comparison to mitigate timing attacks."""
    return hmac.compare_digest(a, b)
