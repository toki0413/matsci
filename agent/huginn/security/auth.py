"""Authentication/authorization dependencies for the Huginn API server.

Supports two authentication modes:

1. **API-key mode** (legacy): set ``HUGINN_API_KEY`` for a shared secret.
   Administrative endpoints additionally require ``HUGINN_ADMIN_API_KEY``.
2. **JWT + RBAC mode**: issue per-user JWT tokens via ``/auth/token``.
   Each token carries the user's role; capabilities are checked with
   ``require_capability()``.

Both modes can coexist — if a request carries a valid JWT it takes
precedence over the API key.  In dev mode (no keys configured) auth is
not enforced.
"""

from __future__ import annotations

import hmac
import os
from collections.abc import Callable
from typing import Any

from fastapi import HTTPException, Request, Security, WebSocket, status
from fastapi.security import APIKeyHeader

from huginn.security.rbac import (
    Role,
    SessionManager,
    User,
    jwt_decode,
    jwt_encode,
)

API_KEY_HEADER = APIKeyHeader(name="X-HUGINN-API-KEY", auto_error=False)
_BEARER_PREFIX = "Bearer "

# Paths that are intentionally public (health checks, OpenAPI docs, auth).
_PUBLIC_PATHS: set[str] = {
    "/health",
    "/health/rust",
    "/health/guidance",
    "/docs",
    "/openapi.json",
    "/redoc",
    "/auth/token",
    "/auth/login",
}

# Module-level singletons (lazy-init by ``get_user_store`` / ``get_session_manager``).
_user_store = None
_session_manager: SessionManager | None = None


def _jwt_secret() -> str | None:
    return os.environ.get("HUGINN_JWT_SECRET") or os.environ.get("HUGINN_API_KEY") or None


def get_user_store():
    """Return the global UserStore (creates on first call)."""
    global _user_store
    if _user_store is None:
        from huginn.security.user_store import UserStore
        _user_store = UserStore()
    return _user_store


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager


def _public_path(request: Request | WebSocket) -> bool:
    path = getattr(request, "url", None)
    if path is None:
        return False
    return path.path in _PUBLIC_PATHS


def _configured_api_key() -> str | None:
    return os.environ.get("HUGINN_API_KEY") or None


def _configured_admin_key() -> str | None:
    return os.environ.get("HUGINN_ADMIN_API_KEY") or None


def _api_key_set() -> bool:
    return bool(_configured_api_key())


# ---------------------------------------------------------------------------
# JWT helpers for route handlers
# ---------------------------------------------------------------------------

def create_token(
    user: User,
    *,
    expires_in: int = 3600,
) -> str:
    """Issue a JWT for *user*."""
    secret = _jwt_secret()
    if secret is None:
        raise RuntimeError("No JWT secret configured (set HUGINN_JWT_SECRET or HUGINN_API_KEY)")
    payload = {
        "sub": user.user_id,
        "username": user.username,
        "role": user.role.value,
    }
    return jwt_encode(payload, secret, expires_in=expires_in)


def _decode_token(token: str) -> dict[str, Any]:
    secret = _jwt_secret()
    if secret is None:
        raise ValueError("No JWT secret configured")
    return jwt_decode(token, secret)


def _extract_bearer(conn: Request | WebSocket) -> str | None:
    """Pull the Bearer token from the Authorization header."""
    auth = getattr(conn, "headers", {}).get("authorization", "")
    if isinstance(auth, bytes):
        auth = auth.decode("utf-8", errors="replace")
    if auth.startswith(_BEARER_PREFIX):
        return auth[len(_BEARER_PREFIX):]
    return None


# ---------------------------------------------------------------------------
# Request context — carries user info through the request lifecycle
# ---------------------------------------------------------------------------

class RequestContext:
    """Lightweight bag attached to each request by the auth dependency."""

    __slots__ = ("user", "api_key", "token", "auth_mode")

    def __init__(
        self,
        user: User | None = None,
        api_key: str = "",
        token: str = "",
        auth_mode: str = "none",
    ):
        self.user = user
        self.api_key = api_key
        self.token = token
        self.auth_mode = auth_mode  # "jwt" | "api_key" | "dev" | "none"


# ---------------------------------------------------------------------------
# FastAPI dependencies
# ---------------------------------------------------------------------------

def require_api_key(
    request: Request = None,  # type: ignore[assignment]
    websocket: WebSocket = None,  # type: ignore[assignment]
) -> str:
    """Require a valid API key **or** JWT for the request.

    In dev mode (``HUGINN_API_KEY`` unset and no JWT secret) this dependency
    allows any request.  In production either a matching API key or a valid
    JWT must be presented.
    """
    conn = request or websocket
    if conn is None:
        return ""

    if _public_path(conn):
        return ""

    # --- try JWT first ---------------------------------------------------
    bearer = _extract_bearer(conn)
    if bearer:
        try:
            claims = _decode_token(bearer)
            # Attach user context to request state
            store = get_user_store()
            user = store.get_user(claims["sub"])
            if user and user.active:
                ctx = RequestContext(user=user, token=bearer, auth_mode="jwt")
                try:
                    request.state.auth = ctx
                except Exception:
                    pass
                return bearer
        except (ValueError, KeyError):
            pass  # fall through to API key check

    # --- fall back to API key ---------------------------------------------
    configured = _configured_api_key()
    if configured is None:
        # Development mode: auth is not enforced.
        return ""

    provided = conn.headers.get("X-HUGINN-API-KEY")
    if not provided:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )

    # Check against user store first (per-user API keys)
    store = get_user_store()
    user = store.get_user_by_api_key(provided)
    if user and user.active:
        ctx = RequestContext(user=user, api_key=provided, auth_mode="api_key")
        try:
            request.state.auth = ctx
        except Exception:
            pass
        return provided

    # Legacy shared API key
    if secrets_match(provided, configured):
        return provided

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API key",
    )


def require_admin_key(
    request: Request,
    api_key: str | None = Security(API_KEY_HEADER),
) -> str:
    """Require the admin API key **or** an admin-role JWT for sensitive endpoints.

    In dev mode (no admin and no regular API key configured) auth is not
    enforced.
    """
    # Check JWT admin role first
    bearer = _extract_bearer(request)
    if bearer:
        try:
            claims = _decode_token(bearer)
            if claims.get("role") == Role.ADMIN.value:
                return bearer
        except (ValueError, KeyError):
            pass

    configured = _configured_admin_key()
    if configured is None:
        configured = _configured_api_key()

    if configured is None:
        return ""

    provided = request.headers.get("X-HUGINN-ADMIN-API-KEY") or api_key
    if not provided or not secrets_match(provided, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin API key",
        )
    return provided


def require_capability(capability: str) -> Callable[..., Any]:
    """Return a FastAPI dependency that asserts the caller has *capability*.

    Usage::

        @router.get("/admin/config", dependencies=[Depends(require_capability("config"))])
        async def get_config(): ...
    """

    def _check(request: Request) -> str:
        # Public paths bypass
        if _public_path(request):
            return ""

        ctx: RequestContext | None = getattr(request.state, "auth", None)

        # JWT path — check user capability
        if ctx and ctx.user:
            if not ctx.user.active:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="User account is deactivated",
                )
            if not ctx.user.can(capability):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{ctx.user.role.value}' lacks capability '{capability}'",
                )
            return ctx.token or ctx.api_key

        # Dev mode (no auth configured)
        if _configured_api_key() is None and _jwt_secret() is None:
            return ""

        # Legacy API-key mode — treat as operator-equivalent
        provided = request.headers.get("X-HUGINN-API-KEY")
        configured = _configured_api_key()
        if provided and configured and secrets_match(provided, configured):
            return provided

        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Capability '{capability}' required",
        )

    return _check


def require_api_key_dependency() -> Callable[..., Any]:
    """Return a FastAPI dependency that validates the API key."""
    return require_api_key


def secrets_match(a: str, b: str) -> bool:
    """Constant-time string comparison to mitigate timing attacks."""
    return hmac.compare_digest(a, b)
