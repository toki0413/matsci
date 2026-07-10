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

import asyncio
import hmac
import logging
import os
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import HTTPException, Request, Security, WebSocket, status
from fastapi.security import APIKeyHeader

from huginn.security.rbac import (
    Role,
    SessionManager,
    TokenRevocationList,
    User,
    jwt_decode,
    jwt_encode,
)

API_KEY_HEADER = APIKeyHeader(name="X-HUGINN-API-KEY", auto_error=False)
_BEARER_PREFIX = "Bearer "

# Paths that are intentionally public (health checks, OpenAPI docs, auth,
# Prometheus scrape). Interactive docs are dropped in production so the
# API surface isn't exposed — see _hide_docs below.
_hide_docs = (
    os.environ.get("HUGINN_ENV", "").lower() == "production"
    or os.environ.get("HUGINN_HIDE_DOCS", "").lower() in ("1", "true", "yes")
)

_PUBLIC_PATHS: set[str] = {
    "/health",
    "/ready",
    "/health/rust",
    "/health/guidance",
    "/metrics",
    "/diagnostics",
    "/diagnostics/tools",
    "/diagnostics/circuit",
    "/diagnostics/trace",
    "/auth/token",
    "/auth/login",
    "/auth/me",
    "/auth/refresh",
}
if not _hide_docs:
    _PUBLIC_PATHS |= {"/docs", "/openapi.json", "/redoc"}

logger = logging.getLogger(__name__)

# Module-level singletons (lazy-init by ``get_user_store`` / ``get_session_manager``).
_user_store = None
_session_manager: SessionManager | None = None
_singleton_lock = threading.Lock()


def _jwt_secret() -> str | None:
    return os.environ.get("HUGINN_JWT_SECRET") or os.environ.get("HUGINN_API_KEY") or None


def get_user_store():
    """Return the global UserStore (creates on first call, thread-safe)."""
    global _user_store
    if _user_store is None:
        with _singleton_lock:
            if _user_store is None:
                from huginn.security.user_store import UserStore
                _user_store = UserStore()
    return _user_store


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        with _singleton_lock:
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
    """Issue a JWT for *user*.

    Includes a unique ``jti`` (JWT ID) claim so the token can be
    individually revoked via :class:`TokenRevocationList`.
    """
    import secrets as _secrets

    secret = _jwt_secret()
    if secret is None:
        raise RuntimeError("No JWT secret configured (set HUGINN_JWT_SECRET or HUGINN_API_KEY)")
    payload = {
        "sub": user.user_id,
        "username": user.username,
        "role": user.role.value,
        "jti": _secrets.token_urlsafe(16),  # unique ID for revocation
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

    # Dev mode bypass: when explicitly enabled, skip all auth checks.
    # This is used by the desktop app to allow unauthenticated local access
    # even after the user configures an API key for the LLM provider.
    if os.environ.get("HUGINN_DEV_MODE", "").lower() in ("1", "true", "yes"):
        return ""

    # --- try JWT first ---------------------------------------------------
    bearer = _extract_bearer(conn)
    if bearer:
        try:
            claims = _decode_token(bearer)
            # Check revocation list (logout)
            jti = claims.get("jti")
            if jti and TokenRevocationList.shared().is_revoked(jti):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Token has been revoked",
                )
            # Attach user context to request state
            store = get_user_store()
            user = store.get_user(claims["sub"])
            if user and user.active:
                ctx = RequestContext(user=user, token=bearer, auth_mode="jwt")
                try:
                    request.state.auth = ctx
                except Exception as exc:
                    logger.debug("Could not attach auth context: %s", exc)
                return bearer
        except (ValueError, KeyError):
            pass  # fall through to API key check

    # --- fall back to API key ---------------------------------------------
    configured = _configured_api_key()
    if configured is None:
        # Development mode: auth is not enforced, but only when explicitly
        # enabled via HUGINN_DEV_MODE=1. This prevents accidental exposure
        # in production environments where no API key was configured.
        if os.environ.get("HUGINN_DEV_MODE", "").lower() in ("1", "true", "yes"):
            return ""
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No API key configured. Set HUGINN_API_KEY or enable "
            "HUGINN_DEV_MODE=1 for local development.",
        )

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
        except Exception as exc:
            logger.debug("Could not attach auth context: %s", exc)
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
    # Dev mode bypass: skip all auth checks when explicitly enabled.
    if os.environ.get("HUGINN_DEV_MODE", "").lower() in ("1", "true", "yes"):
        return ""

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
        # Dev mode: only allow without auth when explicitly enabled
        if os.environ.get("HUGINN_DEV_MODE", "").lower() in ("1", "true", "yes"):
            return ""
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No admin key configured. Set HUGINN_ADMIN_API_KEY or enable "
            "HUGINN_DEV_MODE=1 for local development.",
        )

    provided = request.headers.get("X-HUGINN-ADMIN-API-KEY") or api_key
    if not provided or not secrets_match(provided, configured):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required. Provide a valid admin API key "
            "via the X-HUGINN-ADMIN-API-KEY header.",
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


# ---------------------------------------------------------------------------
# SWR 缓存 + 401 去重（参考 Claude Code 的 utils/auth.ts）
# ---------------------------------------------------------------------------

# API key SWR 缓存：source -> (key, expires_at)
_api_key_cache: dict[str, tuple[str, float]] = {}
_api_key_lock = asyncio.Lock()

# 401 去重：同一 token 的多个并发 401 只触发一次刷新
_pending_401: dict[str, asyncio.Task[bool]] = {}


async def get_api_key_with_swr(
    source: str,
    helper: Callable[[], Awaitable[str]],
    ttl: int = 300,
) -> str:
    """SWR 缓存：返回缓存 key，过期后台刷新。

    - 缓存未过期：直接返回
    - 缓存已过期：返回旧值，后台刷新
    - 无缓存：同步刷新
    """
    now = time.time()
    cached = _api_key_cache.get(source)
    if cached:
        key, expires = cached
        if now < expires:
            return key  # 新鲜，直接返回
        # 已过期 — 先返回旧值，后台刷新
        asyncio.create_task(_refresh_api_key(source, helper, ttl))
        return key
    # 无缓存 — 同步刷新
    async with _api_key_lock:
        # 拿到锁后再检查一次，避免重复刷新
        cached = _api_key_cache.get(source)
        if cached and time.time() < cached[1]:
            return cached[0]
        return await _refresh_api_key(source, helper, ttl)


async def _refresh_api_key(
    source: str,
    helper: Callable[[], Awaitable[str]],
    ttl: int,
) -> str:
    """刷新 API key 并更新缓存。"""
    try:
        key = await helper()
        _api_key_cache[source] = (key, time.time() + ttl)
        return key
    except Exception as e:
        logger.warning("Failed to refresh API key from %s: %s", source, e)
        # 刷新失败时回退到旧值（如果有的话）
        cached = _api_key_cache.get(source)
        if cached:
            return cached[0]
        raise


def clear_api_key_cache(source: str | None = None) -> None:
    """清除 API key 缓存。

    传入 source 时只清这一个来源；不传则全部清空。
    """
    if source:
        _api_key_cache.pop(source, None)
    else:
        _api_key_cache.clear()


async def handle_401_error(failed_token: str) -> bool:
    """去重的 401 处理：同一 token 的并发 401 只刷新一次。

    返回 True 表示刷新成功，可以重试请求；
    返回 False 表示刷新失败或不支持刷新。
    """
    if failed_token in _pending_401:
        return await _pending_401[failed_token]

    task = asyncio.create_task(_do_refresh_401(failed_token))
    _pending_401[failed_token] = task
    try:
        return await task
    finally:
        _pending_401.pop(failed_token, None)


async def _do_refresh_401(failed_token: str) -> bool:
    """实际的 401 刷新逻辑。

    1. 清除旧的 SWR 缓存
    2. 尝试用 refresh token 刷新（具体实现取决于认证方式）
    3. 返回是否成功
    """
    try:
        # 先清掉可能失效的 SWR 缓存，强制下次重新拉取
        clear_api_key_cache()
        # TODO: 接入实际的 refresh token 流程
        # 当前版本不支持自动刷新，由上层决定是否重新登录
        return False
    except Exception as e:
        logger.warning("401 refresh failed: %s", e)
        return False


def invalidate_oauth_cache_if_disk_changed() -> bool:
    """检测磁盘上的 token 文件是否被其他进程修改。

    基于 mtime 检测；如果文件被修改则清除缓存。
    当前为占位实现，后续接入 OAuth token 文件路径后启用。
    """
    # 暂未实现，预留接口
    return False
