"""Auth endpoints: login, token exchange, whoami, refresh.

These routes let clients obtain and refresh JWT tokens so RBAC is
actually usable. Token issuance/validation leans on the helpers in
huginn.security.auth and the raw jwt_encode/jwt_decode from rbac.

All four endpoints are listed in _PUBLIC_PATHS (see security/auth.py)
because they handle their own credential checks -- the global
require_api_key dependency would otherwise reject unauthenticated
login attempts before they reach the handler.
"""

from __future__ import annotations

import os
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel

from huginn.security.auth import _jwt_secret, secrets_match
from huginn.security.rbac import (
    Role,
    _ROLE_CAPABILITIES,
    jwt_decode,
    jwt_encode,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# Default token lifetime -- 1 hour. Matches the convention used by the
# session manager and most OAuth providers.
_TOKEN_TTL = 3600

_BEARER_PREFIX = "Bearer "


# -- request bodies --------------------------------------------------------


class LoginRequest(BaseModel):
    """Body for POST /auth/login.

    Two flavours are accepted:
      * api_key  -- shared secret (HUGINN_API_KEY), grants operator role
      * username + password -- reserved for future per-user auth
    """

    username: str | None = None
    password: str | None = None
    api_key: str | None = None


class TokenRequest(BaseModel):
    """Body for POST /auth/token (OAuth-style token exchange)."""

    grant_type: str
    api_key: str | None = None


# -- helpers ---------------------------------------------------------------


def _configured_api_key() -> str | None:
    return os.environ.get("HUGINN_API_KEY") or None


def _issue_token(username: str, role: Role, expires_in: int = _TOKEN_TTL) -> str:
    """Build and sign a JWT for the given identity."""
    secret = _jwt_secret()
    if secret is None:
        raise RuntimeError(
            "No JWT secret configured -- set HUGINN_JWT_SECRET or HUGINN_API_KEY"
        )
    payload = {
        "sub": username,
        "username": username,
        "role": role.value,
        "jti": uuid4().hex,
    }
    return jwt_encode(payload, secret, expires_in=expires_in)


def _extract_bearer(request: Request) -> str:
    """Pull the raw JWT out of the Authorization header.

    Raises 401 if the header is missing or malformed.
    """
    auth = request.headers.get("authorization", "")
    if isinstance(auth, bytes):
        auth = auth.decode("utf-8", errors="replace")
    if not auth.startswith(_BEARER_PREFIX):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or malformed Authorization header (expected Bearer token)",
        )
    return auth[len(_BEARER_PREFIX):]


def _decode_request_token(request: Request) -> dict[str, Any]:
    """Extract + decode the bearer token from the request.

    Raises 401/503 on failure so handlers stay readable.
    """
    token = _extract_bearer(request)
    secret = _jwt_secret()
    if secret is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No JWT secret configured on the server",
        )
    try:
        return jwt_decode(token, secret)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid or expired token: {exc}",
        ) from exc


def _validate_api_key(provided: str) -> None:
    """Check the shared API key, raising 401/503 on failure."""
    configured = _configured_api_key()
    if configured is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No API key configured on the server",
        )
    if not secrets_match(provided, configured):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )


# -- endpoints -------------------------------------------------------------


@router.post("/login")
async def login(body: LoginRequest) -> dict[str, Any]:
    """Exchange an API key (or username/password) for a JWT.

    The simplest flow: send {"api_key": "<HUGINN_API_KEY>"} and get back
    an operator-scoped JWT. Username/password auth is stubbed out for now
    -- the shape is accepted so clients can probe support.
    """
    if body.api_key:
        _validate_api_key(body.api_key)
        username = body.username or "operator"
        token = _issue_token(username, Role.OPERATOR)
        return {
            "token": token,
            "expires_in": _TOKEN_TTL,
            "role": Role.OPERATOR.value,
        }

    # Username/password isn't wired up yet. Return 501 so callers know
    # to fall back to the api_key flow rather than guessing.
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Username/password login is not configured; use api_key instead",
    )


@router.post("/token")
async def exchange_token(body: TokenRequest) -> dict[str, Any]:
    """OAuth-style token endpoint.

    Currently only grant_type=api_key is supported. The response shape
    mirrors a standard OAuth token response so generic OAuth clients
    can consume it.
    """
    if body.grant_type != "api_key":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported grant_type: {body.grant_type!r}",
        )
    if not body.api_key:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="api_key is required for grant_type=api_key",
        )
    _validate_api_key(body.api_key)
    token = _issue_token("operator", Role.OPERATOR)
    return {
        "access_token": token,
        "token_type": "Bearer",
        "expires_in": _TOKEN_TTL,
        "role": Role.OPERATOR.value,
    }


@router.get("/me")
async def whoami(request: Request) -> dict[str, Any]:
    """Return the identity and capabilities encoded in the caller's JWT."""
    claims = _decode_request_token(request)

    role_str = claims.get("role", "viewer")
    try:
        role = Role(role_str)
    except ValueError:
        role = Role.VIEWER

    caps = sorted(_ROLE_CAPABILITIES.get(role, set()))
    username = claims.get("username") or claims.get("sub") or ""

    return {
        "username": username,
        "role": role_str,
        "capabilities": caps,
    }


@router.post("/refresh")
async def refresh_token(request: Request) -> dict[str, Any]:
    """Issue a new JWT from a still-valid one.

    The caller passes their current (non-expired) token in the
    Authorization header and receives a fresh token with a reset
    expiry window. The old token's ``jti`` is revoked so it can no
    longer be used even before its original ``exp`` passes.
    """
    from huginn.security.rbac import TokenRevocationList

    claims = _decode_request_token(request)

    # Revoke the old token immediately — sliding window refresh must
    # invalidate the credential it replaced, otherwise a leaked token
    # stays usable until its original expiry.
    old_jti = claims.get("jti")
    old_exp = claims.get("exp")
    if old_jti:
        TokenRevocationList.shared().revoke(old_jti, exp=old_exp)

    role_str = claims.get("role", "operator")
    try:
        role = Role(role_str)
    except ValueError:
        role = Role.OPERATOR

    username = claims.get("username") or claims.get("sub") or "operator"
    new_token = _issue_token(username, role)

    return {
        "token": new_token,
        "expires_in": _TOKEN_TTL,
        "role": role_str,
    }


@router.post("/logout")
async def logout(request: Request) -> dict[str, Any]:
    """Revoke the current JWT.

    Adds the token's ``jti`` to the revocation list so subsequent
    requests carrying it are rejected with 401. The revocation entry
    auto-expires when the token's original ``exp`` passes, so the
    list doesn't grow without bound.
    """
    from huginn.security.rbac import TokenRevocationList

    claims = _decode_request_token(request)
    jti = claims.get("jti")
    exp = claims.get("exp")

    if jti:
        TokenRevocationList.shared().revoke(jti, exp=exp)
        return {"revoked": True, "jti": jti}

    # Token without jti — can't revoke individually
    return {"revoked": False, "reason": "Token has no jti claim"}
