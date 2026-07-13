"""Role-Based Access Control (RBAC) for Huginn.

Provides a user model with roles (viewer, operator, admin), JWT token
generation/validation, and per-user session isolation.  Works alongside the
existing API-key auth so deployments can migrate gradually.
"""

from __future__ import annotations

import enum
import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Role model
# ---------------------------------------------------------------------------

class Role(str, enum.Enum):
    """Built-in user roles with escalating privileges."""

    VIEWER = "viewer"
    OPERATOR = "operator"
    ADMIN = "admin"

    @property
    def level(self) -> int:
        return _ROLE_LEVELS[self]


_ROLE_LEVELS: dict[Role, int] = {
    Role.VIEWER: 1,
    Role.OPERATOR: 2,
    Role.ADMIN: 3,
}

# What each role can do --------------------------------------------------
# Keys are capability strings checked by ``User.can()``.
_ROLE_CAPABILITIES: dict[Role, set[str]] = {
    Role.VIEWER: {
        "read",
        "query",
        "health",
    },
    Role.OPERATOR: {
        "read",
        "query",
        "health",
        "execute",
        "write",
        "upload",
    },
    Role.ADMIN: {
        "read",
        "query",
        "health",
        "execute",
        "write",
        "upload",
        "admin",
        "config",
        "user_manage",
    },
}


# ---------------------------------------------------------------------------
# User model
# ---------------------------------------------------------------------------

@dataclass
class User:
    """Represents an authenticated user."""

    user_id: str
    username: str
    role: Role = Role.VIEWER
    api_key_hash: str = ""
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    active: bool = True

    # -- helpers --------------------------------------------------------

    def can(self, capability: str) -> bool:
        """Return True if the user's role grants *capability*."""
        caps = _ROLE_CAPABILITIES.get(self.role, set())
        return capability in caps

    def has_role_or_higher(self, required: Role) -> bool:
        return self.role.level >= required.level

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["role"] = self.role.value
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> User:
        data = dict(data)
        data["role"] = Role(data.get("role", "viewer"))
        return cls(**data)


# ---------------------------------------------------------------------------
# JWT helpers  (HS256 — no external dependency)
# ---------------------------------------------------------------------------

import base64
import struct


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    padding = 4 - len(s) % 4
    if padding != 4:
        s += "=" * padding
    return base64.urlsafe_b64decode(s)


def _hmac_sha256(key: bytes, msg: bytes) -> bytes:
    return hmac.new(key, msg, hashlib.sha256).digest()


def jwt_encode(
    payload: dict[str, Any],
    secret: str | bytes,
    *,
    expires_in: int = 3600,
) -> str:
    """Create an HS256 JWT.

    Parameters
    ----------
    payload : dict
        Claims.  ``iat`` and ``exp`` are added automatically.
    secret : str | bytes
        HMAC signing key.
    expires_in : int
        Token lifetime in seconds (default 1 h).
    """
    if isinstance(secret, str):
        secret = secret.encode("utf-8")

    header = {"alg": "HS256", "typ": "JWT"}
    now = int(time.time())
    payload = {**payload, "iat": now, "exp": now + expires_in}

    h = _b64url_encode(json.dumps(header, separators=(",", ":")).encode())
    p = _b64url_encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{h}.{p}".encode("ascii")
    sig = _hmac_sha256(secret, signing_input)
    return f"{h}.{p}.{_b64url_encode(sig)}"


def jwt_decode(token: str, secret: str | bytes) -> dict[str, Any]:
    """Decode and verify an HS256 JWT.

    Raises ``ValueError`` on any validation failure.
    """
    if isinstance(secret, str):
        secret = secret.encode("utf-8")

    parts = token.split(".")
    if len(parts) != 3:
        raise ValueError("Malformed JWT: expected 3 parts")

    h_b64, p_b64, s_b64 = parts
    signing_input = f"{h_b64}.{p_b64}".encode("ascii")
    expected_sig = _hmac_sha256(secret, signing_input)
    actual_sig = _b64url_decode(s_b64)

    if not hmac.compare_digest(expected_sig, actual_sig):
        raise ValueError("JWT signature mismatch")

    payload: dict[str, Any] = json.loads(_b64url_decode(p_b64))

    exp = payload.get("exp")
    if exp is None:
        raise ValueError("JWT missing exp claim")
    if int(time.time()) > int(exp):
        raise ValueError("JWT has expired")

    return payload


def generate_api_key() -> str:
    """Generate a URL-safe random API key (48 bytes → 64 chars)."""
    return secrets.token_urlsafe(48)


def hash_api_key(key: str) -> str:
    """One-way SHA-256 hash for storing API keys safely."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Session isolation
# ---------------------------------------------------------------------------

@dataclass
class UserSession:
    """Lightweight in-memory session bound to a user."""

    session_id: str
    user_id: str
    created_at: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)

    def touch(self) -> None:
        self.last_active = time.time()

    def is_expired(self, max_idle: float = 7200.0) -> bool:
        return (time.time() - self.last_active) > max_idle


class SessionManager:
    """Thread-safe in-memory session store with idle-expiry."""

    def __init__(self, max_idle: float = 7200.0) -> None:
        self.max_idle = max_idle
        self._sessions: dict[str, UserSession] = {}
        self._user_sessions: dict[str, list[str]] = {}  # user_id → [session_ids]
        self._lock = threading.RLock()

    def create(self, user_id: str) -> UserSession:
        with self._lock:
            sid = secrets.token_urlsafe(32)
            session = UserSession(session_id=sid, user_id=user_id)
            self._sessions[sid] = session
            self._user_sessions.setdefault(user_id, []).append(sid)
            return session

    def get(self, session_id: str) -> UserSession | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            if session.is_expired(self.max_idle):
                self.destroy(session_id)
                return None
            session.touch()
            return session

    def destroy(self, session_id: str) -> None:
        with self._lock:
            session = self._sessions.pop(session_id, None)
            if session:
                sids = self._user_sessions.get(session.user_id, [])
                if session_id in sids:
                    sids.remove(session_id)

    def destroy_user_sessions(self, user_id: str) -> int:
        with self._lock:
            sids = self._user_sessions.pop(user_id, [])
            for sid in sids:
                self._sessions.pop(sid, None)
            return len(sids)

    def cleanup_expired(self) -> int:
        with self._lock:
            expired = [
                sid for sid, s in self._sessions.items() if s.is_expired(self.max_idle)
            ]
            for sid in expired:
                self.destroy(sid)
            return len(expired)

    def active_count(self) -> int:
        with self._lock:
            return len(self._sessions)


# ---------------------------------------------------------------------------
# Token revocation (logout / deny-list)
# ---------------------------------------------------------------------------

class TokenRevocationList:
    """Thread-safe in-memory JWT revocation list.

    When a user logs out, their token's ``jti`` (JWT ID) is added here.
    Subsequent requests carrying that token are rejected even before
    the natural ``exp`` expiry.

    Entries are pruned automatically once their original ``exp`` has
    passed, so the list does not grow without bound.

    Usage::

        from huginn.security.rbac import TokenRevocationList

        revoke_list = TokenRevocationList.shared()
        revoke_list.revoke(jti="abc123", exp=1700000000)
        revoke_list.is_revoked("abc123")  # True
    """

    _singleton: TokenRevocationList | None = None
    _singleton_lock = threading.Lock()

    def __init__(self) -> None:
        self._revoked: dict[str, float] = {}  # jti -> exp
        self._lock = threading.RLock()

    @classmethod
    def shared(cls) -> TokenRevocationList:
        """Return the process-wide singleton."""
        if cls._singleton is None:
            with cls._singleton_lock:
                if cls._singleton is None:
                    cls._singleton = cls()
        return cls._singleton

    def revoke(self, jti: str, exp: float | None = None) -> None:
        """Mark *jti* as revoked.

        If *exp* is given (the token's original expiry timestamp), the
        entry will be auto-pruned after that time. If not, a default
        TTL of 24 hours is used.
        """
        if exp is None:
            exp = time.time() + 86400  # 24h fallback
        with self._lock:
            self._revoked[jti] = exp
            self._prune_locked()

    def is_revoked(self, jti: str) -> bool:
        """Check whether *jti* has been revoked."""
        with self._lock:
            self._prune_locked()
            return jti in self._revoked

    def _prune_locked(self) -> int:
        """Remove entries whose original exp has passed. Caller must hold lock."""
        now = time.time()
        expired = [jti for jti, exp in self._revoked.items() if now > exp]
        for jti in expired:
            del self._revoked[jti]
        return len(expired)

    def clear(self) -> None:
        """Remove all entries (mainly for testing)."""
        with self._lock:
            self._revoked.clear()

    def count(self) -> int:
        with self._lock:
            self._prune_locked()
            return len(self._revoked)

    def revoke_user(self, user_id: str, session_manager: SessionManager | None = None) -> int:
        """Revoke all server-side sessions for *user_id*.

        Since JWTs are stateless, we can't enumerate all outstanding tokens.
        This destroys all server-side sessions as a best-effort measure.
        Returns the number of sessions destroyed.
        """
        if session_manager is not None:
            return session_manager.destroy_user_sessions(user_id)
        return 0
