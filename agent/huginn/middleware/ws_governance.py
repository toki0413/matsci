"""WebSocket governance: connection limits and per-connection message rate limiting.

Keeps a single user (or IP) from opening too many WS connections or flooding
a connection with messages. Used by all /ws/* endpoints via ws_auth_and_track().
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from fastapi import WebSocket, status


class WSConnectionTracker:
    """Track active WebSocket connections per identity for resource governance."""

    def __init__(self, max_per_user: int = 10, max_msgs_per_sec: int = 5):
        self._connections: dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        self._max_per_user = max_per_user
        self._max_msgs_per_sec = max_msgs_per_sec

    def acquire(self, identity: str) -> bool:
        """Try to grab a connection slot. Returns False if the limit is hit."""
        with self._lock:
            if self._connections[identity] >= self._max_per_user:
                return False
            self._connections[identity] += 1
            return True

    def release(self, identity: str) -> None:
        with self._lock:
            if self._connections[identity] > 0:
                self._connections[identity] -= 1

    def active(self, identity: str) -> int:
        with self._lock:
            return self._connections[identity]

    def check_msg_rate(self, identity: str) -> bool:
        # ponytail: per-connection rate limit (WSMessageRateLimiter below) is
        # sufficient for our use case — no need for a global per-identity bucket.
        # If we ever need cross-connection rate limiting, add a token bucket here.
        return True


# Singleton — one tracker for the whole process
_tracker: WSConnectionTracker | None = None


def get_tracker() -> WSConnectionTracker:
    global _tracker
    if _tracker is None:
        _tracker = WSConnectionTracker()
    return _tracker


def _extract_identity(websocket: WebSocket) -> str:
    """Best-effort identity extraction from an authenticated WebSocket.

    require_api_key attaches a RequestContext to request.state.auth when it can,
    but for WebSocket connections that often doesn't land (the state object on
    a WS isn't always populated the same way). Fall back to client IP so we
    still get per-IP governance.
    """
    try:
        auth_ctx = getattr(getattr(websocket, "state", None), "auth", None)
    except Exception:
        auth_ctx = None

    if auth_ctx is not None:
        user = getattr(auth_ctx, "user", None)
        if user is not None:
            uid = getattr(user, "user_id", None)
            if uid:
                return str(uid)
        # API-key mode — no user object, just hash the key prefix
        api_key = getattr(auth_ctx, "api_key", None)
        if api_key:
            return f"key:{api_key[:8]}"

    # Fall back to client IP
    client = getattr(websocket, "client", None)
    if client is not None:
        return f"ip:{client.host}"

    return "anonymous"


async def ws_auth_and_track(websocket: WebSocket) -> str | None:
    """Unified WS auth + connection tracking.

    Returns the identity string on success, or None (after closing the
    socket) on failure. The caller MUST release the slot in a finally block::

        identity = await ws_auth_and_track(websocket)
        if identity is None:
            return  # already closed
        try:
            await websocket.accept()
            # ... WS logic ...
        finally:
            get_tracker().release(identity)
    """
    from huginn.security.auth import require_api_key

    # require_api_key is sync — it raises HTTPException on bad credentials
    try:
        require_api_key(request=None, websocket=websocket)
    except Exception:
        await websocket.close(
            code=status.WS_1008_POLICY_VIOLATION, reason="auth failed"
        )
        return None

    identity = _extract_identity(websocket)

    tracker = get_tracker()
    if not tracker.acquire(identity):
        await websocket.close(
            code=status.WS_1013_TRY_AGAIN_LATER,
            reason="too many connections",
        )
        return None

    return identity


class WSMessageRateLimiter:
    """Per-connection message rate limiter using a sliding window.

    Instantiate one per connection, call check() on every inbound message.
    """

    def __init__(self, max_per_sec: int = 5, window: float = 1.0):
        self._max = max_per_sec
        self._window = window
        self._timestamps: list[float] = []

    def check(self) -> bool:
        """True if the message is allowed, False if rate-limited."""
        now = time.monotonic()
        cutoff = now - self._window
        # Drop expired timestamps
        self._timestamps = [t for t in self._timestamps if t > cutoff]
        if len(self._timestamps) >= self._max:
            return False
        self._timestamps.append(now)
        return True
