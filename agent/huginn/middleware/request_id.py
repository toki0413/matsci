"""Request-ID middleware.

Tags every HTTP request with a short correlation id (``req_<12hex>``) and
propagates it via the ``X-Request-ID`` response header plus a contextvar so
log records emitted during the request all share the same id.

Implemented as a raw ASGI middleware rather than ``BaseHTTPMiddleware``
because contextvar changes made inside ``BaseHTTPMiddleware`` are not
reliably visible to endpoint handlers — Starlette runs the inner app in a
child task that copies the context at spawn time.  A raw ASGI middleware
shares the task with the endpoint, so the set propagates correctly.
"""

from __future__ import annotations

import uuid
from typing import Any

from huginn.utils.json_logging import request_id_var

_HEADER = "x-request-id"


class RequestIDMiddleware:
    """ASGI middleware that assigns a per-request correlation id."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return

        req_id = _resolve_request_id(scope)
        # Make the id visible to loggers for the lifetime of this request.
        token = request_id_var.set(req_id)

        if scope_type == "http":
            async def _send(message: dict[str, Any]) -> None:
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append((b"x-request-id", req_id.encode("latin-1")))
                    message["headers"] = headers
                await send(message)

            try:
                await self.app(scope, receive, _send)
            finally:
                request_id_var.reset(token)
        else:
            try:
                await self.app(scope, receive, send)
            finally:
                request_id_var.reset(token)


def _resolve_request_id(scope: dict[str, Any]) -> str:
    """Honor an inbound ``X-Request-ID`` so distributed traces stay linked,
    otherwise mint a fresh ``req_<12hex>``.
    """
    for name, value in scope.get("headers") or ():
        if name == _HEADER.encode("latin-1"):
            try:
                incoming = value.decode("latin-1").strip()
            except Exception:
                incoming = ""
            if incoming:
                return incoming
    return f"req_{uuid.uuid4().hex[:12]}"
