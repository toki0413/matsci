"""Request size limit and timeout middleware.

Two small ASGI middlewares that protect the server from runaway
requests:

  * RequestSizeLimitMiddleware -- reject bodies larger than a threshold
  * RequestTimeoutMiddleware   -- cancel requests that take too long

Both are raw ASGI middleware (not BaseHTTPMiddleware) for the same
reason RequestIDMiddleware is: contextvars and task ownership behave
more predictably when we share the request task with the endpoint.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

# -- defaults ---------------------------------------------------------------

_DEFAULT_MAX_BODY_MB = 10
_DEFAULT_TIMEOUT_SEC = 180


# -- helpers ----------------------------------------------------------------


async def _send_json_error(
    send: Any, status_code: int, detail: str
) -> None:
    """Short-circuit the response with a JSON error body.

    Only safe to call before the inner app has started writing its own
    response -- once http.response.start has been emitted the status
    code is locked in and we'd be writing invalid ASGI messages.
    """
    body = json.dumps({"detail": detail}).encode("utf-8")
    headers: list[tuple[bytes, bytes]] = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("ascii")),
    ]
    await send(
        {
            "type": "http.response.start",
            "status": status_code,
            "headers": headers,
        }
    )
    await send(
        {
            "type": "http.response.body",
            "body": body,
        }
    )


def _header_value(scope: dict[str, Any], name: str) -> str | None:
    """Look up a header (case-insensitive, latin-1 decoded)."""
    raw_name = name.encode("latin-1")
    for key, value in scope.get("headers") or ():
        if key == raw_name:
            return value.decode("latin-1")
    return None


# -- size limit -------------------------------------------------------------


class RequestSizeLimitMiddleware:
    """Reject request bodies larger than ``max_bytes``.

    Strategy:
      1. If ``Content-Length`` is present, check it up front and 413
         immediately if it exceeds the cap. This covers the vast
         majority of real requests (browsers, curl, etc.).
      2. For chunked transfer encoding (no Content-Length), wrap
         ``receive`` so we can tally bytes as they stream in. If the
         running total crosses the cap we stop feeding body data to the
         app and send a 413 once the app yields control back.
    """

    def __init__(self, app: Any, max_bytes: int | None = None) -> None:
        self.app = app
        if max_bytes is None:
            mb = int(os.environ.get("HUGINN_MAX_BODY_SIZE_MB", _DEFAULT_MAX_BODY_MB))
            max_bytes = mb * 1024 * 1024
        self.max_bytes = max_bytes

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        # Fast path: Content-Length lets us reject without reading the body.
        cl = _header_value(scope, "content-length")
        if cl is not None:
            try:
                if int(cl) > self.max_bytes:
                    await _send_json_error(
                        send,
                        413,
                        f"Request body too large: {cl} bytes exceeds "
                        f"limit of {self.max_bytes} bytes",
                    )
                    return
            except ValueError:
                pass  # malformed Content-Length -- let the app handle it
            # Within limits -- pass straight through, no wrapping needed.
            await self.app(scope, receive, send)
            return

        # No Content-Length (chunked). Wrap receive to enforce the cap
        # while the body streams in.
        consumed = 0
        capped = False

        async def _capped_receive() -> dict[str, Any]:
            nonlocal consumed, capped
            if capped:
                # Already over the limit -- starve the app of more data.
                return {"type": "http.disconnect"}

            message = await receive()
            if message.get("type") == "http.request":
                chunk = message.get("body", b"") or b""
                consumed += len(chunk)
                if consumed > self.max_bytes:
                    capped = True
                    # Hand back an empty terminal chunk so the app
                    # doesn't deadlock waiting for more data.
                    return {
                        "type": "http.request",
                        "body": b"",
                        "more_body": False,
                    }
            return message

        response_started = False

        async def _guarded_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        await self.app(scope, _capped_receive, _guarded_send)

        # If we hit the cap and the app never produced a response,
        # send the 413 now.
        if capped and not response_started:
            await _send_json_error(
                send,
                413,
                f"Request body too large: exceeded limit of "
                f"{self.max_bytes} bytes",
            )


# -- timeout ----------------------------------------------------------------


class RequestTimeoutMiddleware:
    """Cancel requests that take longer than ``timeout_sec``.

    Wraps the inner app call in ``asyncio.wait_for``. If the timeout
    fires before the app has started its response we send a 504; if the
    response has already started we just let the connection drop (we
    can't change the status code mid-stream).
    """

    def __init__(self, app: Any, timeout_sec: float | None = None) -> None:
        self.app = app
        if timeout_sec is None:
            timeout_sec = float(
                os.environ.get("HUGINN_REQUEST_TIMEOUT_SEC", _DEFAULT_TIMEOUT_SEC)
            )
        self.timeout_sec = timeout_sec

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return

        response_started = False

        async def _tracking_send(message: dict[str, Any]) -> None:
            nonlocal response_started
            if message.get("type") == "http.response.start":
                response_started = True
            await send(message)

        try:
            await asyncio.wait_for(
                self.app(scope, receive, _tracking_send),
                timeout=self.timeout_sec,
            )
        except asyncio.TimeoutError:
            if not response_started:
                await _send_json_error(
                    send,
                    504,
                    f"Request timed out after {self.timeout_sec}s",
                )
            # If the response already started we can't inject a 504.
            # The cancelled task will cause the connection to close,
            # which is the best we can do at this point.
