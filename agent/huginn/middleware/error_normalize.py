"""Response normalizer — converts old-format error responses to the unified envelope.

Routes that return {"error": "..."} or {"success": false, "error": "..."}
get transparently rewritten to:

    {
        "error_code": "LEGACY_ERROR",
        "message": "...",
        "request_id": "req-xxx"
    }

This is a stop-gap until all routes are migrated to raise HTTPException
or return huginn_error_response() directly. The middleware is opt-in
via env var HUGINN_NORMALIZE_ERRORS=1 (default: on in non-dev mode).
"""

from __future__ import annotations

import logging
import os
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Only normalize JSON responses with old-format error keys
_LEGACY_ERROR_KEYS = {"error"}
_LEGACY_SUCCESS_ERROR_KEYS = {"success", "error"}


def _is_legacy_error(body: dict[str, Any]) -> bool:
    """Detect old-format error responses."""
    if "error_code" in body:
        return False  # already unified
    if "error" in body and isinstance(body["error"], str):
        return True
    if (
        body.get("success") is False
        and "error" in body
        and isinstance(body["error"], str)
    ):
        return True
    return False


def _normalize_error_body(
    body: dict[str, Any], request_id: str, status_code: int
) -> dict[str, Any]:
    """Convert old-format error to unified envelope."""
    message = body.get("error", "Unknown error")
    # Try to infer a better error code from status
    if status_code == 404:
        code = "RESOURCE_NOT_FOUND"
    elif status_code == 403:
        code = "FORBIDDEN"
    elif status_code == 401:
        code = "UNAUTHORIZED"
    elif status_code == 429:
        code = "RATE_LIMITED"
    elif status_code >= 500:
        code = "INTERNAL_ERROR"
    elif status_code >= 400:
        code = "HTTP_ERROR"
    else:
        code = "HTTP_ERROR"

    normalized: dict[str, Any] = {
        "error_code": code,
        "message": message,
        "request_id": request_id,
    }
    # Preserve extra fields (e.g. "details") but drop the old keys
    for k, v in body.items():
        if k not in ("error", "success", "error_code", "message", "request_id"):
            normalized.setdefault("details", {})[k] = v
    return normalized


class ErrorNormalizeMiddleware(BaseHTTPMiddleware):
    """Rewrite legacy error JSON responses to the unified envelope."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)

        # Skip non-JSON
        content_type = response.headers.get("content-type", "")
        if "application/json" not in content_type:
            return response

        # Only process error status codes (4xx/5xx) — 200 with success=false
        # is a softer case; we handle it too since some routes do this.
        status_code = response.status_code
        if status_code < 400 and status_code != 200:
            return response

        try:
            body_bytes = b"".join([chunk async for chunk in response.body_iterator])
            import json

            body = json.loads(body_bytes)
        except Exception:
            return response  # not JSON or parse error, leave as-is

        if not isinstance(body, dict) or not _is_legacy_error(body):
            return response

        # For 200 responses, only normalize if success=false
        if status_code == 200 and body.get("success") is not False:
            return response

        request_id = getattr(request.state, "request_id", "unknown")
        normalized = _normalize_error_body(body, request_id, status_code)

        # If original was 200 with success=false, bump to 400
        if status_code == 200:
            status_code = 400

        return JSONResponse(
            status_code=status_code,
            content=normalized,
            headers={
                k: v
                for k, v in response.headers.items()
                if k.lower() not in ("content-length", "content-type")
            },
        )


def should_enable_normalize() -> bool:
    """Check env var. Default: enabled unless dev mode."""
    val = os.environ.get("HUGINN_NORMALIZE_ERRORS", "")
    if val.lower() in ("0", "false", "no"):
        return False
    if val.lower() in ("1", "true", "yes"):
        return True
    # Default: enabled unless dev mode
    return os.environ.get("HUGINN_DEV_MODE", "") != "1"
