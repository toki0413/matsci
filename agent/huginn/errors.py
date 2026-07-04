"""Unified error response envelope and error codes.

All API error responses follow the same JSON envelope:

    {
        "error_code": "RESOURCE_NOT_FOUND",
        "message": "Tool 'bash' not found",
        "request_id": "req-abc123",
        "details": {}  # optional
    }

This replaces the previous ad-hoc formats ({"error": "..."},
{"detail": "..."}, {"success": false, "error": "..."}).
"""

from __future__ import annotations

import enum
from typing import Any


class ErrorCode(str, enum.Enum):
    """Stable, machine-readable error codes for API responses."""

    # ── Generic ────────────────────────────────────────────────────
    INTERNAL_ERROR = "INTERNAL_ERROR"
    HTTP_ERROR = "HTTP_ERROR"
    NOT_FOUND = "NOT_FOUND"
    VALIDATION_ERROR = "VALIDATION_ERROR"
    METHOD_NOT_ALLOWED = "METHOD_NOT_ALLOWED"

    # ── Authentication / Authorization ─────────────────────────────
    UNAUTHORIZED = "UNAUTHORIZED"
    FORBIDDEN = "FORBIDDEN"
    TOKEN_REVOKED = "TOKEN_REVOKED"
    TOKEN_EXPIRED = "TOKEN_EXPIRED"

    # ── Rate limiting ──────────────────────────────────────────────
    RATE_LIMITED = "RATE_LIMITED"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"

    # ── Tools ──────────────────────────────────────────────────────
    TOOL_NOT_FOUND = "TOOL_NOT_FOUND"
    TOOL_ERROR = "TOOL_ERROR"
    TOOL_DENIED = "TOOL_DENIED"
    TOOL_TIMEOUT = "TOOL_TIMEOUT"
    CIRCUIT_OPEN = "CIRCUIT_OPEN"
    LOOP_DETECTED = "LOOP_DETECTED"

    # ── Resources ──────────────────────────────────────────────────
    RESOURCE_NOT_FOUND = "RESOURCE_NOT_FOUND"
    RESOURCE_CONFLICT = "RESOURCE_CONFLICT"
    PAYLOAD_TOO_LARGE = "PAYLOAD_TOO_LARGE"

    # ── LLM ───────────────────────────────────────────────────────
    LLM_OVERLOADED = "LLM_OVERLOADED"
    LLM_UNAVAILABLE = "LLM_UNAVAILABLE"
    CONTEXT_OVERFLOW = "CONTEXT_OVERFLOW"

    # ── Maintenance ────────────────────────────────────────────────
    MAINTENANCE_MODE = "MAINTENANCE_MODE"


def huginn_error_response(
    code: str | ErrorCode,
    message: str,
    request_id: str = "unknown",
    *,
    details: dict[str, Any] | None = None,
    status_code: int = 500,
) -> dict[str, Any]:
    """Build a unified error response body.

    Parameters
    ----------
    code
        Stable error code (e.g. ``"TOOL_NOT_FOUND"``).
    message
        Human-readable message safe to return to the client.
    request_id
        Correlation ID from the request, so clients can trace errors.
    details
        Optional additional context (never include secrets).
    status_code
        HTTP status code (for reference in the body).
    """
    if isinstance(code, ErrorCode):
        code = code.value
    body: dict[str, Any] = {
        "error_code": code,
        "message": message,
        "request_id": request_id,
    }
    if details:
        body["details"] = details
    return body


class HuginnError(Exception):
    """Base exception for all Huginn domain errors.

    Subclasses should set ``code`` and ``status_code`` class attributes.
    The global exception handler converts these to the unified JSON
    envelope automatically.
    """

    code: ErrorCode = ErrorCode.INTERNAL_ERROR
    status_code: int = 500

    def __init__(self, message: str, *, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = details

    def to_response(self, request_id: str = "unknown") -> dict[str, Any]:
        return huginn_error_response(
            self.code,
            self.message,
            request_id,
            details=self.details,
            status_code=self.status_code,
        )


class ToolNotFoundError(HuginnError):
    code = ErrorCode.TOOL_NOT_FOUND
    status_code = 404


class ToolDeniedError(HuginnError):
    code = ErrorCode.TOOL_DENIED
    status_code = 403


class RateLimitedError(HuginnError):
    code = ErrorCode.RATE_LIMITED
    status_code = 429
