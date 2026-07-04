"""Maintenance mode middleware.

When ``HUGINN_MAINTENANCE=1`` is set (or toggled at runtime via
``/admin/maintenance``), all non-health requests return 503 with a
``Retry-After`` header so load balancers can drain traffic.

Health endpoints (``/health/*``, ``/ready``, ``/admin/*``) remain
accessible so the orchestrator can still probe liveness.
"""

from __future__ import annotations

import logging
import os

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Paths exempt from maintenance mode
_EXEMPT_PREFIXES = (
    "/health",
    "/ready",
    "/admin/maintenance",
    "/docs",
    "/openapi",
    "/redoc",
)

# Runtime toggle (in addition to env var)
_maintenance_active: bool = False


def is_maintenance_mode() -> bool:
    """Check if maintenance mode is active."""
    return (
        _maintenance_active
        or os.environ.get("HUGINN_MAINTENANCE", "").lower() in ("1", "true", "yes")
    )


def set_maintenance_mode(enabled: bool) -> None:
    """Toggle maintenance mode at runtime."""
    global _maintenance_active
    _maintenance_active = enabled
    logger.info("Maintenance mode %s", "enabled" if enabled else "disabled")


class MaintenanceMiddleware(BaseHTTPMiddleware):
    """Returns 503 for all non-health requests when maintenance is active."""

    async def dispatch(self, request: Request, call_next):
        if not is_maintenance_mode():
            return await call_next(request)

        path = request.url.path
        if any(path.startswith(prefix) for prefix in _EXEMPT_PREFIXES):
            return await call_next(request)

        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=503,
            content={
                "error_code": "MAINTENANCE_MODE",
                "message": "Server is under maintenance. Please retry later.",
                "request_id": request_id,
            },
            headers={"Retry-After": "300"},
        )
