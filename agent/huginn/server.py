"""FastAPI + WebSocket server for Huginn.

Serves the desktop frontend with:
- HTTP API for tools, workflows, and health checks
- WebSocket endpoint for real-time Agent chat
- Compatibility stubs for math-anything frontend APIs

This module is a thin entry point.  All shared state lives in
``huginn.server_core``, lifecycle management in ``huginn.lifespan``,
and route handlers in ``huginn.routes.*``.

Backward-compatible re-exports are provided via a ``sys.modules`` wrapper
so that ``server_module._context = ctx`` in tests correctly propagates
to ``server_core._context``.
"""

from __future__ import annotations

import os
import sys
import time
import types
from collections import defaultdict, deque
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from huginn import __version__
from huginn.lifespan import _get_cors_origins, lifespan
from huginn.middleware.limits import RequestSizeLimitMiddleware, RequestTimeoutMiddleware
from huginn.middleware.request_id import RequestIDMiddleware
from huginn.routes import ALL_ROUTERS, include_v1_routes
from huginn.routes.agents import (
    create_persona,
    get_persona,
    list_personas,
    telemetry_spans,
    telemetry_summary,
)
from huginn.routes.memory import memory_maintenance
from huginn.routes.metrics import (
    RATE_LIMIT_BLOCKED_TOTAL,
    http_metrics_dispatch,
)
from huginn.routes.threads import get_thread
from huginn.routes.unified import unified_plot_endpoint, unified_solve_endpoint
from huginn.security.auth import require_api_key
from huginn.tools import register_all_tools

# ── Import server_core so its symbols are available for re-export ──────
import huginn.server_core as _sc
import huginn.lifespan as _lf

# Register all tools at import time so ToolRegistry is populated
# before any route handler runs.
register_all_tools()

# Hide interactive docs in production to avoid leaking API surface.
_hide_docs = (
    os.environ.get("HUGINN_ENV", "").lower() == "production"
    or os.environ.get("HUGINN_HIDE_DOCS", "").lower() in ("1", "true", "yes")
)

app = FastAPI(
    title="Huginn Server",
    version=__version__,
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
    docs_url=None if _hide_docs else "/docs",
    redoc_url=None if _hide_docs else "/redoc",
    openapi_url=None if _hide_docs else "/openapi.json",
)

_cors_origins = _get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-HUGINN-API-KEY", "X-Request-ID"],
)


# ── Rate limiting (sliding window per client IP) ──────────────────────
# Default to 120 req/min when not explicitly configured. Set to 0 to disable.
_RATE_LIMIT = int(os.environ.get("HUGINN_RATE_LIMIT_PER_MINUTE", "120"))
_rate_buckets: dict[str, deque] = defaultdict(deque)
_RATE_WINDOW = 60.0  # seconds
# Sweep empty buckets every N requests so _rate_buckets doesn't grow
# without bound as we see more and more distinct client IPs.
_BUCKET_SWEEP_INTERVAL = 1000
_request_counter = 0


def _sweep_empty_buckets() -> None:
    """Drop buckets that have drained to keep _rate_buckets bounded."""
    for ip in [ip for ip, bucket in _rate_buckets.items() if not bucket]:
        del _rate_buckets[ip]


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    """Enforce per-IP rate limiting when HUGINN_RATE_LIMIT_PER_MINUTE > 0."""
    if _RATE_LIMIT <= 0:
        return await call_next(request)

    # Skip health checks, docs, and the Prometheus scrape endpoint.
    path = request.url.path
    if path in ("/health", "/docs", "/openapi.json", "/redoc", "/metrics", "/diagnostics"):
        return await call_next(request)

    client_ip = request.client.host if request.client else "unknown"
    now = time.monotonic()
    bucket = _rate_buckets[client_ip]

    # Drop timestamps older than the window
    while bucket and bucket[0] < now - _RATE_WINDOW:
        bucket.popleft()

    # If this client's bucket has drained, drop the dict entry so we don't
    # keep empty deques around for every IP we've ever seen. defaultdict
    # will hand us a fresh one on the next request from this IP.
    if not bucket:
        del _rate_buckets[client_ip]

    if len(bucket) >= _RATE_LIMIT:
        RATE_LIMIT_BLOCKED_TOTAL.labels(session="all").inc()
        from huginn.errors import huginn_error_response
        request_id = getattr(request.state, "request_id", "unknown")
        return JSONResponse(
            status_code=429,
            content=huginn_error_response(
                "RATE_LIMITED",
                "Rate limit exceeded",
                request_id,
                status_code=429,
            ),
            headers={"Retry-After": str(int(_RATE_WINDOW))},
        )

    # defaultdict re-creates the bucket if we deleted it above.
    _rate_buckets[client_ip].append(now)

    # Every so often, reclaim buckets for clients that have gone quiet.
    global _request_counter
    _request_counter += 1
    if _request_counter >= _BUCKET_SWEEP_INTERVAL:
        _request_counter = 0
        _sweep_empty_buckets()

    return await call_next(request)


# Metrics middleware — registered after the rate limiter so it sits outside
# it (and outside CORS), counting every request — including 429s — and
# timing the full handler stack. It still wraps CORS, as required.
app.add_middleware(BaseHTTPMiddleware, dispatch=http_metrics_dispatch)

# Body size limit — reject requests whose payload exceeds
# HUGINN_MAX_BODY_SIZE_MB (default 10 MB) before they reach the app.
app.add_middleware(RequestSizeLimitMiddleware)

# Request timeout — cancel requests that run longer than
# HUGINN_REQUEST_TIMEOUT_SEC (default 180 s).
app.add_middleware(RequestTimeoutMiddleware)

# Request-ID middleware — registered last so it runs outermost and the
# correlation id is in place before any other middleware / handler logs.
app.add_middleware(RequestIDMiddleware)

# Maintenance mode middleware — returns 503 for non-health requests
# when HUGINN_MAINTENANCE=1 or runtime toggle is on.
from huginn.middleware.maintenance import MaintenanceMiddleware  # noqa: E402

app.add_middleware(MaintenanceMiddleware)


# ── Global exception handler ──────────────────────────────────────────
# Catches unhandled exceptions and returns a proper 500 response instead
# of letting FastAPI return a 500 with a generic message.
@app.exception_handler(Exception)
async def _global_exception_handler(request: Request, exc: Exception):
    import logging

    request_id = getattr(request.state, "request_id", "unknown")
    logging.getLogger("huginn.server").error(
        "Unhandled exception in %s %s [request_id=%s]: %s",
        request.method, request.url.path, request_id, exc,
        exc_info=True,
    )
    # Don't leak internal details to the client, but include request_id
    # so the client can correlate with server logs.
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": "Internal server error",
            "request_id": request_id,
        },
    )


# ── Unified HTTPException handler ───────────────────────────────────
# Ensures all HTTPExceptions (including those raised by FastAPI's own
# validation) return the same JSON envelope.
@app.exception_handler(HTTPException)
async def _http_exception_handler(request: Request, exc: HTTPException):
    from huginn.errors import huginn_error_response
    import json

    request_id = getattr(request.state, "request_id", "unknown")
    body = huginn_error_response(
        code=getattr(exc, "huginn_code", "HTTP_ERROR"),
        message=str(exc.detail) if exc.detail else "Error",
        request_id=request_id,
        status_code=exc.status_code,
    )
    return JSONResponse(status_code=exc.status_code, content=body)


# Mount the full route surface. include_v1_routes mounts every router
# under the /v1 version prefix (the canonical API going forward) and, for
# backward compatibility, also at the root path — with a deprecation
# warning + header nudging callers toward /v1.
include_v1_routes(app, keep_root_compat=True)


# ── sys.modules wrapper for backward-compatible attribute access ───────
#
# Tests do ``import huginn.server as m; m._context = ctx`` which must
# propagate to server_core._context (where get_context() reads from).
# A plain ``from server_core import _context`` creates a local copy that
# doesn't reflect later mutations.  The wrapper delegates shared-state
# reads/writes to the authoritative modules.

_DELEGATED_SC = {"_context", "_checkpoints", "_threads"}
_DELEGATED_LF = {"_init_mcp_tools", "_shutdown_mcp"}
_THIS = sys.modules[__name__]


class _ServerModule(types.ModuleType):
    """Module wrapper delegating shared-state attrs to server_core / lifespan."""

    def __getattr__(self, name: str) -> Any:
        if name in _DELEGATED_SC:
            return getattr(_sc, name)
        if name in _DELEGATED_LF:
            return getattr(_lf, name)
        raise AttributeError(f"module 'huginn.server' has no attribute {name!r}")

    def __setattr__(self, name: str, value: Any) -> None:
        if name in _DELEGATED_SC:
            setattr(_sc, name, value)
            return
        if name in _DELEGATED_LF:
            setattr(_lf, name, value)
            return
        super().__setattr__(name, value)


_wrapper = _ServerModule(__name__)
_wrapper.__dict__.update(
    {k: v for k, v in _THIS.__dict__.items() if not k.startswith("__")}
)
_wrapper.__file__ = __file__
_wrapper.__package__ = __package__
_wrapper.__path__ = getattr(_THIS, "__path__", [])
_wrapper.__spec__ = _THIS.__spec__
sys.modules[__name__] = _wrapper


if __name__ == "__main__":
    import sys
    import uvicorn

    # Accept --port from sidecar; default to 8000
    port = 8000
    for i, arg in enumerate(sys.argv):
        if arg == "--port" and i + 1 < len(sys.argv):
            try:
                port = int(sys.argv[i + 1])
            except ValueError:
                pass

    uvicorn.run(app, host="127.0.0.1", port=port)
