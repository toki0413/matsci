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

import sys
import types
from typing import Any

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from huginn import __version__
from huginn.lifespan import _get_cors_origins, lifespan
from huginn.routes import ALL_ROUTERS
from huginn.routes.agents import (
    create_persona,
    get_persona,
    list_personas,
    telemetry_spans,
    telemetry_summary,
)
from huginn.routes.memory import memory_maintenance
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

app = FastAPI(
    title="Huginn Server",
    version=__version__,
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)

_cors_origins = _get_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

for _router in ALL_ROUTERS:
    app.include_router(_router)


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
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
