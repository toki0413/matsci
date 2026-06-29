"""Application lifespan, MCP initialization, and CORS configuration."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from huginn.config import HuginnConfig
from huginn.pet import configure_pet
from huginn.server_core import (
    _CODEBASE_AVAILABLE,
    _KB_AVAILABLE,
    get_codebase_index,
    get_context,
    get_knowledge_base,
)

logger = logging.getLogger(__name__)


# ── MCP helpers ─────────────────────────────────────────────────────


async def _connect_mcp_server(
    manager: Any,
    name: str,
    config: Any,
    timeout: float = 10.0,
) -> bool:
    """Connect to a single MCP server with timeout and full error containment."""
    try:
        async with asyncio.timeout(timeout):
            await manager.connect(config)
        return True
    except TimeoutError:
        print(f"[MCP] Warning: {name} connection timed out ({timeout}s)")
    except Exception as e:
        print(f"[MCP] Warning: failed to connect to {name}: {e}")
    return False


async def _init_mcp_tools():
    """Connect to local MCP servers and register their tools.

    Failures are fully contained — this function never raises.
    """
    try:
        from pathlib import Path

        from huginn.mcp_client import MCPClientManager, MCPServerConfig
        from huginn.tools.mcp_adapter import register_mcp_tools

        get_context().mcp_manager = MCPClientManager()
        base = Path(__file__).parent.parent.parent  # repo root

        servers: list[tuple[str, MCPServerConfig]] = []

        mat_db_path = base / "servers" / "mat-db-mcp" / "server.py"
        if mat_db_path.exists():
            servers.append(
                ("mat-db", MCPServerConfig(name="mat-db", command="python", args=[str(mat_db_path)]))
            )

        math_path = base / "servers" / "math-anything-mcp" / "server.py"
        if math_path.exists():
            servers.append(
                ("math-anything", MCPServerConfig(name="math-anything", command="python", args=[str(math_path)]))
            )

        for name, cfg in servers:
            task = asyncio.create_task(
                _connect_mcp_server(get_context().mcp_manager, name, cfg)
            )
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            await task

        registered = register_mcp_tools(get_context().mcp_manager)
        print(f"[MCP] Registered {len(registered)} tools from MCP servers")
    except Exception as e:
        print(f"[MCP] Warning: Could not initialize MCP tools: {e}")


async def _shutdown_mcp():
    """Disconnect all MCP servers."""
    if get_context().mcp_manager:
        await get_context().mcp_manager.disconnect_all()
        get_context().mcp_manager = None


async def _mcp_health_monitor(manager: Any) -> None:
    """Background task: periodically probe MCP servers and reconnect failures.

    Uses exponential backoff (1s → 2s → 4s → … → 30s cap) and gives up
    after 5 consecutive failures per server.  Runs until cancelled.
    """
    from huginn.mcp_client import (
        _BACKOFF_BASE,
        _BACKOFF_FACTOR,
        _BACKOFF_MAX,
        _HEALTH_CHECK_INTERVAL,
    )

    while True:
        try:
            await asyncio.sleep(_HEALTH_CHECK_INTERVAL)
        except asyncio.CancelledError:
            return

        for name in list(manager._configs.keys()):
            if manager.should_stop_retrying(name):
                continue

            # Check if currently connected
            if manager.is_connected(name):
                healthy = await manager.check_health(name)
                if healthy:
                    continue
                # Unhealthy — fall through to reconnect

            # Exponential backoff based on consecutive failures
            failures = manager._consecutive_failures.get(name, 0)
            delay = min(
                _BACKOFF_BASE * (_BACKOFF_FACTOR ** failures),
                _BACKOFF_MAX,
            )
            logger.info(
                f"[MCP] Reconnecting '{name}' (attempt {failures + 1}, "
                f"backoff {delay:.1f}s)"
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return

            success = await manager.reconnect(name)
            if success:
                # Re-register tools in ToolRegistry after successful reconnect
                try:
                    from huginn.tools.mcp_adapter import register_mcp_tools
                    registered = register_mcp_tools(manager, server_name=name)
                    logger.info(
                        f"[MCP] Re-registered {len(registered)} tools from '{name}'"
                    )
                except Exception as e:
                    logger.warning(f"[MCP] Tool re-registration failed for '{name}': {e}")
            else:
                logger.warning(
                    f"[MCP] Reconnect failed for '{name}' "
                    f"(consecutive failures: {manager._consecutive_failures.get(name, 0)})"
                )


# ── lifespan ────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    await _init_mcp_tools()
    if _KB_AVAILABLE and get_context().kb is None:
        try:
            cfg = HuginnConfig.from_env()
            get_context().kb = get_knowledge_base(cfg.workspace)
        except Exception as e:
            print(f"[KB] Warning: could not initialize knowledge base: {e}")
    if _CODEBASE_AVAILABLE and get_context().codebase is None:
        try:
            cfg = HuginnConfig.from_env()
            get_context().codebase = get_codebase_index(cfg.workspace)
        except Exception as e:
            print(f"[Codebase] Warning: could not initialize codebase index: {e}")
    try:
        cfg = HuginnConfig.from_env()
        configure_pet(cfg.pet_name, cfg.pet_personality)
    except Exception as e:
        print(f"[Pet] Warning: could not configure pet: {e}")

    # Pre-warm embedding model in background
    async def _warm_embeddings():
        try:
            from huginn.rag.vector_store import _embedding_model_cached

            if _embedding_model_cached():

                def _do_warm():
                    from chromadb.utils.embedding_functions import (
                        DefaultEmbeddingFunction,
                    )

                    fn = DefaultEmbeddingFunction()
                    fn(["warmup"])

                await asyncio.to_thread(_do_warm)
                print("[init] Embedding model pre-warmed")
        except Exception as e:
            print(f"[init] Embedding pre-warm skipped: {e}")

    asyncio.create_task(_warm_embeddings())

    # Start MCP health monitor (periodic ping + auto-reconnect)
    monitor_task: asyncio.Task | None = None
    mcp_mgr = get_context().mcp_manager
    if mcp_mgr and mcp_mgr._configs:
        monitor_task = asyncio.create_task(_mcp_health_monitor(mcp_mgr))
        logger.info("[MCP] Health monitor started")

    yield

    # Shutdown: cancel monitor first, then disconnect MCP servers
    if monitor_task and not monitor_task.done():
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        logger.info("[MCP] Health monitor stopped")
    await _shutdown_mcp()


# ── CORS ────────────────────────────────────────────────────────────


def _get_cors_origins() -> list[str]:
    """Return allowed CORS origins."""
    raw = os.environ.get("HUGINN_CORS_ORIGINS", "")
    if raw:
        return [origin.strip() for origin in raw.split(",") if origin.strip()]
    return [
        "http://localhost:3000",
        "http://localhost:1420",
        "http://localhost:8000",
        "tauri://localhost",
        "http://tauri.localhost",
    ]
