"""Application lifespan, MCP initialization, and CORS configuration."""

from __future__ import annotations

import asyncio
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

    yield
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
    ]
