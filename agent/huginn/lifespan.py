"""Application lifespan, MCP initialization, and CORS configuration."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from huginn.config import HuginnConfig, get_config
from huginn.pet import configure_pet
from huginn.server_core import (
    _CODEBASE_AVAILABLE,
    _KB_AVAILABLE,
    get_codebase_index,
    get_context,
    get_knowledge_base,
)
from huginn.utils.json_logging import setup_json_logging

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
        await asyncio.wait_for(manager.connect(config), timeout=timeout)
        return True
    except asyncio.TimeoutError:
        logger.info(f"[MCP] Warning: {name} connection timed out ({timeout}s)")
    except Exception as e:
        logger.info(f"[MCP] Warning: failed to connect to {name}: {e}")
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

        # ToolUniverse (curated: only materials-science tools pass the whitelist
        # in mcp_adapter.MATERIAL_SCIENCE_TOOL_WHITELIST). Off by default — user
        # must set HUGINN_TOOLUNIVERSE_ENABLED=1 and `pip install tooluniverse`.
        tu_enabled = os.environ.get("HUGINN_TOOLUNIVERSE_ENABLED", "0") == "1"
        if tu_enabled:
            servers.append(
                ("tooluniverse", MCPServerConfig(
                    name="tooluniverse",
                    command="python",
                    args=["-m", "tooluniverse.smcp_server"],
                    env={},
                ))
            )

        for name, cfg in servers:
            task = asyncio.create_task(
                _connect_mcp_server(get_context().mcp_manager, name, cfg)
            )
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            await task

        # ToolUniverse 单独走白名单 (只注册 7 个材料相关工具, 过滤 350+ 生物医学工具).
        # 不能先做一次无过滤的 generic 注册 —— 那会把 343 个非白名单 tooluniverse 工具
        # 也灌进 registry, 白名单就形同虚设了. 所以 tu_enabled 时按 server 分别注册.
        if tu_enabled:
            from huginn.tools.mcp_adapter import MATERIAL_SCIENCE_TOOL_WHITELIST

            registered: list = []
            for srv in list(get_context().mcp_manager._configs.keys()):
                if srv == "tooluniverse":
                    registered.extend(register_mcp_tools(
                        get_context().mcp_manager,
                        server_name="tooluniverse",
                        whitelist=MATERIAL_SCIENCE_TOOL_WHITELIST,
                    ))
                else:
                    registered.extend(
                        register_mcp_tools(get_context().mcp_manager, server_name=srv)
                    )
            logger.info(
                "[MCP] Registered %d tools (ToolUniverse curated by whitelist)",
                len(registered),
            )
        else:
            registered = register_mcp_tools(get_context().mcp_manager)
            logger.info(f"[MCP] Registered {len(registered)} tools from MCP servers")
    except Exception as e:
        logger.info(f"[MCP] Warning: Could not initialize MCP tools: {e}")

    # ── Load Star plugins ─────────────────────────────────────────
    await _load_star_plugins()


async def _load_star_plugins() -> None:
    """Discover and load Star plugins from the plugins directory.

    Plugins live in ``huginn/plugins/<name>/`` with a ``metadata.yaml``
    and ``main.py`` defining a ``Star`` subclass.  The loader is
    fault-tolerant — one bad plugin won't prevent the server from
    starting.
    """
    try:
        from pathlib import Path

        from huginn.plugins.loader import PluginLoader

        base = Path(__file__).resolve().parent  # huginn/
        plugins_dir = base / "plugins"

        loader = PluginLoader(plugins_dir=str(plugins_dir))
        get_context().plugin_loader = loader

        discovered = loader.discover()
        loaded = 0
        for plugin_dir in discovered:
            try:
                await loader.load_one_async(plugin_dir)
                loaded += 1
            except Exception as e:
                logger.warning("[plugins] Failed to load %s: %s", plugin_dir.name, e)

        if loaded:
            logger.info("[plugins] Loaded %d Star plugin(s)", loaded)
    except ImportError:
        logger.debug("[plugins] Plugin framework not available")
    except Exception as e:
        logger.warning("[plugins] Plugin loading failed: %s", e)


async def _shutdown_mcp():
    """Disconnect all MCP servers."""
    if get_context().mcp_manager:
        await get_context().mcp_manager.disconnect_all()
        get_context().mcp_manager = None


async def _close_sqlite_stores() -> None:
    """Close SQLite connections held by long-lived singletons on shutdown.

    Every close is wrapped so one failure doesn't block the others. The
    longterm memory and credential store open per-operation connections
    (context-managed, closed automatically), so they have nothing for us
    to do here — we only touch the stores that keep a persistent connection
    open for the lifetime of the process.
    """
    # research_log — module-level singleton; may never have been touched if
    # no research records were written this run.
    try:
        import huginn.research_log as _rl_mod

        _rl = getattr(_rl_mod, "_research_log_singleton", None)
        if _rl is not None and hasattr(_rl, "close"):
            _rl.close()
            logger.info("[shutdown] research_log connection closed")
    except Exception as e:
        logger.warning("[shutdown] failed to close research_log: %s", e)

    # anomaly_log + campaign store hang off the AgentFactory, which is built
    # lazily — they won't exist if no agent was ever created this session.
    try:
        factory = getattr(get_context(), "agent_factory", None)
        if factory is None:
            return

        anomaly_store = getattr(factory, "_anomaly_store", None)
        if anomaly_store is not None and hasattr(anomaly_store, "close"):
            anomaly_store.close()
            logger.info("[shutdown] anomaly_log connection closed")

        scheduler = getattr(factory, "_shared_scheduler", None)
        if scheduler is None:
            return

        # Cancel in-flight async jobs before pulling the store out from
        # under them — otherwise their finally-block writes hit a closed
        # handle and the WAL checkpoint may not flush.
        live = getattr(scheduler, "_live_tasks", {}) or {}
        pending: list[asyncio.Task] = []
        for _jid, task in list(live.items()):
            if not task.done():
                task.cancel()
                pending.append(task)
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            logger.info(
                "[shutdown] cancelled %d in-flight scheduler task(s)", len(pending)
            )

        # Stop the background drainer coroutine if one is running.
        _stop = getattr(scheduler, "stop", None)
        if callable(_stop):
            _stop()

        store = getattr(scheduler, "store", None)
        if store is not None and hasattr(store, "close"):
            store.close()
            logger.info("[shutdown] campaign store connection closed")
    except Exception as e:
        logger.warning("[shutdown] failed to close factory stores: %s", e)

    # ── WAL checkpoint + VACUUM ────────────────────────────────────
    # Flush the WAL so the -wal/-shm sidecar files don't grow unbounded
    # across restarts.  Without this, long-running deployments accumulate
    # multi-GB WAL files that slow startup and waste disk.
    await _wal_checkpoint_all()


async def _wal_checkpoint_all() -> None:
    """Run ``PRAGMA wal_checkpoint(TRUNCATE)`` on all known SQLite DBs."""
    import sqlite3
    from pathlib import Path

    db_paths: list[Path] = []

    # Long-term memory
    try:
        from huginn.config import get_config
        cfg = get_config()
        mem_db = getattr(cfg.memory, "long_term_db", None)
        if mem_db:
            db_paths.append(Path(mem_db))
    except Exception:
        logger.debug("long-term memory db 路径解析失败", exc_info=True)

    # Checkpointer
    try:
        ctx = get_context()
        ckpt = getattr(ctx, "checkpointer_path", None) or getattr(ctx.config, "checkpointer_path", None)
        if ckpt:
            db_paths.append(Path(ckpt))
    except Exception:
        logger.debug("checkpointer 路径解析失败", exc_info=True)

    # Research log
    try:
        import huginn.research_log as _rl_mod
        _rl = getattr(_rl_mod, "_research_log_singleton", None)
        if _rl is not None:
            db_path = getattr(_rl, "db_path", None)
            if db_path:
                db_paths.append(Path(db_path))
    except Exception:
        logger.debug("research log db 路径解析失败", exc_info=True)

    for db_path in db_paths:
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path), timeout=5)
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            conn.close()
            logger.info("[shutdown] WAL checkpoint: %s", db_path.name)
        except Exception as e:
            logger.warning("[shutdown] WAL checkpoint failed for %s: %s", db_path.name, e)


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
    # Switch the root logger to structured JSON output early so every startup
    # log line below is already correlated. Opt out with HUGINN_JSON_LOGS=0.
    setup_json_logging()
    # Run pending schema migrations for every SQLite store up front, so
    # backups + integrity checks happen in one place rather than lazily
    # when each store is first touched. Failures are contained — a bad
    # migration on one DB shouldn't stop the whole app from booting.
    try:
        from huginn.utils.migrations import run_all_migrations
        await asyncio.to_thread(run_all_migrations)
    except Exception as e:
        logger.warning(f"[migrations] startup sweep failed: {e}")
    # Store the main event loop so sync code (tool adapter running in
    # a thread) can schedule event publishes back on the main loop
    try:
        from huginn.events.integration import set_main_loop
        set_main_loop(asyncio.get_running_loop())
    except Exception:
        pass
    await _init_mcp_tools()
    if _KB_AVAILABLE and get_context().kb is None:
        try:
            cfg = get_config()
            get_context().kb = get_knowledge_base(cfg.workspace)
        except Exception as e:
            logger.info(f"[KB] Warning: could not initialize knowledge base: {e}")
    if _CODEBASE_AVAILABLE and get_context().codebase is None:
        try:
            cfg = get_config()
            get_context().codebase = get_codebase_index(cfg.workspace)
        except Exception as e:
            logger.info(f"[Codebase] Warning: could not initialize codebase index: {e}")
    try:
        cfg = get_config()
        configure_pet(cfg.pet_name, cfg.pet_personality)
    except Exception as e:
        logger.info(f"[Pet] Warning: could not configure pet: {e}")

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
                logger.info("[init] Embedding model pre-warmed")
        except Exception as e:
            logger.info(f"[init] Embedding pre-warm skipped: {e}")

    warmup_task = asyncio.create_task(_warm_embeddings())

    # Start MCP health monitor (periodic ping + auto-reconnect)
    monitor_task: asyncio.Task | None = None
    mcp_mgr = get_context().mcp_manager
    if mcp_mgr and mcp_mgr._configs:
        monitor_task = asyncio.create_task(_mcp_health_monitor(mcp_mgr))
        logger.info("[MCP] Health monitor started")

    yield

    # Shutdown: cancel background tasks first so they don't touch closing resources
    if warmup_task and not warmup_task.done():
        warmup_task.cancel()
        try:
            await warmup_task
        except asyncio.CancelledError:
            pass
        logger.info("[shutdown] embedding warmup task cancelled")

    if monitor_task and not monitor_task.done():
        monitor_task.cancel()
        try:
            await monitor_task
        except asyncio.CancelledError:
            pass
        logger.info("[MCP] Health monitor stopped")
    await _shutdown_mcp()

    # Close persistent SQLite connections so WAL checkpoints flush and the
    # file handles drop cleanly. Done after MCP teardown so nothing new will
    # try to touch the databases while we're tearing them down.
    await _close_sqlite_stores()


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

