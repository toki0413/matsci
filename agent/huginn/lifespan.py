"""Application lifespan, MCP initialization, and CORS configuration."""

from __future__ import annotations

import asyncio
import logging
import os
import sqlite3
import sys
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Any

from fastapi import FastAPI

from huginn.config import get_config
from huginn.pet import configure_pet, get_pet_bus
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
    origin: str = "builtin",
    timeout: float = 60.0,
) -> bool:
    """Connect to a single MCP server with timeout and full error containment.

    默认 60s: 本地 stdio 子进程冷启动要 import mcp.server 才能应答握手,
    慢文件系统/冷 pyc 下这一步可能 15-25s, 旧 10s 超时会在 initialize 中途
    被取消, 进而触发 anyio 的 cancel-scope 竞态并使 server 启动踩坑。
    """
    try:
        await asyncio.wait_for(manager.connect(config, origin=origin), timeout=timeout)
        return True
    except TimeoutError:
        logger.info(f"[MCP] Warning: {name} connection timed out ({timeout}s)")
    except Exception as e:
        logger.info(f"[MCP] Warning: failed to connect to {name}: {e}")
    return False


def _load_mcp_json_servers(base: Path) -> list[tuple[str, Any]]:
    """从仓库根的 `.mcp.json` 读取 `mcpServers`（标准 MCP 配置文件格式）。

    让 huginn 真正消费 IDE/桌面共用那份 `.mcp.json` —— 例如 `flint-chart`
    （`npx -y flint-chart-mcp`）这类外部 MCP server 也会被拉起并注册工具，
    而不只是 IDE 能连。best-effort：文件缺失/解析失败/单条配置非法都直接跳过，
    绝不阻塞启动。条目格式见 huginn.mcp_client.MCPServerConfig。
    """
    from huginn.mcp_client import MCPServerConfig

    mcp_json = base / ".mcp.json"
    if not mcp_json.is_file():
        return []
    try:
        import json

        data = json.loads(mcp_json.read_text(encoding="utf-8"))
        servers = data.get("mcpServers", {}) or {}
    except Exception as e:
        logger.info(f"[MCP] Could not parse .mcp.json, skipping: {e}")
        return []

    out: list[tuple[str, Any]] = []
    for name, cfg in servers.items():
        if not isinstance(cfg, dict):
            continue
        out.append((
            name,
            MCPServerConfig(
                name=name,
                command=cfg.get("command", "python"),
                args=list(cfg.get("args", []) or []),
                env=dict(cfg.get("env") or {}),
                transport=cfg.get("transport", "stdio"),
                url=cfg.get("url"),
            ),
        ))
    return out


async def _bg_register_optional_tools():
    """Register optional tools in the background after startup.

    Runs in a thread pool because importlib.import_module is blocking.
    Failures are logged but never crash the server.
    """
    try:
        await asyncio.to_thread(_do_register_optional_tools_sync)
    except Exception as e:
        logger.warning(f"[startup] background tool registration failed: {e}")


def _do_register_optional_tools_sync():
    """Sync helper for _bg_register_optional_tools."""
    from huginn.tools import register_optional_tools
    register_optional_tools()


async def _init_mcp_tools():
    """Connect to local MCP servers and register their tools.

    Failures are fully contained — this function never raises.
    """
    try:
        from huginn.mcp_client import MCPClientManager, MCPServerConfig
        from huginn.tools.mcp_adapter import register_mcp_tools

        get_context().mcp_manager = MCPClientManager()
        # dev: repo root (three parents up from huginn/lifespan.py).
        # frozen: PyInstaller lands --add-data payloads at sys._MEIPASS, and
        # __file__ in a onedir build sits under _internal/, so the naive three-
        # parents-up points one level too high and the bundled servers/.mcp.json
        # are unreachable.
        base = Path(getattr(sys, "_MEIPASS", Path(__file__).parent.parent.parent))

        servers: list[tuple[str, MCPServerConfig]] = []
        # 来源归一 (catalog 用): builtin = servers/ 目录内置, .mcp.json = 根配置.
        origin_by_name: dict[str, str] = {}

        mat_db_path = base / "servers" / "mat-db-mcp" / "server.py"
        if mat_db_path.exists():
            # 透传父进程 env, 并把用户配置的 MP_API_KEY 注入 mat-db MCP 子进程,
            # 这样无需重启后端也能用真实 Materials Project (改 key 走 /config 重连)。
            mat_db_env = {**os.environ}
            mp_key = None
            try:
                from huginn.config import get_config
                mp_key = get_config().mp_api_key
            except Exception as e:
                # best-effort: 读不到 MP_API_KEY 就不注入, 但不静默吞掉——留日志可观测.
                logger.debug(f"[MCP] could not read MP_API_KEY for mat-db: {e}", exc_info=True)
            if mp_key:
                mat_db_env["MP_API_KEY"] = mp_key
            servers.append(
                ("mat-db", MCPServerConfig(
                    # command 用 sys.executable (服务进程自身解释器): 保证 MCP
                    # 子进程能 import mcp (与后端同环境)。用 "python" 会解析到
                    # PATH 里的系统解释器, 常缺 mcp 导致子进程 ModuleNotFoundError。
                    name="mat-db", command=sys.executable, args=[str(mat_db_path)], env=mat_db_env,
                ))
            )

        math_path = base / "servers" / "math-anything-mcp" / "server.py"
        if math_path.exists():
            servers.append(
                ("math-anything", MCPServerConfig(name="math-anything", command=sys.executable, args=[str(math_path)]))
            )

        # 消费仓库根 .mcp.json 里的 mcpServers（标准 MCP 配置，IDE/桌面共用）。
        # 例如 flint-chart（npx -y flint-chart-mcp）会被拉起并注册工具，真正进入
        # agent 的工具池，而不只是 IDE 能连。best-effort，连不上只是告警不阻塞。
        mcp_json_servers = _load_mcp_json_servers(base)
        servers.extend(mcp_json_servers)
        # 来源归一: .mcp.json 来源单独标 origin (其余 builtin 走默认).
        for _name, _cfg in mcp_json_servers:
            origin_by_name[_name] = ".mcp.json"

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

        # Connect to MCP servers sequentially. 并行连接在慢文件系统下会互相
        # 争抢 IO (每个子进程冷导入 mcp.server 都要 15-25s), 放大延迟到超时;
        # 而超时取消进行中的握手会触发 anyio 的 cancel-scope 竞态 (见 mcp_client
        # stdio_client). 逐个连接 + 每连接独立超时最稳, 且 _connect_mcp_server
        # 内部已 try/except 全量兜底, 不会因单个失败阻塞后续.
        for name, cfg in servers:
            await _connect_mcp_server(
                get_context().mcp_manager, name, cfg,
                origin=origin_by_name.get(name, "builtin"),
            )

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

    # ── Catalog: 采集一次 (骨架期), 让统一清单带上已连接的 MCP server ──
    try:
        _cat = getattr(get_context(), "catalog", None)
        if _cat is not None:
            _cat.discover_all(mcp_manager=get_context().mcp_manager)
    except Exception as e:
        logger.warning(f"[Catalog] discover after MCP init failed: {e}")

    # ── Load Star plugins ─────────────────────────────────────────
    await _load_star_plugins()


def _make_plugin_context_factory():
    """创建注入真实依赖的 context_factory.

    之前 PluginLoader 用 default_context_factory, 只填 plugin_name,
    llm_client / tool_registry / storage / event_bus 全是 None.
    插件调 Star.llm / Star.storage 必然 RuntimeError.
    这里从 ServerContext 拿真实组件注入.
    """
    from huginn.api.context import PluginContext
    from huginn.plugins.event_bus import EventBus
    from huginn.tools.registry import ToolRegistry

    bus = EventBus()

    class _LLMAdapter:
        """适配 ModelRegistry 到 LLMClient 协议 (chat + stream)."""

        def __init__(self) -> None:
            from huginn.config import get_config
            from huginn.models.registry import ModelRegistry

            self._registry = ModelRegistry.from_config(get_config())
            self._default_alias: str | None = None
            for ref in self._registry.list():
                if ref.enabled:
                    self._default_alias = ref.alias
                    break

        def _resolve(self, model: str | None):
            alias = model or self._default_alias
            if alias is None:
                raise RuntimeError("no enabled LLM model configured for plugins")
            return self._registry.get(alias)

        @staticmethod
        def _to_lc(messages: list[dict], system: str):
            from langchain_core.messages import (
                AIMessage,
                HumanMessage,
                SystemMessage,
            )

            msgs = []
            if system:
                msgs.append(SystemMessage(content=system))
            for m in messages:
                role = m.get("role", "user")
                content = m.get("content", "")
                if role == "system":
                    msgs.append(SystemMessage(content=content))
                elif role == "assistant":
                    msgs.append(AIMessage(content=content))
                else:
                    msgs.append(HumanMessage(content=content))
            return msgs

        async def chat(
            self,
            messages: list[dict],
            *,
            system: str = "",
            model: str | None = None,
            **kwargs: Any,
        ) -> str:
            llm = self._resolve(model)
            resp = await llm.ainvoke(self._to_lc(messages, system))
            return getattr(resp, "content", str(resp))

        def stream(
            self,
            messages: list[dict],
            *,
            system: str = "",
            model: str | None = None,
            **kwargs: Any,
        ):
            return self._stream(self._resolve(model), messages, system)

        async def _stream(self, llm, messages, system):
            async for chunk in llm.astream(self._to_lc(messages, system)):
                text = getattr(chunk, "content", None)
                if text:
                    yield text

    class _DictStorage:
        """简单的内存 KV 存储, 满足 Storage 协议.

        ponytail: dict 够用. 升级路径: 文件持久化.
        """

        def __init__(self) -> None:
            self._d: dict[str, Any] = {}

        def get(self, key: str, default: Any = None) -> Any:
            return self._d.get(key, default)

        def set(self, key: str, value: Any) -> None:
            self._d[key] = value

        def delete(self, key: str) -> bool:
            return self._d.pop(key, None) is not None

        def keys(self) -> list[str]:
            return list(self._d)

    _llm = _LLMAdapter()

    def factory(meta):
        return PluginContext(
            llm_client=_llm,
            tool_registry=ToolRegistry,
            storage=_DictStorage(),
            event_bus=bus,
            plugin_name=meta.name,
        )

    return factory, bus


async def _load_star_plugins() -> None:
    """Discover and load Star plugins from the plugins directory.

    Plugins live in ``huginn/plugins/<name>/`` with a ``metadata.yaml``
    and ``main.py`` defining a ``Star`` subclass.  The loader is
    fault-tolerant — one bad plugin won't prevent the server from
    starting.
    """
    try:
        from huginn.plugins.loader import PluginLoader

        base = Path(__file__).resolve().parent  # huginn/
        plugins_dir = base / "plugins"

        context_factory, event_bus = _make_plugin_context_factory()
        loader = PluginLoader(
            plugins_dir=str(plugins_dir),
            context_factory=context_factory,
            event_bus=event_bus,
        )
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
            logger.debug("best-effort op failed", exc_info=True)
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
                logger.debug("best-effort op failed", exc_info=True)
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


def _ls_setup_runtime(app: FastAPI) -> None:
    """Switch to structured logging early and size the default executor."""
    setup_json_logging()
    try:
        import concurrent.futures

        pool_size = int(os.environ.get("HUGINN_THREAD_POOL_SIZE", "64"))
        loop = asyncio.get_running_loop()
        loop.set_default_executor(
            concurrent.futures.ThreadPoolExecutor(max_workers=pool_size)
        )
        logger.info(f"[startup] ThreadPoolExecutor set to {pool_size} workers")
    except Exception as e:
        logger.warning(f"[startup] could not set thread pool size: {e}")


async def _ls_run_migrations(app: FastAPI) -> None:
    try:
        from huginn.utils.migrations import run_all_migrations

        await asyncio.to_thread(run_all_migrations)
    except Exception as e:
        logger.warning(f"[migrations] startup sweep failed: {e}")


async def _ls_run_startup_probe(app: FastAPI) -> None:
    try:
        from huginn.diagnostics.startup_validator import run_startup_check

        await asyncio.to_thread(run_startup_check)
    except Exception as e:
        logger.warning(f"[startup] dep/dead-code probe failed: {e}")


def _ls_bootstrap_bot_bridge(app: FastAPI) -> None:
    try:
        from huginn.bot.bridge import bootstrap_qq_from_env
        from huginn.bot.wechat_bridge import bootstrap_wechat_from_env

        bootstrap_qq_from_env()
        bootstrap_wechat_from_env()
        logger.info("[startup] bot bridge configs bootstrapped from env")
    except Exception as e:
        logger.warning(f"[startup] bot bridge bootstrap failed: {e}")


def _ls_set_main_loop(app: FastAPI) -> None:
    try:
        from huginn.events.integration import set_main_loop

        set_main_loop(asyncio.get_running_loop())
    except Exception:
        logger.debug("failed to set main event loop", exc_info=True)


def _ls_register_tools(app: FastAPI) -> None:
    try:
        from huginn.tools import register_core_tools

        register_core_tools()
        logger.info("[startup] core tools registered")
        from huginn.utils.concurrency import track_task

        track_task(_bg_register_optional_tools(), name="bg-register-optional-tools")
    except Exception as e:
        logger.warning(f"[startup] tool registration failed: {e}")


def _ls_spawn_mcp_init(app: FastAPI) -> None:
    try:
        from huginn.utils.concurrency import track_task

        track_task(_init_mcp_tools(), name="init-mcp-tools")
    except Exception as e:
        logger.warning(f"[startup] MCP init task spawn failed: {e}")


def _ls_init_knowledge_base(app: FastAPI, state: dict[str, Any]) -> Any | None:
    if not (_KB_AVAILABLE and get_context().kb is None):
        return None
    try:
        cfg = get_config()
        kb = get_knowledge_base(cfg.workspace)
        get_context().kb = kb
    except Exception as e:
        logger.info(f"[KB] Warning: could not initialize knowledge base: {e}")
        return None
    state["kb"] = kb
    return kb


def _ls_init_encrypted_rag(app: FastAPI, state: dict[str, Any]) -> Any | None:
    if get_context().encrypted_rag is not None:
        return None
    try:
        cfg = get_config()
        if not (
            cfg.encryption_enabled
            and (cfg.encryption_password or cfg.encryption_key_file)
        ):
            return None
        from huginn.rag.encrypted_rag import EncryptedRAGManager

        persist_dir = str(Path(cfg.workspace) / ".huginn_encrypted_rag")
        mgr = EncryptedRAGManager(
            password=cfg.encryption_password,
            key_file=cfg.encryption_key_file,
            persist_dir=persist_dir,
            doc_encryption=cfg.encrypt_rag_documents,
            meta_encryption=cfg.encrypt_rag_metadata,
        )
        get_context().encrypted_rag = mgr
        logger.info("[EncryptedRAG] initialized (persist_dir=%s)", persist_dir)
    except Exception as e:
        logger.warning(f"[EncryptedRAG] could not initialize: {e}")
        return None
    state["encrypted_rag"] = mgr
    return mgr


def _ls_init_codebase(app: FastAPI, state: dict[str, Any]) -> Any | None:
    if not (_CODEBASE_AVAILABLE and get_context().codebase is None):
        return None
    try:
        cfg = get_config()
        codebase = get_codebase_index(cfg.workspace)
        get_context().codebase = codebase
    except Exception as e:
        logger.info(f"[Codebase] Warning: could not initialize codebase index: {e}")
        return None
    state["codebase"] = codebase
    return codebase


def _ls_configure_pet(app: FastAPI) -> None:
    try:
        cfg = get_config()
        # load 先恢复 gamification (level/xp/hunger), configure 再覆盖其余字段
        get_pet_bus().load_state()
        configure_pet(cfg.pet_name, cfg.pet_personality, accessories=cfg.pet_accessories)
    except Exception as e:
        logger.info(f"[Pet] Warning: could not configure pet: {e}")


async def _ls_warm_embeddings() -> None:
    """Pre-warm the embedding model in the background."""
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


def _ls_start_embedding_warmup(app: FastAPI) -> asyncio.Task:
    return asyncio.create_task(_ls_warm_embeddings())


def _ls_start_mcp_monitor(app: FastAPI) -> asyncio.Task | None:
    mcp_mgr = get_context().mcp_manager
    if mcp_mgr and mcp_mgr._configs:
        monitor_task = asyncio.create_task(_mcp_health_monitor(mcp_mgr))
        logger.info("[MCP] Health monitor started")
        return monitor_task
    return None


def _ls_start_memory_maintainer(app: FastAPI) -> Any | None:
    try:
        from huginn.memory.maintainer import MemoryMaintainer

        memory_maintainer = MemoryMaintainer(interval_hours=1)
        memory_maintainer.start()
        logger.info("[startup] memory maintainer started")
        return memory_maintainer
    except Exception as exc:
        logger.debug(f"[startup] memory maintainer skipped: {exc}")
        return None


async def _ls_pmk_loop() -> None:
    """Background persona-memory-knowledge consolidation loop."""
    from huginn.memory.task_state import get_tracker

    interval = 300  # 5 min
    while True:
        try:
            await asyncio.sleep(interval)
            ctx = get_context()
            # 1. Memory: run maintenance (decay + dedup)
            if ctx.memory_manager:
                try:
                    ctx.memory_manager.maintenance()
                    stats = ctx.memory_manager.stats()
                    logger.debug(f"[PMK] memory maintenance done: {stats}")
                except Exception as e:
                    logger.debug(f"[PMK] memory maintenance failed: {e}")
            # 2. KB: check for unindexed docs
            if ctx.kb:
                try:
                    docs = (
                        ctx.kb.list_documents()
                        if hasattr(ctx.kb, "list_documents")
                        else []
                    )
                    logger.debug(f"[PMK] KB has {len(docs)} docs")
                except Exception:
                    logger.debug("KB document list failed", exc_info=True)
            # 3. Task state: log active threads
            tracker = get_tracker()
            if tracker._cache:
                logger.debug(f"[PMK] tracking {len(tracker._cache)} threads")
        except asyncio.CancelledError:
            logger.debug("best-effort op failed", exc_info=True)
            break
        except Exception as e:
            logger.debug(f"[PMK] loop error: {e}")


def _ls_start_pmk_loop(app: FastAPI) -> asyncio.Task | None:
    try:
        pmk_task = asyncio.create_task(_ls_pmk_loop())
        logger.info("[startup] persona-memory-knowledge loop started (300s interval)")
        return pmk_task
    except Exception as exc:
        logger.debug(f"[startup] PMK loop skipped: {exc}")
        return None


def _ls_start_job_monitor(app: FastAPI) -> Any | None:
    try:
        from huginn.hpc.monitor import JobMonitor

        job_monitor = JobMonitor(workspace=os.environ.get("HUGINN_WORKSPACE", "."))
        job_monitor.start()
        logger.info("[startup] HPC job monitor started")
        return job_monitor
    except Exception as exc:
        logger.debug(f"[startup] HPC job monitor skipped: {exc}")
        return None


async def _ls_dispatch_loaded(app: FastAPI) -> None:
    """Dispatch ON_HUGINN_LOADED once the engine is ready (best-effort)."""
    try:
        from huginn.api.event import Event, EventType
        from huginn.plugins.event_bus import EventBus as _PluginBus

        _bus = _PluginBus()
        await _bus.dispatch(Event(
            type=EventType.ON_HUGINN_LOADED,
            data={"status": "ready"},
        ))
    except Exception:
        logger.debug("ON_HUGINN_LOADED dispatch failed (non-fatal)", exc_info=True)


def _ls_stop_service(service: Any) -> None:
    if service:
        with suppress(Exception):
            service.stop()


async def _ls_cancel_task(task: asyncio.Task | None, message: str) -> None:
    if task and not task.done():
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        logger.info(message)


async def _ls_shutdown_telemetry() -> None:
    """Flush buffered OTLP traces; never raises."""
    try:
        from huginn.otel import get_default_exporter

        exp = get_default_exporter()
        if exp is not None:
            exp.shutdown(block=False)
            logger.info("[shutdown] telemetry OTLP exporter flushed")
    except Exception as e:
        logger.debug(f"[shutdown] telemetry exporter flush skipped: {e}")


def _ls_lock_encrypted_rag() -> None:
    try:
        _rag = getattr(get_context(), "encrypted_rag", None)
        if _rag is not None:
            _rag.lock()
            logger.info("[shutdown] encrypted RAG vault locked")
    except Exception as e:
        logger.warning(f"[shutdown] failed to lock encrypted RAG: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    state: dict[str, Any] = {}

    _ls_setup_runtime(app)
    await _ls_run_migrations(app)
    await _ls_run_startup_probe(app)
    _ls_bootstrap_bot_bridge(app)
    _ls_set_main_loop(app)
    _ls_register_tools(app)
    _ls_spawn_mcp_init(app)

    _ls_init_knowledge_base(app, state)
    _ls_init_encrypted_rag(app, state)
    _ls_init_codebase(app, state)
    _ls_configure_pet(app)

    state["warmup_task"] = _ls_start_embedding_warmup(app)
    state["monitor_task"] = _ls_start_mcp_monitor(app)
    state["memory_maintainer"] = _ls_start_memory_maintainer(app)
    state["pmk_task"] = _ls_start_pmk_loop(app)
    state["job_monitor"] = _ls_start_job_monitor(app)
    await _ls_dispatch_loaded(app)

    yield

    # Shutdown: cancel background tasks first so they don't touch closing resources
    _ls_stop_service(state["memory_maintainer"])
    _ls_stop_service(state["job_monitor"])
    await _ls_cancel_task(
        state["warmup_task"], "[shutdown] embedding warmup task cancelled"
    )
    await _ls_cancel_task(state["monitor_task"], "[MCP] Health monitor stopped")
    await _ls_cancel_task(state["pmk_task"], "[shutdown] PMK loop cancelled")

    await _shutdown_mcp()
    await _ls_shutdown_telemetry()
    _ls_lock_encrypted_rag()
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

