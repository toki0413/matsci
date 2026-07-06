"""MCP Client Manager — connects to external MCP servers and exposes their tools.

Supports stdio-based MCP servers (e.g., mat-db-mcp, math-anything-mcp).
Discovers tools dynamically and routes calls to the correct server.
Provides health monitoring, automatic reconnection with exponential backoff,
and idempotent tool registration.

Also provides a registry interface for managing multiple server configurations
(list_servers, register_server, connect_server, disconnect_server, remove_server).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import platform
import signal
import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# SSE transport lives in mcp.client.sse; older releases may not ship it, so
# import defensively and fall back to None — connect() raises a clear error
# if someone actually asks for SSE on a build that lacks it.
try:
    from mcp.client.sse import sse_client as _sse_client
except ImportError:  # pragma: no cover - depends on installed mcp version
    _sse_client = None

# Notification types used to fan out resource-update events to subscribers.
# Wrapped the same way for forward compatibility.
try:
    import anyio.lowlevel as _anyio_lowlevel
    from mcp.types import (
        ResourceUpdatedNotification as _ResourceUpdatedNotification,
        ServerNotification as _ServerNotification,
    )
except ImportError:  # pragma: no cover
    _anyio_lowlevel = None
    _ResourceUpdatedNotification = None
    _ServerNotification = None

logger = logging.getLogger(__name__)

# Reconnection defaults
_BACKOFF_BASE: float = 1.0       # initial delay in seconds
_BACKOFF_FACTOR: float = 2.0     # multiplier per retry
_BACKOFF_MAX: float = 30.0       # maximum delay cap
_MAX_RECONNECT_ATTEMPTS: int = 5  # give up after this many consecutive failures
_HEALTH_CHECK_INTERVAL: float = 30.0  # seconds between health polls
_HEALTH_CHECK_TIMEOUT: float = 5.0    # seconds before a health probe is considered hung


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server connection."""

    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    # "stdio" (default) or "sse". SSE ignores command/args and uses url instead.
    transport: str = "stdio"
    url: str | None = None  # SSE endpoint, required when transport == "sse"


@dataclass
class MCPToolInfo:
    """Discovered MCP tool with routing info."""

    name: str
    description: str
    input_schema: dict[str, Any]
    server_name: str


class MCPClientManager:
    """Manages connections to multiple MCP servers.

    Usage:
        mgr = MCPClientManager()
        await mgr.connect(MCPServerConfig("mat-db", "python", ["server.py"]))
        tools = mgr.list_tools()
        result = await mgr.call_tool("query_materials_project", {"formula": "Si"})
        await mgr.disconnect_all()
    """

    def __init__(self):
        self._sessions: dict[str, ClientSession] = {}
        self._clients: dict[str, Any] = {}  # stdio client contexts
        self._tools: list[MCPToolInfo] = []
        self._tool_index: dict[str, MCPToolInfo] = {}
        self._initialized: set[str] = set()
        # Stored configs for reconnection
        self._configs: dict[str, MCPServerConfig] = {}
        # Reconnection tracking
        self._consecutive_failures: dict[str, int] = {}
        self._last_health_check: dict[str, float] = {}
        # Registry for server configs (sync interface for tests)
        self._registry: dict[str, dict[str, Any]] = {}
        # Serializes connect/disconnect to prevent races between
        # concurrent lifecycle operations on the same server.
        self._lock = asyncio.Lock()
        # call_tool_with_retry 的连续错误计数（按服务器名）
        self._consecutive_errors: dict[str, int] = {}
        # 连接缓存：cache_key -> server_name，避免重复握手
        self._connection_cache: dict[str, str] = {}
        # Resource subscription callbacks: uri -> [callback, ...]
        self._resource_callbacks: dict[str, list[Any]] = {}

    # ------------------------------------------------------------------ #
    # Registry API (sync, used by tests and CLI)
    # ------------------------------------------------------------------ #
    def list_servers(self) -> list[dict[str, Any]]:
        """List registered server configurations."""
        return [
            {"name": name, "config": cfg}
            for name, cfg in self._registry.items()
        ]

    def register_server(self, name: str, config: dict[str, Any]) -> None:
        """Register a server configuration without connecting."""
        self._registry[name] = config

    def remove_server(self, name: str) -> None:
        """Remove a registered server configuration."""
        self._registry.pop(name, None)

    def connect_server(self, name: str) -> None:
        """Connect to a registered server by name (sync wrapper)."""
        cfg = self._registry.get(name)
        if not cfg:
            raise ValueError(f"Server '{name}' not registered")

        config = MCPServerConfig(
            name=name,
            command=cfg.get("command", "python"),
            args=cfg.get("args", []),
            env=cfg.get("env"),
            transport=cfg.get("transport", "stdio"),
            url=cfg.get("url"),
        )

        try:
            try:
                loop = asyncio.get_running_loop()
                # Schedule in running loop
                asyncio.create_task(self.connect(config))
            except RuntimeError:
                # No running loop — create one
                loop = asyncio.new_event_loop()
                loop.run_until_complete(self.connect(config))
        except Exception as e:
            logger.warning(f"Failed to connect MCP server '{name}': {e}")
            raise

    def disconnect_server(self, name: str) -> None:
        """Disconnect a server by name (sync wrapper)."""
        try:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
            if loop.is_running():
                asyncio.create_task(self.disconnect(name))
            else:
                loop.run_until_complete(self.disconnect(name))
        except Exception as e:
            logger.warning(f"Disconnect error for '{name}': {e}")

    def load_from_config(self, config: dict[str, dict[str, Any]] | None) -> None:
        """Load MCP server configurations from HuginnConfig.mcp_servers."""
        if not config:
            return
        for name, cfg in config.items():
            self.register_server(name, cfg)
            logger.info(f"Registered MCP server '{name}' from config")

    def load_from_huginn_config(self) -> None:
        """Auto-load from the global HuginnConfig."""
        try:
            from huginn.config import get_settings

            settings = get_settings()
            self.load_from_config(getattr(settings.config, "mcp_servers", None))
        except Exception as e:
            logger.warning(f"Failed to auto-load MCP config: {e}")

    # ------------------------------------------------------------------ #
    # Async connection API
    # ------------------------------------------------------------------ #
    async def connect(self, config: MCPServerConfig) -> None:
        """Connect to an MCP server via stdio or SSE (see config.transport)."""
        async with self._lock:
            if config.name in self._sessions:
                logger.warning(f"MCP server '{config.name}' already connected")
                return

            # Store config for future reconnection attempts
            self._configs[config.name] = config

            # Clear any stale tools from a previous connection (idempotent)
            self._tools = [t for t in self._tools if t.server_name != config.name]
            self._tool_index = {
                k: v for k, v in self._tool_index.items() if v.server_name != config.name
            }

            try:
                if config.transport == "sse":
                    if _sse_client is None:
                        raise RuntimeError(
                            "SSE transport requested but mcp.client.sse is unavailable; "
                            "upgrade the mcp package or use stdio"
                        )
                    if not config.url:
                        raise ValueError(
                            f"SSE server '{config.name}' requires a url"
                        )
                    client = _sse_client(config.url)
                else:
                    params = StdioServerParameters(
                        command=config.command,
                        args=config.args,
                        env=config.env,
                    )
                    client = stdio_client(params)

                read_stream, write_stream = await client.__aenter__()
                # Hand in a message handler so resource-update notifications
                # reach subscribers; see _make_message_handler.
                session = ClientSession(
                    read_stream,
                    write_stream,
                    message_handler=self._make_message_handler(),
                )
                await session.__aenter__()
            except Exception:
                with contextlib.suppress(Exception):
                    await client.__aexit__(None, None, None)
                logger.error(f"Failed to connect to MCP server '{config.name}'")
                raise

            try:
                await session.initialize()

                self._clients[config.name] = client
                self._sessions[config.name] = session
                self._initialized.add(config.name)
                self._consecutive_failures[config.name] = 0
                self._last_health_check[config.name] = time.monotonic()

                tools_result = await session.list_tools()
                for tool in tools_result.tools:
                    info = MCPToolInfo(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema
                        or {"type": "object", "properties": {}},
                        server_name=config.name,
                    )
                    self._tools.append(info)
                    self._tool_index[tool.name] = info

                logger.info(
                    f"Connected to MCP server '{config.name}' with {len(tools_result.tools)} tools"
                )

            except Exception as e:
                with contextlib.suppress(Exception):
                    await session.__aexit__(None, None, None)
                with contextlib.suppress(Exception):
                    await client.__aexit__(None, None, None)
                logger.error(f"Failed to connect to MCP server '{config.name}': {e}")
                raise

    async def connect_sse(self, name: str, url: str) -> None:
        """Connect to an SSE/HTTP MCP server by URL.

        Thin wrapper over :meth:`connect` that builds an SSE config so callers
        don't have to construct MCPServerConfig themselves.
        """
        await self.connect(
            MCPServerConfig(
                name=name, command="", args=[], transport="sse", url=url
            )
        )

    async def disconnect(self, name: str) -> None:
        """Disconnect a specific MCP server."""
        async with self._lock:
            session = self._sessions.pop(name, None)
            client = self._clients.pop(name, None)
            self._initialized.discard(name)
            # 清理连续错误计数和连接缓存中的条目
            self._consecutive_errors.pop(name, None)
            stale_keys = [k for k, v in self._connection_cache.items() if v == name]
            for k in stale_keys:
                self._connection_cache.pop(k, None)

            # Remove tools from this server
            self._tools = [t for t in self._tools if t.server_name != name]
            self._tool_index = {
                k: v for k, v in self._tool_index.items() if v.server_name != name
            }

        if session:
            with contextlib.suppress(Exception):
                await session.__aexit__(None, None, None)
        if client:
            # 先用升级序列清理底层进程，再让上下文管理器收尾
            proc = self._extract_process(client)
            if proc is not None:
                await self._terminate_process(proc, timeout=5.0)
            with contextlib.suppress(Exception):
                await client.__aexit__(None, None, None)

        logger.info(f"Disconnected MCP server '{name}'")

    async def disconnect_all(self) -> None:
        """Disconnect all MCP servers."""
        for name in list(self._sessions.keys()):
            await self.disconnect(name)

    def list_tools(self) -> list[MCPToolInfo]:
        """List all discovered MCP tools."""
        return list(self._tools)

    def get_tool_info(self, name: str) -> MCPToolInfo | None:
        return self._tool_index.get(name)

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call an MCP tool by name."""
        info = self._tool_index.get(name)
        if not info:
            raise ValueError(
                f"MCP tool '{name}' not found. Available: {list(self._tool_index.keys())}"
            )

        session = self._sessions.get(info.server_name)
        if not session:
            raise RuntimeError(f"MCP server '{info.server_name}' not connected")

        result = await session.call_tool(name, arguments)

        # Convert MCP result to simple dict
        outputs = []
        for content in result.content:
            if hasattr(content, "text"):
                outputs.append(content.text)
            else:
                outputs.append(str(content))

        return {
            "success": not result.isError,
            "output": "\n".join(outputs),
            "is_error": result.isError,
        }

    async def read_resource(self, uri: str, server_name: str | None = None) -> str:
        """Read an MCP resource."""
        if server_name:
            session = self._sessions.get(server_name)
            if not session:
                raise RuntimeError(f"MCP server '{server_name}' not connected")
            result = await session.read_resource(uri)
            return result.contents[0].text if result.contents else ""

        # Try all servers
        for _name, session in self._sessions.items():
            try:
                result = await session.read_resource(uri)
                return result.contents[0].text if result.contents else ""
            except Exception:
                continue
        raise ValueError(f"Resource '{uri}' not found on any connected server")

    # ------------------------------------------------------------------ #
    # Prompts
    # ------------------------------------------------------------------ #

    async def list_prompts(self) -> dict[str, list]:
        """List prompts advertised by every connected server.

        Returns ``{server_name: [{name, description, arguments}, ...]}``.
        Servers that don't support prompts (or error out) are skipped, the
        same tolerant style as :meth:`read_resource`.
        """
        result: dict[str, list] = {}
        for name, session in self._sessions.items():
            try:
                prompts = await session.list_prompts()
            except Exception as e:
                logger.debug(f"list_prompts failed on '{name}': {e}")
                continue
            result[name] = [
                {
                    "name": p.name,
                    "description": p.description or "",
                    "arguments": [
                        {
                            "name": a.name,
                            "description": a.description or "",
                            "required": bool(a.required),
                        }
                        for a in (p.arguments or [])
                    ],
                }
                for p in prompts.prompts
            ]
        return result

    async def get_prompt(
        self, name: str, arguments: dict[str, Any] | None = None
    ) -> str:
        """Get a prompt by name from the first server that exposes it.

        Returns the concatenated message text. Raises ``ValueError`` if no
        connected server has the prompt.
        """
        for srv, session in self._sessions.items():
            try:
                result = await session.get_prompt(name, arguments or {})
            except Exception as e:
                logger.debug(f"get_prompt '{name}' failed on '{srv}': {e}")
                continue
            parts: list[str] = []
            for msg in result.messages:
                content = msg.content
                text = getattr(content, "text", None)
                parts.append(text if text is not None else str(content))
            return "\n\n".join(parts)
        raise ValueError(f"Prompt '{name}' not found on any connected server")

    # ------------------------------------------------------------------ #
    # Resource subscriptions
    # ------------------------------------------------------------------ #

    async def subscribe_resource(self, uri: str, callback: Any) -> None:
        """Subscribe to updates for *uri* on all connected servers.

        *callback* is invoked with the uri whenever the server emits a
        ``resources/updated`` notification. It may be sync or async, and
        multiple callbacks per uri are allowed. The subscription request is
        best-effort per server — servers without resource support are skipped.
        """
        callbacks = self._resource_callbacks.setdefault(uri, [])
        if callback not in callbacks:
            callbacks.append(callback)
        for name, session in self._sessions.items():
            try:
                await session.subscribe_resource(uri)
            except Exception as e:
                logger.debug(f"subscribe_resource failed on '{name}': {e}")

    async def unsubscribe_resource(self, uri: str) -> None:
        """Stop receiving updates for *uri* and drop all registered callbacks."""
        self._resource_callbacks.pop(uri, None)
        for name, session in self._sessions.items():
            try:
                await session.unsubscribe_resource(uri)
            except Exception as e:
                logger.debug(f"unsubscribe_resource failed on '{name}': {e}")

    async def _dispatch_resource_update(self, uri: str) -> None:
        """Fan a resource-update notification out to every registered callback.

        Failures are isolated per callback so one bad subscriber can't starve
        the rest or kill the session's receive loop.
        """
        for cb in list(self._resource_callbacks.get(uri, [])):
            try:
                res = cb(uri)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                logger.debug("resource callback raised", exc_info=True)

    def _make_message_handler(self):
        """Build a per-session message handler.

        Replaces mcp's default handler (which only checkpoints) so that
        ``notifications/resources/updated`` actually reaches subscribers via
        :meth:`_dispatch_resource_update`. Other notifications still fall
        through to the cooperative checkpoint so scheduling stays intact.
        """

        async def handler(message):
            try:
                if (
                    _ServerNotification is not None
                    and _ResourceUpdatedNotification is not None
                    and isinstance(message, _ServerNotification)
                    and isinstance(message.root, _ResourceUpdatedNotification)
                ):
                    await self._dispatch_resource_update(str(message.root.params.uri))
            except Exception:
                logger.debug("resource update dispatch failed", exc_info=True)
            if _anyio_lowlevel is not None:
                await _anyio_lowlevel.checkpoint()

        return handler

    def is_connected(self, name: str) -> bool:
        return name in self._initialized

    # ------------------------------------------------------------------ #
    # Health monitoring and reconnection
    # ------------------------------------------------------------------ #

    async def check_health(self, name: str) -> bool:
        """Probe whether a connected MCP server is still responsive.

        Uses ``list_tools()`` as a lightweight ping.  Returns ``True`` if
        the server replies within the timeout, ``False`` otherwise.  Never
        raises.
        """
        session = self._sessions.get(name)
        if not session:
            return False
        try:
            async with asyncio.timeout(_HEALTH_CHECK_TIMEOUT):
                await session.list_tools()
            self._last_health_check[name] = time.monotonic()
            return True
        except Exception:
            return False

    async def reconnect(self, name: str) -> bool:
        """Disconnect and re-connect a single MCP server using the stored config.

        Returns ``True`` on success, ``False`` on failure.  Updates the
        consecutive-failure counter used by the health monitor for backoff.
        """
        config = self._configs.get(name)
        if not config:
            logger.warning(f"Cannot reconnect '{name}': no stored config")
            return False

        # Tear down existing connection (best-effort, ignore errors)
        try:
            await self.disconnect(name)
        except Exception as e:
            logger.debug(f"Error during disconnect of '{name}' before reconnect: {e}")

        try:
            await self.connect(config)
            self._consecutive_failures[name] = 0
            return True
        except Exception as e:
            self._consecutive_failures[name] = (
                self._consecutive_failures.get(name, 0) + 1
            )
            logger.warning(
                f"Reconnect failed for '{name}' "
                f"(attempt {self._consecutive_failures[name]}): {e}"
            )
            return False

    def should_stop_retrying(self, name: str) -> bool:
        """Return True when the server has exceeded max reconnection attempts."""
        return self._consecutive_failures.get(name, 0) >= _MAX_RECONNECT_ATTEMPTS

    def get_server_status(self) -> dict[str, dict[str, Any]]:
        """Return a status snapshot for every known server (connected or not)."""
        names = set(self._configs.keys()) | set(self._sessions.keys())
        status: dict[str, dict[str, Any]] = {}
        for name in sorted(names):
            status[name] = {
                "connected": name in self._initialized,
                "tools": len([t for t in self._tools if t.server_name == name]),
                "failures": self._consecutive_failures.get(name, 0),
                "has_config": name in self._configs,
            }
        return status

    # ------------------------------------------------------------------ #
    # Session expiry 检测与带重试的工具调用
    # ------------------------------------------------------------------ #

    def _is_session_expired(self, error: Exception) -> bool:
        """检测 JSON-RPC -32001 session expired 错误。

        JSON-RPC 协议中 -32001 表示 session 已过期；HTTP 传输层
        则会返回 404。两者都意味着需要重新建立连接。
        """
        # JSON-RPC 错误码 -32001
        error_code = getattr(error, "code", None)
        if error_code == -32001:
            return True

        # 兜底：检查错误消息中的关键词
        error_str = str(error).lower()
        if "session" in error_str and "expired" in error_str:
            return True

        # HTTP 404 表示 session 不存在
        status_code = getattr(error, "status_code", None)
        if status_code == 404:
            return True

        return False

    async def call_tool_with_retry(
        self, name: str, args: dict[str, Any], max_errors: int = 3
    ) -> dict[str, Any]:
        """调用工具，检测 session expiry 自动重连。

        连续失败达到 max_errors 次或检测到 session expired 时触发重连。
        成功后重置错误计数。
        """
        info = self._tool_index.get(name)
        if not info:
            raise ValueError(
                f"MCP tool '{name}' not found. Available: {list(self._tool_index.keys())}"
            )

        server_name = info.server_name
        last_error: Exception | None = None

        for attempt in range(max_errors):
            try:
                result = await self.call_tool(name, args)
                # 成功时重置错误计数
                self._consecutive_errors.pop(server_name, None)
                return result
            except Exception as e:
                last_error = e
                self._consecutive_errors[server_name] = (
                    self._consecutive_errors.get(server_name, 0) + 1
                )

                # 检测到 session expired，立即重连
                if self._is_session_expired(e):
                    logger.warning(
                        f"Session expired for '{server_name}'，尝试重连"
                    )
                    reconnected = await self.reconnect(server_name)
                    if not reconnected:
                        raise
                    # 重连成功后重置计数，继续重试
                    self._consecutive_errors[server_name] = 0
                    continue

                # 错误计数达到上限，尝试重连
                if self._consecutive_errors[server_name] >= max_errors:
                    logger.warning(
                        f"工具 '{name}' 连续失败 "
                        f"{self._consecutive_errors[server_name]} 次，"
                        f"尝试重连 '{server_name}'"
                    )
                    reconnected = await self.reconnect(server_name)
                    if not reconnected:
                        raise
                    # 重连成功后重置计数，给一次重试机会
                    self._consecutive_errors[server_name] = 0

                # 指数退避后重试
                backoff = min(
                    _BACKOFF_BASE * (_BACKOFF_FACTOR ** attempt),
                    _BACKOFF_MAX,
                )
                await asyncio.sleep(backoff)

        assert last_error is not None
        raise last_error

    # ------------------------------------------------------------------ #
    # Memoized 连接
    # ------------------------------------------------------------------ #

    @staticmethod
    @lru_cache(maxsize=128)
    def _compute_cache_key(
        name: str,
        command: str,
        args: tuple[str, ...],
        env: tuple[tuple[str, str], ...],
    ) -> str:
        """根据配置计算缓存键，lru_cache 加速重复计算。"""
        return f"{name}|{command}|{args}|{env}"

    async def connect_memoized(self, config: MCPServerConfig) -> ClientSession:
        """带缓存的连接，避免重复握手。

        按 (name, command, args, env) 的 hash 做 cache key，
        如果已有相同配置的活跃连接则直接返回 session。

        ponytail: cache key 不含 transport/url，所以 SSE server 只靠 name
        区分。正常使用下 name 唯一所以安全；若同名 SSE server 换了 URL，
        升级路径是把 config.transport / config.url 也并入 _compute_cache_key。
        """
        args_tuple = tuple(config.args)
        env_tuple = tuple(sorted((config.env or {}).items()))
        cache_key = self._compute_cache_key(
            config.name, config.command, args_tuple, env_tuple
        )

        # 缓存命中：已有相同配置的活跃连接
        cached_name = self._connection_cache.get(cache_key)
        if cached_name and cached_name in self._sessions:
            logger.debug(f"复用已有连接 '{config.name}'（缓存命中）")
            return self._sessions[cached_name]

        # 缓存未命中：建立新连接
        await self.connect(config)
        self._connection_cache[cache_key] = config.name
        return self._sessions[config.name]

    # ------------------------------------------------------------------ #
    # 并发批处理连接
    # ------------------------------------------------------------------ #

    async def connect_batch(
        self, configs: list[MCPServerConfig], concurrency: int = 4
    ) -> dict[str, list]:
        """并发连接多个服务器。

        本地 stdio 服务器建议并发度 4，远程 HTTP/SSE 服务器建议并发度 2。
        返回 {"success": [names], "failed": [(name, error)]}。
        """
        semaphore = asyncio.Semaphore(concurrency)
        success: list[str] = []
        failed: list[tuple[str, str]] = []

        async def _connect_one(cfg: MCPServerConfig) -> None:
            async with semaphore:
                try:
                    await self.connect(cfg)
                    success.append(cfg.name)
                except Exception as e:
                    failed.append((cfg.name, str(e)))

        await asyncio.gather(*[_connect_one(c) for c in configs])
        return {"success": success, "failed": failed}

    # ------------------------------------------------------------------ #
    # 进程清理升级序列
    # ------------------------------------------------------------------ #

    @staticmethod
    def _extract_process(client: Any) -> Any:
        """尝试从 stdio_client 上下文管理器中提取底层进程对象。

        stdio_client 由 contextlib.asynccontextmanager 装饰，其内部
        异步生成器挂起在 yield 处时，gi_frame.f_locals 中持有 process 变量。
        如果无法提取则返回 None（不影响后续 __aexit__ 的正常清理）。
        """
        if client is None:
            return None
        # 不同 Python 版本属性名可能不同，都试一下
        gen = getattr(client, "gen", None) or getattr(client, "_agen", None)
        if gen is None:
            return None
        frame = getattr(gen, "gi_frame", None)
        if frame is None:
            return None
        return frame.f_locals.get("process")

    async def _terminate_process(self, proc: Any, timeout: float = 5.0) -> None:
        """SIGINT → SIGTERM → SIGKILL 升级序列清理进程。

        Windows 上没有 SIGINT/SIGTERM 信号概念，使用 terminate() → kill() 两步。
        Unix 上按 SIGINT → SIGTERM → SIGKILL 逐步升级，每步等待 timeout 秒。
        """
        if proc is None:
            return

        # 进程已退出则无需处理
        returncode = getattr(proc, "returncode", None)
        if returncode is not None:
            return

        system = platform.system()

        try:
            if system == "Windows":
                # Windows: terminate() (TerminateProcess) → kill() 升级
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timeout)
                    return
                except asyncio.TimeoutError:
                    pass

                # 升级到 kill
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timeout)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass
            else:
                # Unix: SIGINT → SIGTERM → SIGKILL
                for sig in (signal.SIGINT, signal.SIGTERM):
                    with contextlib.suppress(ProcessLookupError, PermissionError):
                        proc.send_signal(sig)
                    try:
                        await asyncio.wait_for(proc.wait(), timeout=timeout)
                        return
                    except asyncio.TimeoutError:
                        continue
                    except ProcessLookupError:
                        return

                # 最终升级到 SIGKILL
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    proc.kill()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=timeout)
                except (asyncio.TimeoutError, ProcessLookupError):
                    pass
        except Exception as e:
            logger.debug(f"进程清理异常: {e}")

    def __del__(self):
        # Best-effort cleanup
        if self._sessions:
            try:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self.disconnect_all())
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.disconnect_all())
            except Exception:
                pass


# Module-level default manager for tests and CLI integration
mcp_manager = MCPClientManager()
