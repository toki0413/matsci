"""Tests for MCP resilience: health check, reconnect, idempotent connect, server status."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytest.importorskip("mcp", reason="MCP SDK not installed (pip install mcp)")

from huginn.mcp_client import (
    MCPClientManager,
    MCPServerConfig,
    MCPToolInfo,
    _BACKOFF_BASE,
    _BACKOFF_FACTOR,
    _BACKOFF_MAX,
    _HEALTH_CHECK_INTERVAL,
    _MAX_RECONNECT_ATTEMPTS,
)


# ── Helpers ──────────────────────────────────────────────────────


def _make_manager() -> MCPClientManager:
    """Create a fresh manager without any connections."""
    return MCPClientManager()


def _fake_tool(name: str, server: str = "srv") -> MCPToolInfo:
    return MCPToolInfo(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
        server_name=server,
    )


# ── Idempotent connect ───────────────────────────────────────────


class TestIdempotentConnect:
    @pytest.mark.asyncio
    async def test_connect_stores_config(self):
        mgr = _make_manager()
        cfg = MCPServerConfig(name="test", command="echo", args=["hi"])

        # Patch the low-level connect machinery so we don't spawn a process
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=MagicMock(tools=[])
        )
        mock_client = AsyncMock()

        with (
            patch("huginn.mcp_client.stdio_client", return_value=mock_client),
            patch.object(mock_client, "__aenter__", return_value=(AsyncMock(), AsyncMock())),
            patch.object(mock_client, "__aexit__", new_callable=AsyncMock),
            patch("huginn.mcp_client.ClientSession", return_value=mock_session),
            patch.object(mock_session, "__aenter__", return_value=mock_session),
            patch.object(mock_session, "__aexit__", new_callable=AsyncMock),
        ):
            await mgr.connect(cfg)

        assert "test" in mgr._configs
        assert mgr._configs["test"] is cfg
        assert mgr._consecutive_failures.get("test") == 0

    @pytest.mark.asyncio
    async def test_connect_clears_stale_tools(self):
        """Re-connecting after a stale entry should clear old tools first."""
        mgr = _make_manager()
        # Simulate leftover tools from a previous connection
        stale = _fake_tool("old_tool", "test")
        mgr._tools = [stale]
        mgr._tool_index = {"old_tool": stale}

        cfg = MCPServerConfig(name="test", command="echo", args=["hi"])
        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(
            return_value=MagicMock(tools=[])
        )
        mock_client = AsyncMock()

        with (
            patch("huginn.mcp_client.stdio_client", return_value=mock_client),
            patch.object(mock_client, "__aenter__", return_value=(AsyncMock(), AsyncMock())),
            patch.object(mock_client, "__aexit__", new_callable=AsyncMock),
            patch("huginn.mcp_client.ClientSession", return_value=mock_session),
            patch.object(mock_session, "__aenter__", return_value=mock_session),
            patch.object(mock_session, "__aexit__", new_callable=AsyncMock),
        ):
            await mgr.connect(cfg)

        # Stale tool should be gone
        assert all(t.name != "old_tool" for t in mgr._tools)
        assert "old_tool" not in mgr._tool_index

    @pytest.mark.asyncio
    async def test_connect_refuses_duplicate(self):
        """Connecting to an already-connected server should be a no-op."""
        mgr = _make_manager()
        mgr._sessions["test"] = MagicMock()  # simulate connected

        cfg = MCPServerConfig(name="test", command="echo", args=[])
        # Should return early without raising
        await mgr.connect(cfg)
        # Config should NOT be stored (early return before that line)
        # Actually with our change, config is stored before the guard.
        # Let me re-check... The guard is `if config.name in self._sessions: return`
        # and config storage is after. So config should NOT be stored.
        assert "test" not in mgr._configs


# ── Health check ──────────────────────────────────────────────────


class TestHealthCheck:
    @pytest.mark.asyncio
    async def test_check_health_returns_true_when_ok(self):
        mgr = _make_manager()
        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        mgr._sessions["srv"] = mock_session

        assert await mgr.check_health("srv") is True

    @pytest.mark.asyncio
    async def test_check_health_returns_false_on_failure(self):
        mgr = _make_manager()
        mock_session = AsyncMock()
        mock_session.list_tools = AsyncMock(side_effect=ConnectionError("dead"))
        mgr._sessions["srv"] = mock_session

        assert await mgr.check_health("srv") is False

    @pytest.mark.asyncio
    async def test_check_health_returns_false_when_not_connected(self):
        mgr = _make_manager()
        assert await mgr.check_health("nonexistent") is False


# ── Reconnect ─────────────────────────────────────────────────────


class TestReconnect:
    @pytest.mark.asyncio
    async def test_reconnect_without_config_returns_false(self):
        mgr = _make_manager()
        assert await mgr.reconnect("unknown") is False

    @pytest.mark.asyncio
    async def test_reconnect_increments_failures_on_error(self):
        mgr = _make_manager()
        cfg = MCPServerConfig(name="srv", command="false", args=[])
        mgr._configs["srv"] = cfg
        mgr._consecutive_failures["srv"] = 2

        # disconnect will be a no-op (nothing in _sessions)
        # connect will fail because 'false' is not a valid MCP server
        result = await mgr.reconnect("srv")
        assert result is False
        assert mgr._consecutive_failures["srv"] == 3

    @pytest.mark.asyncio
    async def test_reconnect_resets_failures_on_success(self):
        mgr = _make_manager()
        cfg = MCPServerConfig(name="srv", command="echo", args=["hi"])
        mgr._configs["srv"] = cfg
        mgr._consecutive_failures["srv"] = 3

        mock_session = AsyncMock()
        mock_session.initialize = AsyncMock()
        mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=[]))
        mock_client = AsyncMock()

        with (
            patch("huginn.mcp_client.stdio_client", return_value=mock_client),
            patch.object(mock_client, "__aenter__", return_value=(AsyncMock(), AsyncMock())),
            patch.object(mock_client, "__aexit__", new_callable=AsyncMock),
            patch("huginn.mcp_client.ClientSession", return_value=mock_session),
            patch.object(mock_session, "__aenter__", return_value=mock_session),
            patch.object(mock_session, "__aexit__", new_callable=AsyncMock),
        ):
            result = await mgr.reconnect("srv")

        assert result is True
        assert mgr._consecutive_failures["srv"] == 0


# ── Server status ─────────────────────────────────────────────────


class TestServerStatus:
    def test_empty_status(self):
        mgr = _make_manager()
        assert mgr.get_server_status() == {}

    def test_status_with_config(self):
        mgr = _make_manager()
        mgr._configs["srv"] = MCPServerConfig(name="srv", command="echo", args=[])
        status = mgr.get_server_status()
        assert "srv" in status
        assert status["srv"]["connected"] is False
        assert status["srv"]["has_config"] is True
        assert status["srv"]["tools"] == 0
        assert status["srv"]["failures"] == 0

    def test_status_with_connected_server(self):
        mgr = _make_manager()
        mgr._configs["srv"] = MCPServerConfig(name="srv", command="echo", args=[])
        mgr._sessions["srv"] = MagicMock()
        mgr._initialized.add("srv")
        mgr._tools = [_fake_tool("t1", "srv"), _fake_tool("t2", "srv")]
        mgr._consecutive_failures["srv"] = 0

        status = mgr.get_server_status()
        assert status["srv"]["connected"] is True
        assert status["srv"]["tools"] == 2

    def test_should_stop_retrying(self):
        mgr = _make_manager()
        assert mgr.should_stop_retrying("srv") is False
        mgr._consecutive_failures["srv"] = _MAX_RECONNECT_ATTEMPTS
        assert mgr.should_stop_retrying("srv") is True
        mgr._consecutive_failures["srv"] = _MAX_RECONNECT_ATTEMPTS + 1
        assert mgr.should_stop_retrying("srv") is True


# ── Backoff constants ─────────────────────────────────────────────


class TestBackoffConstants:
    def test_defaults_are_sane(self):
        assert _BACKOFF_BASE >= 0.5
        assert _BACKOFF_FACTOR >= 1.5
        assert _BACKOFF_MAX >= 10.0
        assert _MAX_RECONNECT_ATTEMPTS >= 3
        assert _HEALTH_CHECK_INTERVAL >= 10.0

    def test_backoff_sequence(self):
        """Verify the exponential progression caps correctly."""
        delays = [
            min(_BACKOFF_BASE * (_BACKOFF_FACTOR ** i), _BACKOFF_MAX)
            for i in range(10)
        ]
        # First delay should be the base
        assert delays[0] == pytest.approx(_BACKOFF_BASE)
        # Should monotonically increase then cap
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1]
        # Eventually hits the cap
        assert delays[-1] == _BACKOFF_MAX


# ── register_mcp_tools dedup ──────────────────────────────────────


class TestRegisterMCPToolsDedup:
    def test_register_with_server_name_filter(self):
        from huginn.tools.mcp_adapter import register_mcp_tools
        from huginn.tools.registry import ToolRegistry

        mgr = _make_manager()
        mgr._tools = [
            _fake_tool("tool_a", "server1"),
            _fake_tool("tool_b", "server2"),
        ]

        # Save current registry state
        saved = dict(ToolRegistry._tools)

        # Remove only our test tool names to avoid cross-test pollution
        for name in ("tool_a", "tool_b"):
            ToolRegistry._tools.pop(name, None)

        try:
            registered = register_mcp_tools(mgr, server_name="server1")
            assert len(registered) == 1
            assert registered[0].name == "tool_a"
            assert ToolRegistry.get("tool_a") is not None
            # tool_b should NOT have been registered
            assert ToolRegistry.get("tool_b") is None
        finally:
            # Restore: remove our test tools, put back saved state
            for name in ("tool_a", "tool_b"):
                ToolRegistry._tools.pop(name, None)
            ToolRegistry._tools.update(saved)

    def test_register_replaces_existing(self):
        from huginn.tools.mcp_adapter import register_mcp_tools
        from huginn.tools.registry import ToolRegistry

        mgr = _make_manager()
        mgr._tools = [_fake_tool("tool_a", "srv")]

        saved = dict(ToolRegistry._tools)
        ToolRegistry._tools.pop("tool_a", None)

        try:
            # Register once
            first = register_mcp_tools(mgr)
            first_adapter = first[0]

            # Register again (simulates reconnect) — should replace, not duplicate
            second = register_mcp_tools(mgr)
            assert len(second) == 1
            assert ToolRegistry.get("tool_a") is second[0]
            assert ToolRegistry.get("tool_a") is not first_adapter
        finally:
            ToolRegistry._tools.pop("tool_a", None)
            ToolRegistry._tools.update(saved)
