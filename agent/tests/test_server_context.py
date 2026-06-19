"""Tests for server-scoped context."""

from __future__ import annotations

from huginn.server_context import (
    ServerContext,
    create_server_context,
    set_server_context,
)


class TestServerContext:
    def test_create_server_context(self):
        ctx = create_server_context()
        assert isinstance(ctx, ServerContext)
        assert ctx.agent_factory is not None
        assert ctx.memory_manager is not None
        assert ctx.orchestrator is not None

    def test_set_server_context(self):
        ctx = create_server_context()
        set_server_context(ctx)
        from huginn.server_context import get_server_context

        assert get_server_context() is ctx
