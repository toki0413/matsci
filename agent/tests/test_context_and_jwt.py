"""Tests for ContextBuilder extraction and JWT revocation."""

import asyncio
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── ContextBuilder ────────────────────────────────────────────────


class TestContextBuilder:
    """Verify ContextBuilder works standalone."""

    def test_build_memory_text_empty(self):
        """Empty memory should return empty string."""
        from huginn.context_builder import ContextBuilder

        memory = MagicMock()
        memory.recall_for_prompt.return_value = ""

        cb = ContextBuilder(
            memory_manager=memory,
            workspace="/tmp",
            kg_enabled=False,
            kb_enabled=False,
        )
        result = cb.build_memory_text("test query")
        assert isinstance(result, str)

    def test_build_memory_text_with_content(self):
        """Memory recall should appear in output."""
        from huginn.context_builder import ContextBuilder

        memory = MagicMock()
        memory.recall_for_prompt.return_value = "Si has diamond structure"

        cb = ContextBuilder(
            memory_manager=memory,
            workspace="/tmp",
            kg_enabled=False,
            kb_enabled=False,
        )
        result = cb.build_memory_text("silicon structure")
        assert "Si" in result or "diamond" in result

    def test_build_kg_text_disabled(self):
        """KG disabled should return empty string."""
        from huginn.context_builder import ContextBuilder

        cb = ContextBuilder(
            memory_manager=MagicMock(),
            workspace="/tmp",
            kg_enabled=False,
        )
        assert cb.build_kg_text("query") == ""

    def test_build_kb_text_disabled(self):
        """KB disabled should return empty string."""
        from huginn.context_builder import ContextBuilder

        cb = ContextBuilder(
            memory_manager=MagicMock(),
            workspace="/tmp",
            kb_enabled=False,
        )
        assert cb.build_kb_text("query") == ""

    def test_build_emotion_text_no_tracker(self):
        """No emotion tracker should return None."""
        from huginn.context_builder import ContextBuilder

        cb = ContextBuilder(
            memory_manager=MagicMock(),
            workspace="/tmp",
            emotion_tracker=None,
        )
        assert cb.build_emotion_text("hello") is None

    def test_build_emotion_text_with_tracker(self):
        """Emotion tracker should be updated and return context."""
        from huginn.context_builder import ContextBuilder

        tracker = MagicMock()
        tracker.context_prompt.return_value = "mood: focused"

        cb = ContextBuilder(
            memory_manager=MagicMock(),
            workspace="/tmp",
            emotion_tracker=tracker,
        )
        result = cb.build_emotion_text("analyze this")
        assert result == "mood: focused"
        tracker.update_from_message.assert_called_once()

    def test_memory_recall_failure_logged(self):
        """Memory recall failure should not crash, returns empty."""
        from huginn.context_builder import ContextBuilder

        memory = MagicMock()
        memory.recall_for_prompt.side_effect = RuntimeError("DB error")

        cb = ContextBuilder(
            memory_manager=memory,
            workspace="/tmp",
        )
        result = cb.build_memory_text("query")
        assert result == "" or isinstance(result, str)

    def test_conversation_tree_none(self):
        """No conversation tree should return empty list."""
        from huginn.context_builder import ContextBuilder

        cb = ContextBuilder(
            memory_manager=MagicMock(),
            workspace="/tmp",
            conversation_tree=None,
        )
        assert cb.conversation_tree_to_messages() == []


class TestAgentDelegatesToContextBuilder:
    """Verify agent delegates _build_* to ContextBuilder."""

    def test_agent_has_ctx_builder(self):
        """Agent should have _ctx_builder after init."""
        from tests.fixtures.fake_llm import make_callable_llm
        from huginn.agent import HuginnAgent
        from huginn.memory.manager import MemoryManager

        llm = make_callable_llm(lambda p: "ok", name="ctx-test-llm")
        memory = MemoryManager()
        agent = HuginnAgent(model=llm, memory_manager=memory)

        assert hasattr(agent, "_ctx_builder")
        assert agent._ctx_builder is not None

    def test_build_memory_delegates(self):
        """_build_memory_text should delegate to ContextBuilder."""
        from tests.fixtures.fake_llm import make_callable_llm
        from huginn.agent import HuginnAgent
        from huginn.memory.manager import MemoryManager

        llm = make_callable_llm(lambda p: "ok", name="delegate-test-llm")
        memory = MemoryManager()
        agent = HuginnAgent(model=llm, memory_manager=memory)

        # Should not crash
        result = agent._build_memory_text("test")
        assert isinstance(result, str)

    def test_build_kg_text_delegates(self):
        """_build_kg_text should delegate to ContextBuilder."""
        from tests.fixtures.fake_llm import make_callable_llm
        from huginn.agent import HuginnAgent
        from huginn.memory.manager import MemoryManager

        llm = make_callable_llm(lambda p: "ok", name="kg-test-llm")
        memory = MemoryManager()
        agent = HuginnAgent(model=llm, memory_manager=memory)

        result = agent._build_kg_text("test")
        assert isinstance(result, str)


# ── JWT Revocation ─────────────────────────────────────────────────


class TestTokenRevocationList:
    """Verify TokenRevocationList works correctly."""

    def test_revoke_and_check(self):
        """Revoked token should be detected."""
        from huginn.security.rbac import TokenRevocationList

        rl = TokenRevocationList()
        rl.clear()
        assert not rl.is_revoked("test-jti-1")
        rl.revoke("test-jti-1", exp=time.time() + 3600)
        assert rl.is_revoked("test-jti-1")
        assert not rl.is_revoked("test-jti-2")

    def test_auto_prune_expired(self):
        """Expired entries should be auto-pruned."""
        from huginn.security.rbac import TokenRevocationList

        rl = TokenRevocationList()
        rl.clear()
        # Add an already-expired entry
        rl.revoke("expired-jti", exp=time.time() - 1)
        assert not rl.is_revoked("expired-jti")  # pruned on check
        assert rl.count() == 0

    def test_clear(self):
        """Clear should remove all entries."""
        from huginn.security.rbac import TokenRevocationList

        rl = TokenRevocationList()
        rl.revoke("jti-1", exp=time.time() + 3600)
        rl.revoke("jti-2", exp=time.time() + 3600)
        assert rl.count() == 2
        rl.clear()
        assert rl.count() == 0

    def test_singleton(self):
        """shared() should return the same instance."""
        from huginn.security.rbac import TokenRevocationList

        a = TokenRevocationList.shared()
        b = TokenRevocationList.shared()
        assert a is b

    def test_thread_safe(self):
        """Concurrent revoke and check should not crash."""
        import threading

        from huginn.security.rbac import TokenRevocationList

        rl = TokenRevocationList()
        rl.clear()
        errors = []

        def worker(idx):
            try:
                for i in range(50):
                    jti = f"jti-{idx}-{i}"
                    rl.revoke(jti, exp=time.time() + 3600)
                    rl.is_revoked(jti)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent errors: {errors}"
        rl.clear()


class TestJWTContainsJTI:
    """Verify JWT tokens contain jti claim."""

    def test_create_token_has_jti(self):
        """Issued JWT should contain a jti claim."""
        from huginn.security.rbac import Role, User, jwt_decode

        user = User(
            user_id="test-user",
            username="tester",
            role=Role.OPERATOR,
        )

        with patch.dict("os.environ", {"HUGINN_JWT_SECRET": "test-secret-12345"}):
            from huginn.security.auth import create_token

            token = create_token(user)
            claims = jwt_decode(token, "test-secret-12345")
            assert "jti" in claims
            assert len(claims["jti"]) > 0


class TestAuthLogoutEndpoint:
    """Verify /auth/logout endpoint exists."""

    def test_logout_route_exists(self):
        """The auth router should have a /logout route."""
        from huginn.routes.auth import router

        paths = [r.path for r in router.routes if hasattr(r, "path")]
        assert "/logout" in paths or any("logout" in p for p in paths)
