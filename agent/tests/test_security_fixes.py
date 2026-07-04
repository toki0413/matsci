"""Tests for security fixes — path traversal, WS auth, cache thread safety."""

import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ── Checkpoint path traversal ──────────────────────────────────────


class TestCheckpointPathValidation:
    """Verify _validate_workspace_path rejects paths outside workspace."""

    def test_rejects_absolute_path_outside_workspace(self, tmp_path):
        """Path like /etc should be rejected."""
        from huginn.routes.checkpoints import _validate_workspace_path

        with patch("huginn.server_core.get_context") as mock_ctx:
            mock_ctx.return_value.config.workspace = str(tmp_path)
            from fastapi import HTTPException

            with pytest.raises(HTTPException) as exc_info:
                _validate_workspace_path("/etc")
            assert exc_info.value.status_code == 403

    def test_rejects_parent_traversal(self, tmp_path):
        """Path like ../../../etc should be rejected."""
        from huginn.routes.checkpoints import _validate_workspace_path

        with patch("huginn.server_core.get_context") as mock_ctx:
            mock_ctx.return_value.config.workspace = str(tmp_path)
            from fastapi import HTTPException

            with pytest.raises(HTTPException):
                _validate_workspace_path("../../../etc/passwd")

    def test_accepts_path_within_workspace(self, tmp_path):
        """Path inside workspace should be accepted."""
        from huginn.routes.checkpoints import _validate_workspace_path

        sub = tmp_path / "subdir"
        sub.mkdir()

        with patch("huginn.server_core.get_context") as mock_ctx:
            mock_ctx.return_value.config.workspace = str(tmp_path)
            result = _validate_workspace_path(str(sub))
            assert str(tmp_path) in str(result)

    def test_accepts_dot_path(self, tmp_path):
        """Dot (current dir) should resolve to workspace."""
        import os
        from huginn.routes.checkpoints import _validate_workspace_path

        # Change cwd to tmp_path so "." resolves within workspace
        old_cwd = os.getcwd()
        os.chdir(str(tmp_path))
        try:
            with patch("huginn.server_core.get_context") as mock_ctx:
                mock_ctx.return_value.config.workspace = str(tmp_path)
                result = _validate_workspace_path(".")
                assert tmp_path.resolve() == result or str(tmp_path) in str(result)
        finally:
            os.chdir(old_cwd)


# ── TimedLRUCache thread safety ───────────────────────────────────


class TestCacheThreadSafety:
    """Verify TimedLRUCache is thread-safe."""

    def test_concurrent_set_get(self):
        """Concurrent set/get should not corrupt the cache."""
        from huginn.utils.cache import TimedLRUCache

        cache = TimedLRUCache(max_size=100, ttl=60.0)
        errors: list[Exception] = []

        def worker(idx: int):
            try:
                for i in range(50):
                    key = f"key-{idx}-{i}"
                    cache.set(key, f"value-{idx}-{i}")
                    cache.get(key)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent access caused errors: {errors}"
        # Cache should have at most max_size entries
        assert len(cache) <= 100

    def test_concurrent_clear(self):
        """Concurrent clear + set should not crash."""
        from huginn.utils.cache import TimedLRUCache

        cache = TimedLRUCache(max_size=50, ttl=60.0)
        errors: list[Exception] = []

        def setter():
            try:
                for i in range(100):
                    cache.set(f"k{i}", f"v{i}")
            except Exception as e:
                errors.append(e)

        def clearer():
            try:
                for _ in range(50):
                    cache.clear()
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=setter)
        t2 = threading.Thread(target=clearer)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert not errors, f"Concurrent clear caused errors: {errors}"


# ── HPC routes require admin ──────────────────────────────────────


class TestHPCAdminAuth:
    """Verify HPC routes require admin key."""

    def test_hpc_router_has_admin_dependency(self):
        """The HPC router should have require_admin_key dependency."""
        from huginn.routes.hpc import router

        # Check that the router has dependencies
        assert router.dependencies is not None
        # The dependency should include require_admin_key
        dep_str = str(router.dependencies)
        assert "require_admin_key" in dep_str or "admin" in dep_str.lower()

    def test_hpc_router_not_public(self):
        """HPC routes should not be in public paths."""
        from huginn.security.auth import _PUBLIC_PATHS

        for path in _PUBLIC_PATHS:
            assert not path.startswith("/hpc"), f"HPC path {path} should not be public"


# ── Rate limiting default ────────────────────────────────────────


class TestRateLimitDefault:
    """Verify rate limiting is enabled by default."""

    def test_default_rate_limit_not_zero(self):
        """Default _RATE_LIMIT should not be 0."""
        from huginn.server import _RATE_LIMIT

        # In test mode it might be overridden, but the default in source
        # should be 120, not 0
        import huginn.server as srv
        import inspect

        source = inspect.getsource(srv)
        # Check the default value in the source code
        assert '"120"' in source or "'120'" in source, \
            "Default rate limit should be 120, not 0"


# ── Dead code removed ─────────────────────────────────────────────


class TestDeadCodeRemoved:
    """Verify dead code has been removed."""

    def test_retry_llm_call_removed(self):
        """retry_llm_call should no longer exist in agent.py."""
        import huginn.agent as agent_mod

        assert not hasattr(agent_mod, "retry_llm_call"), \
            "retry_llm_call should have been removed"

    def test_create_langchain_model_wrapper_removed(self):
        """_create_langchain_model wrapper should be removed from agent.py."""
        import huginn.agent as agent_mod

        assert not hasattr(agent_mod, "_create_langchain_model"), \
            "_create_langchain_model wrapper should have been removed"

    def test_llm_py_uses_registry_directly(self):
        """llm.py should import from models.registry, not agent.py."""
        import huginn.llm as llm_mod
        import inspect

        source = inspect.getsource(llm_mod)
        assert "from huginn.models.registry import" in source
        assert "from huginn.agent import" not in source


# ── Dockerfile healthcheck ────────────────────────────────────────


class TestDockerfileHealthcheck:
    """Verify Dockerfile has HEALTHCHECK directive."""

    def test_dockerfile_has_healthcheck(self):
        from pathlib import Path

        dockerfile = Path(__file__).resolve().parent.parent / "Dockerfile"
        content = dockerfile.read_text()
        assert "HEALTHCHECK" in content.upper()

    def test_compose_has_healthcheck(self):
        from pathlib import Path

        compose = Path(__file__).resolve().parent.parent / "docker-compose.yml"
        content = compose.read_text()
        assert "healthcheck" in content.lower()
