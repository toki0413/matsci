"""Tests for execution backend selection."""

from __future__ import annotations

import pytest

from huginn.security import SandboxError, get_executor
from huginn.security.container_executor import ContainerExecutor
from huginn.security.execution import build_executor
from huginn.security.sandbox import SandboxExecutor
from huginn.tools.defaults import ToolMetadata


class TestGetExecutor:
    def test_local_fallback_when_allowed(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
        executor = get_executor()
        assert isinstance(executor, SandboxExecutor)

    def test_raises_when_no_container_and_no_local(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "0")
        with pytest.raises(SandboxError):
            get_executor()

    def test_raises_when_container_runtime_missing(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "docker")
        monkeypatch.setenv("HUGINN_CONTAINER_IMAGE", "huginn:latest")
        monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "0")
        # Docker is pre-installed on GitHub Actions runners — mock it as
        # missing so the test is environment-independent.
        monkeypatch.setattr("shutil.which", lambda _name: None)
        with pytest.raises(SandboxError, match="Container runtime 'docker' not found"):
            get_executor()


class TestBuildExecutor:
    """T-BCSE-09: 工具级沙箱后端绑定 (sandbox_hint)."""

    def test_host_hint_returns_sandbox(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        ex = build_executor(sandbox_hint="host")
        assert isinstance(ex, SandboxExecutor)

    def test_any_hint_uses_env(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
        ex = build_executor(tool_metadata=ToolMetadata(sandbox_hint="any"))
        assert isinstance(ex, SandboxExecutor)

    def test_container_hint_requires_runtime(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        with pytest.raises(SandboxError, match="container execution"):
            build_executor(sandbox_hint="container")

    def test_container_hint_returns_container(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "docker")
        monkeypatch.setenv("HUGINN_CONTAINER_IMAGE", "huginn@sha256:" + "a" * 64)
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        ex = build_executor(sandbox_hint="container")
        assert isinstance(ex, ContainerExecutor)

    def test_paranoid_refuses_without_container(self, monkeypatch):
        """paranoid 缺容器 → 拒绝而非静默回退 (T-BCSE-09 ACC)."""
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
        with pytest.raises(SandboxError, match="Refusing to run in a weaker sandbox"):
            build_executor(sandbox_hint="paranoid")

    def test_paranoid_returns_hardened_container(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "docker")
        monkeypatch.setenv("HUGINN_CONTAINER_IMAGE", "huginn@sha256:" + "b" * 64)
        monkeypatch.setattr("shutil.which", lambda _name: "/usr/bin/docker")
        ex = build_executor(sandbox_hint="paranoid")
        assert isinstance(ex, ContainerExecutor)
        assert ex.security_config.require_digest is True
        assert ex.security_config.network_none is True
        assert ex.security_config.no_new_privileges is True
        assert ex.security_config.drop_all_capabilities is True

    def test_destructive_defaults_to_paranoid(self, monkeypatch):
        """is_destructive 工具默认 paranoid → 缺容器时拒绝."""
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        meta = ToolMetadata(is_destructive=True, sandbox_hint="any")
        with pytest.raises(SandboxError, match="Refusing to run in a weaker sandbox"):
            build_executor(tool_metadata=meta)

    def test_metadata_hint_used(self, monkeypatch):
        monkeypatch.setenv("HUGINN_CONTAINER_RUNTIME", "none")
        monkeypatch.setenv("HUGINN_ALLOW_LOCAL_BASH", "1")
        ex = build_executor(tool_metadata=ToolMetadata(sandbox_hint="host"))
        assert isinstance(ex, SandboxExecutor)
