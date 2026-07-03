"""Tests for execution backend selection."""

from __future__ import annotations

import pytest

from huginn.security import SandboxError, get_executor
from huginn.security.sandbox import SandboxExecutor


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
