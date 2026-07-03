"""Tests for ContainerExecutor."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from huginn.security.container_executor import ContainerExecutor
from huginn.security.sandbox import SandboxConfig


class TestContainerExecutor:
    def test_unsupported_runtime(self):
        with pytest.raises(ValueError):
            ContainerExecutor(runtime="rocket", image="foo")

    def test_dry_run(self):
        exe = ContainerExecutor(
            runtime="docker",
            image="huginn/sandbox",
            sandbox_config=SandboxConfig(dry_run=True),
        )
        result = exe.run(["python", "-c", "print('hello')"], cwd=".")
        assert result.dry_run is True
        assert result.success is True
        assert "docker" in result.stdout
        assert "python" in result.stdout

    def test_runtime_not_in_path(self, monkeypatch):
        monkeypatch.setattr("shutil.which", lambda _name: None)
        exe = ContainerExecutor(runtime="docker", image="huginn/sandbox")
        result = exe.run(["python", "-c", "print('hello')"], cwd=".")
        assert result.success is False
        assert "not found in PATH" in result.stderr

    @pytest.mark.skipif(sys.platform == "win32", reason="apptainer args differ")
    def test_build_command_docker(self):
        exe = ContainerExecutor(runtime="docker", image="huginn/sandbox")
        cmd = exe._build_command(
            "docker", ["python", "-c", "print(1)"], Path("/tmp/ws"), {"FOO": "bar"}
        )
        assert cmd[0] == "docker"
        assert "run" in cmd
        assert "--rm" in cmd
        assert "/tmp/ws:/huginn_work" in cmd
        assert "FOO=bar" in cmd
        assert cmd[-3:] == ["python", "-c", "print(1)"]

    @pytest.mark.skipif(sys.platform == "win32", reason="apptainer args differ")
    def test_build_command_apptainer(self):
        exe = ContainerExecutor(runtime="apptainer", image="sandbox.sif")
        cmd = exe._build_command(
            "apptainer", ["python", "-c", "print(1)"], Path("/tmp/ws"), {"FOO": "bar"}
        )
        assert cmd[0] == "apptainer"
        assert "exec" in cmd
        assert "/tmp/ws:/huginn_work" in cmd
        assert "sandbox.sif" in cmd
