"""Mock tests for Lean and HPC modules.

Lean 4 requires `lake` executable; HPC requires SSH/paramiko.
Both are mocked here to exercise all code paths.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from huginn.hpc.client import HPCConfig, HPCClient, JobStatus, _sanitize_job_name, _validate_path_component
from huginn.hpc.resource_selector import ResourceSelector
from huginn.lean.interface import LeanInterface
from huginn.lean.sympy_to_lean import SymPyToLean
from huginn.tools.lean_tool import LeanTool, LeanToolInput
from huginn.types import ToolContext

CTX = ToolContext(session_id="test", workspace=".")


# ── HPC ──
class TestHPCClient:
    def test_sanitize_job_name(self):
        assert _sanitize_job_name("test-job_1.txt") == "test-job_1.txt"
        assert _sanitize_job_name("; rm -rf /") == "__rm_-rf__"
        with pytest.raises(ValueError):
            _sanitize_job_name(";")

    def test_validate_path_component(self):
        _validate_path_component("/home/user/job")
        with pytest.raises(ValueError):
            _validate_path_component("/home; rm -rf /")

    def test_hpc_config_defaults(self):
        cfg = HPCConfig(host="hpc.example.com", username="user")
        assert cfg.scheduler == "slurm"
        assert cfg.port == 22

    def test_connect_paramiko(self):
        cfg = HPCConfig(host="hpc.example.com", username="user", key_path="/fake/key")
        client = HPCClient(cfg)
        with patch("paramiko.SSHClient") as mock_ssh:
            instance = MagicMock()
            mock_ssh.return_value = instance
            instance.get_host_keys.return_value.keys.return_value = ["host1"]
            client.connect(timeout=1)
            instance.connect.assert_called_once()

    def test_job_status_str(self):
        js = JobStatus(job_id="123", state="RUNNING", exit_code=0)
        assert js.state == "RUNNING"


class TestResourceSelector:
    def test_selector_init(self):
        cfg = HPCConfig(host="hpc.example.com", username="user")
        sel = ResourceSelector(cfg)
        assert sel is not None


# ── Lean Interface ──
class TestLeanInterface:
    def test_init_no_project(self, tmp_path: Path):
        with pytest.raises(ValueError):
            LeanInterface(tmp_path)

    def test_init_with_project(self, tmp_path: Path):
        project = tmp_path / "HuginnLean"
        project.mkdir()
        (project / "lakefile.toml").write_text("name = 'HuginnLean'\n")
        with patch("shutil.which", return_value="/fake/lake"):
            li = LeanInterface(project)
        assert li.project_path == project.resolve()

    def test_build_no_lake(self, tmp_path: Path):
        project = tmp_path / "HuginnLean"
        project.mkdir()
        (project / "lakefile.toml").write_text("name = 'HuginnLean'\n")
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError):
                li = LeanInterface(project)

    def test_eval_no_lake(self, tmp_path: Path):
        project = tmp_path / "HuginnLean"
        project.mkdir()
        (project / "lakefile.toml").write_text("name = 'HuginnLean'\n")
        with patch("shutil.which", return_value=None):
            with pytest.raises(RuntimeError):
                li = LeanInterface(project)


# ── SymPy to Lean Translator ──
class TestSymPyToLean:
    def test_init(self):
        t = SymPyToLean()
        assert t is not None

    def test_translate_basic(self):
        import sympy as sp
        t = SymPyToLean()
        x = sp.Symbol("x")
        expr = x**2 + 2*x + 1
        result = t.translate(expr)
        assert "x" in result or "def" in result or "theorem" in result


# ── Lean Tool ──
class TestLeanTool:
    def test_init_no_project(self):
        with patch.object(LeanTool, "_resolve_project_path", return_value=None):
            tool = LeanTool()
        assert tool._project_path is None

    @pytest.mark.asyncio
    async def test_call_build_no_project(self):
        with patch.object(LeanTool, "_resolve_project_path", return_value=None):
            tool = LeanTool()
        result = await tool.call(LeanToolInput(action="build"), CTX)
        assert result.success is False
        assert "project" in result.error.lower()

    @pytest.mark.asyncio
    async def test_call_verify_no_project(self):
        with patch.object(LeanTool, "_resolve_project_path", return_value=None):
            tool = LeanTool()
        result = await tool.call(LeanToolInput(action="verify", theorem_name="test"), CTX)
        assert result.success is False

    @pytest.mark.asyncio
    async def test_call_eval_no_project(self):
        with patch.object(LeanTool, "_resolve_project_path", return_value=None):
            tool = LeanTool()
        result = await tool.call(LeanToolInput(action="eval", lean_code="#eval 1 + 1"), CTX)
        assert result.success is False

    def test_is_read_only(self):
        tool = LeanTool()
        assert tool.is_read_only(LeanToolInput(action="build")) is True
