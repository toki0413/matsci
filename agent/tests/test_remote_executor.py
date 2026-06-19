"""Tests for RemoteExecutor and HPC execution backend wiring."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from huginn.config import HuginnConfig
from huginn.execution.remote_executor import RemoteExecutor, build_executor
from huginn.hpc.client import HPCConfig, JobStatus
from huginn.security.sandbox import SandboxExecutor, SandboxResult
from huginn.tools import register_all_tools
from huginn.tools.registry import ToolRegistry


class _FakeHPCClient:
    """Minimal fake of HPCClient for RemoteExecutor tests."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.uploaded: dict[str, str] = {}
        self.downloaded: dict[str, str] = {}
        self.script: str | None = None
        self.job_id = "12345"
        self.status = JobStatus(job_id="12345", state="COMPLETED", exit_code=0)

    def connect(self) -> None:
        self.calls.append("connect")

    def disconnect(self) -> None:
        self.calls.append("disconnect")

    def _exec(self, command: Any) -> tuple[str, str, int]:
        self.calls.append(f"exec:{command}")
        return "", "", 0

    def upload_file(self, local_path: str, remote_path: str) -> None:
        self.uploaded[remote_path] = local_path

    def download_file(self, remote_path: str, local_path: str) -> None:
        self.downloaded[remote_path] = local_path
        # Create a minimal tar.gz so the executor can extract it.
        with tarfile.open(local_path, "w:gz") as tar:
            data = b"fake remote output"
            info = tarfile.TarInfo(name="slurm-12345.out")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

    def generate_job_script(
        self,
        command: str,
        job_name: str = "huginn_job",
        **kwargs: Any,
    ) -> str:
        self.calls.append(f"script:{job_name}")
        self.script = command
        self.script_kwargs = kwargs
        return "# fake script"

    def submit_job(self, script_content: str, job_name: str = "huginn_job") -> str:
        self.calls.append("submit")
        return self.job_id

    def wait_for_job(
        self,
        job_id: str,
        poll_interval: int = 30,
        timeout: int = 86400,
    ) -> JobStatus:
        self.calls.append("wait")
        return self.status


@pytest.fixture
def fake_executor(tmp_path: Path) -> tuple[RemoteExecutor, _FakeHPCClient]:
    config = HPCConfig(host="hpc.example.com", username="user")
    executor = RemoteExecutor(config)
    fake_client = _FakeHPCClient()
    executor._client = fake_client  # type: ignore[assignment]
    return executor, fake_client


class TestRemoteExecutor:
    def test_run_stages_submits_and_downloads(
        self, fake_executor: tuple[RemoteExecutor, _FakeHPCClient], tmp_path: Path
    ) -> None:
        executor, client = fake_executor
        work_dir = tmp_path / "job"
        work_dir.mkdir()
        (work_dir / "POSCAR").write_text("H 0 0 0", encoding="utf-8")

        result = executor.run(["vasp_std"], cwd=str(work_dir), timeout=60)

        assert isinstance(result, SandboxResult)
        assert result.success is True
        assert result.returncode == 0
        assert "connect" in client.calls
        assert "submit" in client.calls
        assert "wait" in client.calls
        assert any("tar -xzf" in c for c in client.calls)
        assert any("tar -czf" in c for c in client.calls)

    def test_run_returns_failure_on_exception(
        self, fake_executor: tuple[RemoteExecutor, _FakeHPCClient], tmp_path: Path
    ) -> None:
        executor, client = fake_executor
        client.connect = MagicMock(side_effect=RuntimeError("connection refused"))

        work_dir = tmp_path / "job"
        work_dir.mkdir()
        result = executor.run(["vasp_std"], cwd=str(work_dir), timeout=60)

        assert result.success is False
        assert "connection refused" in result.stderr

    def test_run_passes_gpu_queue_to_scheduler(
        self, fake_executor: tuple[RemoteExecutor, _FakeHPCClient], tmp_path: Path
    ) -> None:
        executor, client = fake_executor
        work_dir = tmp_path / "job"
        work_dir.mkdir()

        result = executor.run(
            ["vasp_std"],
            cwd=str(work_dir),
            timeout=60,
            queue="gpu",
            job_name="vasp-gpu",
        )

        assert result.success is True
        assert client.script_kwargs.get("queue") == "gpu"
        jobs = executor.list_jobs()
        assert len(jobs) == 1
        assert jobs[0].queue == "gpu"
        assert jobs[0].local_id == "vasp-gpu"

    def test_run_retries_on_transient_connection_error(
        self, fake_executor: tuple[RemoteExecutor, _FakeHPCClient], tmp_path: Path
    ) -> None:
        executor, client = fake_executor
        attempts = {"count": 0}

        def flaky_connect() -> None:
            attempts["count"] += 1
            if attempts["count"] < 2:
                raise TimeoutError("ssh connection timeout")
            client.calls.append("connect")

        client.connect = flaky_connect
        work_dir = tmp_path / "job"
        work_dir.mkdir()

        result = executor.run(["vasp_std"], cwd=str(work_dir), timeout=60)

        assert result.success is True
        assert attempts["count"] == 2

    def test_cancel_job_issues_scancel(
        self, fake_executor: tuple[RemoteExecutor, _FakeHPCClient], tmp_path: Path
    ) -> None:
        executor, client = fake_executor
        work_dir = tmp_path / "job"
        work_dir.mkdir()

        executor.run(["vasp_std"], cwd=str(work_dir), timeout=60, job_name="to-cancel")
        ok = executor.cancel_job("to-cancel")

        assert ok is True
        assert any("scancel 12345" in c for c in client.calls)
        assert executor.get_job("to-cancel").status == "CANCELLED"

    def test_run_routes_gpu_hint_to_gpu_queue(
        self, fake_executor: tuple[RemoteExecutor, _FakeHPCClient], tmp_path: Path
    ) -> None:
        executor, client = fake_executor
        executor.config.gpu_queue = "gpu"
        work_dir = tmp_path / "job"
        work_dir.mkdir()

        result = executor.run(
            ["vasp_std"],
            cwd=str(work_dir),
            timeout=60,
            gpu=True,
            job_name="gpu-job",
        )

        assert result.success is True
        assert client.script_kwargs.get("queue") == "gpu"
        assert client.script_kwargs.get("gpus_per_node") == 1
        job = executor.get_job("gpu-job")
        assert job is not None
        assert job.queue == "gpu"

    def test_run_uses_queue_map_profile(
        self, fake_executor: tuple[RemoteExecutor, _FakeHPCClient], tmp_path: Path
    ) -> None:
        executor, client = fake_executor
        executor.config.queue_map = {"fat": "fat_nodes"}
        executor.config.default_queue = "normal"
        work_dir = tmp_path / "job"
        work_dir.mkdir()

        result = executor.run(
            ["vasp_std"],
            cwd=str(work_dir),
            timeout=60,
            profile="fat",
            job_name="fat-job",
        )

        assert result.success is True
        assert client.script_kwargs.get("queue") == "fat_nodes"

    def test_run_explicit_queue_overrides_profile(
        self, fake_executor: tuple[RemoteExecutor, _FakeHPCClient], tmp_path: Path
    ) -> None:
        executor, client = fake_executor
        executor.config.queue_map = {"fat": "fat_nodes"}
        work_dir = tmp_path / "job"
        work_dir.mkdir()

        result = executor.run(
            ["vasp_std"],
            cwd=str(work_dir),
            timeout=60,
            profile="fat",
            queue="custom",
            job_name="override-job",
        )

        assert result.success is True
        assert client.script_kwargs.get("queue") == "custom"

    def test_retry_uses_configured_max_retries(
        self, fake_executor: tuple[RemoteExecutor, _FakeHPCClient], tmp_path: Path
    ) -> None:
        executor, client = fake_executor
        executor.config.max_retries = 2
        client.connect = MagicMock(side_effect=RuntimeError("ssh connection timeout"))

        work_dir = tmp_path / "job"
        work_dir.mkdir()
        result = executor.run(["vasp_std"], cwd=str(work_dir), timeout=60)

        assert result.success is False
        assert client.connect.call_count == 2


class TestBuildExecutor:
    def test_local_backend_returns_sandbox(self) -> None:
        cfg = HuginnConfig(execution_backend="local")
        executor = build_executor(cfg)
        assert isinstance(executor, SandboxExecutor)

    def test_remote_backend_without_host_falls_back(self) -> None:
        cfg = HuginnConfig(
            execution_backend="remote", hpc_host=None, hpc_username="user"
        )
        executor = build_executor(cfg)
        assert isinstance(executor, SandboxExecutor)

    def test_remote_backend_returns_remote_executor(self) -> None:
        cfg = HuginnConfig(
            execution_backend="remote",
            hpc_host="hpc.example.com",
            hpc_username="user",
            hpc_scheduler="slurm",
        )
        executor = build_executor(cfg)
        assert isinstance(executor, RemoteExecutor)


class TestRegisterAllToolsRemote:
    def test_register_all_tools_with_remote_config_passes_executor(self) -> None:
        ToolRegistry.clear()
        cfg = HuginnConfig(
            execution_backend="remote",
            hpc_host="hpc.example.com",
            hpc_username="user",
        )
        register_all_tools(config=cfg)

        from huginn.tools.vasp_tool import VaspTool

        vasp = ToolRegistry.get("vasp_tool")
        assert isinstance(vasp, VaspTool)
        assert isinstance(vasp.sandbox, RemoteExecutor)
        ToolRegistry.clear()
