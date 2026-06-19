"""Regression tests for CLI commands."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from huginn.cli import cli
from huginn.execution.remote_job_store import RemoteJobRecord, RemoteJobStore


class TestCliCommands:
    def test_version(self):
        result = CliRunner().invoke(cli, ["version"])
        assert result.exit_code == 0
        assert "Huginn" in result.output

    def test_tools_list(self):
        result = CliRunner().invoke(cli, ["tools"])
        assert result.exit_code == 0
        assert "Available Tools" in result.output

    def test_seed_knowledge_help(self):
        result = CliRunner().invoke(cli, ["seed-knowledge", "--help"])
        assert result.exit_code == 0
        assert "--force" in result.output

    def test_model_list_no_config(self):
        result = CliRunner().invoke(cli, ["model-list"])
        assert result.exit_code == 0
        assert "Configured Models" in result.output

    def test_help_lists_commands(self):
        result = CliRunner().invoke(cli, ["--help"])
        assert result.exit_code == 0
        # Core commands should appear in top-level help
        for command in (
            "chat",
            "coder",
            "serve",
            "tools",
            "version",
            "configure",
            "bench",
            "evolve",
            "seed-knowledge",
            "model-list",
            "hpc",
            "unified",
            "persona",
            "swarm",
            "remote",
        ):
            assert command in result.output, f"{command} not in help"

    def test_hpc_help(self):
        result = CliRunner().invoke(cli, ["hpc", "--help"])
        assert result.exit_code == 0
        assert "test" in result.output
        assert "submit" in result.output
        assert "status" in result.output

    def test_unified_help(self):
        result = CliRunner().invoke(cli, ["unified", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "derive" in result.output

    def test_persona_help(self):
        result = CliRunner().invoke(cli, ["persona", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "create" in result.output

    def test_cli_shim_import(self):
        """Ensure the old huginn.cli module still exposes cli and main."""
        from huginn.cli import cli as cli_shim
        from huginn.cli import main as main_shim

        assert callable(cli_shim)
        assert callable(main_shim)

    def test_plot_help(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "plot", "--help"])
        assert result.exit_code == 0
        assert "--y" in result.output

    def test_plot_csv_generates_png(self, tmp_path: Path):
        csv_path = tmp_path / "data.csv"
        csv_path.write_text("x,y\n1,2\n2,4\n3,6\n", encoding="utf-8")
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "plot",
                str(csv_path),
                "--x",
                "x",
                "--y",
                "y",
                "--kind",
                "scatter",
            ],
        )
        assert result.exit_code == 0
        assert (tmp_path / "data_scatter.png").exists()

    def test_remote_help(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "remote", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "status" in result.output
        assert "cancel" in result.output
        assert "logs" in result.output
        assert "watch" in result.output
        assert "export" in result.output

    def test_remote_list_empty(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "remote", "list"])
        assert result.exit_code == 0
        assert "No remote jobs" in result.output

    def test_remote_list_and_status(self, tmp_path: Path):
        store = RemoteJobStore(workspace=tmp_path)
        store.add_or_update(
            RemoteJobRecord(
                local_id="vasp-gpu",
                scheduler_id="12345",
                command=["vasp_std"],
                cwd=str(tmp_path),
                queue="gpu",
                status="COMPLETED",
                submitted_at=1.0,
            )
        )

        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "remote", "list"])
        assert result.exit_code == 0
        assert "vasp-gpu" in result.output
        assert "12345" in result.output
        assert "gpu" in result.output

        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "remote", "status", "vasp-gpu", "--no-refresh"]
        )
        assert result.exit_code == 0
        assert "vasp-gpu" in result.output
        assert "COMPLETED" in result.output

    def test_remote_cancel_and_logs(self, tmp_path: Path):
        store = RemoteJobStore(workspace=tmp_path)
        store.add_or_update(
            RemoteJobRecord(
                local_id="j1",
                scheduler_id="999",
                command=["echo", "hello"],
                cwd=str(tmp_path),
                status="RUNNING",
                submitted_at=1.0,
            )
        )
        (tmp_path / "slurm-999.out").write_text("job output\n", encoding="utf-8")

        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "remote", "cancel", "j1"]
        )
        assert result.exit_code == 0
        assert store.get("j1").status == "CANCELLED"

        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "remote", "logs", "j1"])
        assert result.exit_code == 0
        assert "job output" in result.output

        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "remote", "logs", "j1", "--stderr"]
        )
        assert result.exit_code == 0
        assert "No stderr log found" in result.output

    def test_remote_export_json(self, tmp_path: Path):
        store = RemoteJobStore(workspace=tmp_path)
        store.add_or_update(
            RemoteJobRecord(
                local_id="j1",
                scheduler_id="999",
                command=["echo", "hello"],
                cwd=str(tmp_path),
                status="COMPLETED",
                submitted_at=1.0,
            )
        )
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "remote", "export", "--format", "json"]
        )
        assert result.exit_code == 0
        assert "j1" in result.output
        assert "999" in result.output

    def test_remote_export_csv(self, tmp_path: Path):
        store = RemoteJobStore(workspace=tmp_path)
        store.add_or_update(
            RemoteJobRecord(
                local_id="j2",
                scheduler_id="888",
                command=["date"],
                cwd=str(tmp_path),
                status="RUNNING",
                submitted_at=2.0,
            )
        )
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "remote", "export", "--format", "csv"]
        )
        assert result.exit_code == 0
        assert "local_id" in result.output
        assert "j2" in result.output

    def test_remote_prune(self, tmp_path: Path):
        store = RemoteJobStore(workspace=tmp_path)
        for i, status in enumerate(["COMPLETED", "FAILED", "RUNNING"]):
            store.add_or_update(
                RemoteJobRecord(
                    local_id=f"j{i}",
                    scheduler_id=f"s{i}",
                    command=["cmd"],
                    cwd=str(tmp_path),
                    status=status,
                    submitted_at=float(i),
                    completed_at=float(i) if status != "RUNNING" else None,
                )
            )
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "remote",
                "prune",
                "--status",
                "COMPLETED,FAILED",
                "--yes",
            ],
        )
        assert result.exit_code == 0
        assert store.get("j0") is None
        assert store.get("j1") is None
        assert store.get("j2") is not None
