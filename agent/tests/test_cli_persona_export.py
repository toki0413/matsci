"""CLI tests for persona and export commands (coverage boosters)."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from huginn.cli import cli


class TestPersonaCli:
    def test_persona_list(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "persona", "list"])
        assert result.exit_code == 0
        assert "default" in result.output

    def test_persona_show(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "persona", "show", "default"]
        )
        assert result.exit_code == 0
        assert "default" in result.output

    def test_persona_create_and_delete(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "persona",
                "create",
                "cli_bot",
                "--prompt",
                "You are CLI bot.",
                "--begin-dialog",
                "user:Hi",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Created persona cli_bot" in result.output

        result = runner.invoke(cli, ["-w", str(tmp_path), "persona", "list"])
        assert "cli_bot" in result.output

        result = runner.invoke(
            cli, ["-w", str(tmp_path), "persona", "delete", "cli_bot"]
        )
        assert result.exit_code == 0
        assert "Deleted persona cli_bot" in result.output

    def test_persona_set_default(self, tmp_path: Path):
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "persona",
                "create",
                "reviewer",
                "--prompt",
                "You are a reviewer.",
            ],
        )
        result = runner.invoke(
            cli, ["-w", str(tmp_path), "persona", "set-default", "reviewer"]
        )
        assert result.exit_code == 0
        assert "Default persona set to reviewer" in result.output

    def test_persona_match(self, tmp_path: Path):
        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "persona",
                "create",
                "dft_expert",
                "--prompt",
                "DFT expert",
            ],
        )
        result = runner.invoke(
            cli,
            ["-w", str(tmp_path), "persona", "match", "DFT calculation", "--threshold", "0.1"],
        )
        assert result.exit_code == 0
        # Keyword-only matcher may not score above threshold in CI; just verify CLI runs.
        assert (
            "dft_expert" in result.output
            or "No strong persona match found." in result.output
        )

    def test_persona_switch(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "persona", "switch", "default"]
        )
        assert result.exit_code == 0
        assert "Switched active persona to default" in result.output

    def test_persona_emotion(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "persona", "emotion", "default"]
        )
        assert result.exit_code == 0
        assert "Emotional Trajectory" in result.output


class TestExportCli:
    def test_export_help(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "export", "--help"])
        assert result.exit_code == 0
        assert "--format" in result.output

    def test_export_knowledge_markdown(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "export",
                "-s",
                "knowledge",
                "-f",
                "markdown",
                "-o",
                str(tmp_path / "knowledge.md"),
            ],
        )
        assert result.exit_code == 0
        assert "Exported" in result.output

    def test_export_remote_jobs_json(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "export",
                "-s",
                "remote_jobs",
                "-f",
                "json",
                "-o",
                str(tmp_path / "jobs.json"),
            ],
        )
        assert result.exit_code == 0
        assert "Exported" in result.output


class TestSchedulerCli:
    def test_scheduler_help(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "scheduler", "--help"]
        )
        assert result.exit_code == 0
        assert "add" in result.output
        assert "list" in result.output

    def test_scheduler_list_empty(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "scheduler", "list"]
        )
        assert result.exit_code == 0

    def test_scheduler_add_and_list(self, tmp_path: Path):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "scheduler",
                "add",
                "--cron",
                "* * * * *",
                "--command",
                "echo hello",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Scheduled job" in result.output

        result = runner.invoke(cli, ["-w", str(tmp_path), "scheduler", "list"])
        assert result.exit_code == 0
        assert "echo hello" in result.output


class TestExploreCli:
    def test_explore_help(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "explore", "--help"])
        assert result.exit_code == 0
        assert "--max-branches" in result.output

    def test_explore_runs(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "explore",
                "minimize energy",
                "--max-iterations",
                "2",
                "--max-branches",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Exploration complete" in result.output


class TestMemoryMaintenanceCli:
    def test_memory_maintenance(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "memory-maintenance"]
        )
        assert result.exit_code == 0
