"""Additional CLI command tests to boost huginn/cli/commands/ coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

from huginn.cli import cli


class TestChatCli:
    def test_chat_help(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "chat", "--help"])
        assert result.exit_code == 0
        assert "interactive" in result.output.lower()

    def test_chat_mocked_agent(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        mock_agent = MagicMock()
        mock_agent.langchain_tools = []
        mock_agent.invoke.return_value = {
            "messages": [MagicMock(content="hello from agent")]
        }
        monkeypatch.setattr(
            "huginn.cli.commands.chat.build_agent_from_ctx", lambda ctx: mock_agent
        )
        monkeypatch.setattr("huginn.cli.commands.chat.init_mcp", lambda cfg: None)
        monkeypatch.setattr("huginn.cli.commands.chat.shutdown_mcp", lambda: None)

        result = CliRunner().invoke(
            cli,
            ["-w", str(tmp_path), "chat"],
            input="hello\nexit\n",
        )
        assert result.exit_code == 0, result.output
        assert "Agent initialized" in result.output
        assert "hello from agent" in result.output


class TestCoderCli:
    def test_coder_help(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "coder", "--help"])
        assert result.exit_code == 0
        assert "--auto-approve" in result.output

    def test_coder_mocked_runner(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        class FakeRunner:
            def __init__(self, *args, **kwargs):
                pass

            def run(self, task: str):
                return {"final_answer": f"completed: {task}"}

        monkeypatch.setattr("huginn.coder.CoderRunner", FakeRunner)
        result = CliRunner().invoke(
            cli,
            ["-w", str(tmp_path), "coder", "add docstring", "--auto-approve"],
        )
        assert result.exit_code == 0, result.output
        assert "completed: add docstring" in result.output


class TestConfigureCli:
    def test_configure_creates_config(self, tmp_path: Path):
        config_path = tmp_path / "huginn.toml"
        # Provide default-ish answers for each prompt (enter = empty)
        result = CliRunner().invoke(
            cli,
            ["-w", str(tmp_path), "configure", "--path", str(config_path)],
            input="\n\n\n\n\n\n\n",
        )
        assert result.exit_code == 0, result.output
        assert config_path.exists()
        assert "Config saved" in result.output


class TestDiagnoseCli:
    def test_diagnose_vasp_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from huginn.types import ToolResult

        async def fake_call(self, inp, ctx):
            return ToolResult(
                success=True,
                data={"diagnosis": "EDWAV error", "suggestion": "increase NBANDS"},
            )

        monkeypatch.setattr("huginn.tools.diagnose_tool.DiagnoseTool.call", fake_call)
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "diagnose",
                "EDWAV: failed to converge",
                "--software",
                "VASP",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "EDWAV error" in result.output


class TestEncryptConfigCli:
    def test_encrypt_config(self, tmp_path: Path):
        config_path = tmp_path / "huginn.toml"
        config_path.write_text(
            'provider = "openai"\nmodel = "gpt-4o"\n', encoding="utf-8"
        )
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "encrypt-config",
                str(config_path),
                "--password",
                "secret",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "huginn.toml.enc").exists()
        assert "Encrypted config saved" in result.output


class TestExecuteCli:
    def test_execute_inline_stages(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from huginn.execution.orchestrator import StageResult, WorkflowExecutionRecord

        fake_record = WorkflowExecutionRecord(workflow_name="execute")
        fake_record.overall_success = True
        fake_record.stage_results = [
            StageResult(
                stage_id="s1", stage_name="stage1", tool_name="dummy", success=True
            )
        ]

        async def fake_run(self, stages, workflow_name):
            return fake_record

        monkeypatch.setattr(
            "huginn.execution.orchestrator.ExecutionOrchestrator.run", fake_run
        )
        monkeypatch.setattr("huginn.tools.registry.ToolRegistry.list_tools", lambda: [])

        stages = json.dumps([{"id": "s1", "tool": "dummy", "action": "noop"}])
        result = CliRunner().invoke(
            cli,
            ["-w", str(tmp_path), "execute", stages, "--working-dir", str(tmp_path)],
        )
        assert result.exit_code == 0, result.output
        assert "overall_success" in result.output


class TestExploreCli:
    def test_explore_bayesian(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        class FakeResult:
            convergence_reason = "max iterations"
            n_branches_explored = 3
            n_branches_pruned = 1
            pareto_front = [{"name": "b1"}]
            best_branch = {"name": "b1"}

        async def fake_explore(self, **kwargs):
            return FakeResult()

        monkeypatch.setattr(
            "huginn.exploration.orchestrator.ExplorationOrchestrator.explore",
            fake_explore,
        )
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "explore",
                "minimize energy",
                "--strategy",
                "bayesian",
                "--max-iterations",
                "2",
                "--max-branches",
                "2",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Exploration complete" in result.output


class TestHpcCli:
    def test_hpc_test(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        class FakeClient:
            def __init__(self, cfg):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def _exec(self, cmd):
                return ("hpc-node", "", 0)

        monkeypatch.setattr("huginn.hpc.client.HPCClient", FakeClient)
        result = CliRunner().invoke(
            cli,
            ["-w", str(tmp_path), "hpc", "test", "--host", "hpc", "--username", "u"],
        )
        assert result.exit_code == 0, result.output
        assert "Connected to hpc" in result.output

    def test_hpc_submit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        class FakeClient:
            def __init__(self, cfg):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def generate_job_script(self, **kwargs):
                return "script"

            def submit_job(self, script, job_name):
                return "12345"

        monkeypatch.setattr("huginn.hpc.client.HPCClient", FakeClient)
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "hpc",
                "submit",
                "--host",
                "hpc",
                "--username",
                "u",
                "--command",
                "echo hi",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Submitted" in result.output
        assert "12345" in result.output

    def test_hpc_status(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        class FakeStatus:
            job_id = "12345"
            state = "RUNNING"
            exit_code = None
            runtime = 120.0
            message = "ok"

        class FakeClient:
            def __init__(self, cfg):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def poll_status(self, job_id):
                return FakeStatus()

        monkeypatch.setattr("huginn.hpc.client.HPCClient", FakeClient)
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "hpc",
                "status",
                "--host",
                "hpc",
                "--username",
                "u",
                "--job-id",
                "12345",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "RUNNING" in result.output


class TestKgCli:
    def _make_fake_kg_class(self, monkeypatch: pytest.MonkeyPatch):
        """Patch ProjectKnowledgeGraph so kg subcommands can run offline."""
        fake_cls = MagicMock()
        fake_cls.FILENAME = "project_kg.json"
        fake_instance = MagicMock()
        fake_instance.path = Path("/tmp/project_kg.json")
        fake_instance._graph = MagicMock()
        fake_cls.return_value = fake_instance
        monkeypatch.setattr("huginn.cli.commands.kg.ProjectKnowledgeGraph", fake_cls)
        return fake_instance

    def test_kg_build_and_stats(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        fake_instance = self._make_fake_kg_class(monkeypatch)
        fake_instance.stats.return_value = {
            "nodes": 5,
            "edges": 4,
            "node_types": {"topic": 5},
        }
        monkeypatch.setattr(
            "huginn.cli.commands.kg.build_from_seeds",
            lambda kg: {"topics": 2, "links": 1},
        )

        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "build-kg", "--from-seeds"]
        )
        assert result.exit_code == 0, result.output
        assert "Project Knowledge Graph" in result.output

        # stats checks the actual file on disk before instantiating the graph
        kg_file = tmp_path / ".huginn" / "project_kg.json"
        kg_file.parent.mkdir(parents=True, exist_ok=True)
        kg_file.write_text("{}", encoding="utf-8")
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "kg", "stats"])
        assert result.exit_code == 0, result.output
        assert "Nodes: 5" in result.output

    def test_kg_query_and_export(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        fake_instance = self._make_fake_kg_class(monkeypatch)
        fake_instance.export.return_value = {"nodes": [], "edges": []}

        # Ensure kg file exists so query/export are not short-circuited
        kg_file = tmp_path / ".huginn" / "project_kg.json"
        kg_file.parent.mkdir(parents=True, exist_ok=True)
        kg_file.write_text("{}", encoding="utf-8")

        monkeypatch.setattr(
            "huginn.cli.commands.kg.GraphQuery.query",
            lambda self, seed, depth, top_k: {"seed": seed, "nodes": []},
        )

        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "kg", "query", "DFT", "--depth", "2"]
        )
        assert result.exit_code == 0, result.output
        assert "DFT" in result.output

        out_path = tmp_path / "kg.json"
        result = CliRunner().invoke(
            cli,
            ["-w", str(tmp_path), "kg", "export", "--output", str(out_path)],
        )
        assert result.exit_code == 0, result.output
        assert out_path.exists()


class TestMemoryMaintenanceCli:
    def test_memory_maintenance_with_threshold(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli,
            ["-w", str(tmp_path), "memory-maintenance", "--prune-threshold", "0.1"],
        )
        assert result.exit_code == 0, result.output
        assert "Memory Maintenance" in result.output


class TestModelListCli:
    def test_model_list_with_thinking(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "--thinking", "low", "model-list"]
        )
        assert result.exit_code == 0, result.output
        assert "Configured Models" in result.output


class TestPlotCli:
    def test_plot_line(self, tmp_path: Path):
        csv = tmp_path / "data.csv"
        csv.write_text("x,y\n1,2\n2,4\n3,6\n", encoding="utf-8")
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "plot",
                str(csv),
                "--x",
                "x",
                "--y",
                "y",
                "--kind",
                "line",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "data_line.png").exists()

    def test_plot_bar(self, tmp_path: Path):
        csv = tmp_path / "data.csv"
        csv.write_text("x,y\na,2\nb,4\n", encoding="utf-8")
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "plot",
                str(csv),
                "--x",
                "x",
                "--y",
                "y",
                "--kind",
                "bar",
            ],
        )
        assert result.exit_code == 0, result.output
        assert (tmp_path / "data_bar.png").exists()


class TestRefactorCli:
    def test_refactor_dry_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        class Edit:
            path = "a.py"

        class FakeEngine:
            def __init__(self, root, config):
                pass

            def plan(self, task, target_files=None):
                return [Edit()]

            def apply(self, plan, dry_run=False):
                return {
                    "diff": "-old\\n+new",
                    "errors": [],
                    "applied": 1,
                    "snapshots": {},
                }

        monkeypatch.setattr("huginn.cli.commands.refactor.RefactorEngine", FakeEngine)
        result = CliRunner().invoke(
            cli,
            ["-w", str(tmp_path), "refactor", "rename foo to bar", "--dry-run"],
        )
        assert result.exit_code == 0, result.output
        assert "Dry run" in result.output

    def test_refactor_rollback(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        snapshot = tmp_path / ".huginn_refactor_snapshots.json"
        snapshot.write_text('{"a.py": "original"}', encoding="utf-8")

        class FakeEngine:
            def __init__(self, root, config):
                pass

            def rollback(self, snapshots):
                pass

        monkeypatch.setattr("huginn.cli.commands.refactor.RefactorEngine", FakeEngine)
        result = CliRunner().invoke(
            cli,
            ["-w", str(tmp_path), "refactor", "--rollback", "dummy-task"],
        )
        assert result.exit_code == 0, result.output
        assert "Rolled back" in result.output


class TestSchedulerCli:
    def test_scheduler_delete_and_run_now(self, tmp_path: Path):
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
                "echo hi",
            ],
        )
        assert result.exit_code == 0, result.output
        job_id = result.output.split("Scheduled job ")[-1].split(":")[0].strip()

        result = runner.invoke(
            cli, ["-w", str(tmp_path), "scheduler", "remove", job_id]
        )
        assert result.exit_code == 0, result.output
        assert f"Removed job {job_id}" in result.output

        result = runner.invoke(cli, ["-w", str(tmp_path), "scheduler", "run-now"])
        assert result.exit_code == 0, result.output
        assert "No jobs due" in result.output


class TestSeedKnowledgeCli:
    def test_seed_knowledge_help(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "seed-knowledge", "--help"]
        )
        assert result.exit_code == 0
        assert "--force" in result.output

    def test_seed_knowledge_force(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        fake_kb = MagicMock()
        monkeypatch.setattr(
            "huginn.knowledge.get_knowledge_base", lambda w: fake_kb
        )
        monkeypatch.setattr(
            "huginn.knowledge.seed_knowledge_base",
            lambda kb, force: {"added": 2, "skipped": 0, "failed": 0},
        )
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "seed-knowledge", "--force"]
        )
        assert result.exit_code == 0, result.output
        assert "Added: 2" in result.output


class TestServeCli:
    def test_serve_help(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "serve", "--help"])
        assert result.exit_code == 0
        assert "--port" in result.output


class TestSwarmCli:
    def test_swarm_run(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        mock_agent = MagicMock()
        monkeypatch.setattr(
            "huginn.cli.commands.swarm.build_agent_from_ctx",
            lambda ctx, profile_id: mock_agent,
        )

        class FakeSwarm:
            def __init__(self, workers):
                pass

            async def run(self, task):
                return {
                    "final_output": f"swarm result for {task}",
                    "trace": [
                        {
                            "role": "planner",
                            "agent_name": "planner",
                            "duration_ms": 10.0,
                        }
                    ],
                }

        monkeypatch.setattr("huginn.agents.swarm.HuginnSwarm", FakeSwarm)
        monkeypatch.setattr("huginn.agents.swarm.SwarmAgent", MagicMock)

        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "swarm", "run", "optimize lattice"]
        )
        assert result.exit_code == 0, result.output
        assert "swarm result for optimize lattice" in result.output


class TestTelemetryCli:
    def test_telemetry_summary(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        mock_agent = MagicMock()
        mock_agent.telemetry_summary.return_value = {
            "total_spans": 7,
            "by_name": {"tool_call": {"count": 5, "duration_ms": 12.3}},
        }
        mock_agent.close = MagicMock()

        monkeypatch.setattr(
            "huginn.cli.commands.telemetry.HuginnAgent.from_config", lambda cfg: mock_agent
        )
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "telemetry"])
        assert result.exit_code == 0, result.output
        assert "Total spans: 7" in result.output
        assert "tool_call" in result.output


class TestUnifiedCli:
    def test_unified_list(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "unified", "list"])
        assert result.exit_code == 0, result.output
        assert "Unified Models" in result.output

    def test_unified_derive(self, tmp_path: Path):
        result = CliRunner().invoke(
            cli, ["-w", str(tmp_path), "unified", "derive", "heat_equation_fem"]
        )
        assert result.exit_code == 0, result.output
        assert "Equations:" in result.output


class TestVersionCli:
    def test_version(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "version"])
        assert result.exit_code == 0, result.output
        assert "Huginn" in result.output


class TestVisualizeCli:
    def test_visualize_evolution_convergence(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        report = tmp_path / "evolution_history.json"
        report.write_text(
            json.dumps(
                [
                    {
                        "total_rules": 1,
                        "total_skills": 1,
                        "avg_confidence": 0.8,
                    }
                ]
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "huginn.cli.commands.visualize.plot_from_file",
            lambda kind, path, output, plot_type=None: Path(output),
        )
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "visualize",
                "evolution",
                str(report),
                "--type",
                "convergence",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Evolution plot saved" in result.output

    def test_visualize_explore_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        report = tmp_path / "explore_result.json"
        report.write_text(
            json.dumps(
                {
                    "pareto_front": [
                        {"objectives": {"energy": 1.0, "force": 0.1}}
                    ],
                    "best_branch": {
                        "objectives": {"energy": 1.0, "force": 0.1}
                    },
                }
            ),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            "huginn.cli.commands.visualize.plot_from_file",
            lambda kind, path, output, plot_type=None: Path(output),
        )
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "visualize",
                "explore",
                str(report),
                "--type",
                "2d",
            ],
        )
        assert result.exit_code == 0, result.output
        assert "Exploration plot saved" in result.output


class TestWorkflowCli:
    def test_workflow_help(self, tmp_path: Path):
        result = CliRunner().invoke(cli, ["-w", str(tmp_path), "workflow", "--help"])
        assert result.exit_code == 0
        assert "TEMPLATE" in result.output

    def test_workflow_execute_symbolic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        from huginn.workflows.stages import WorkflowResult

        async def fake_execute(self, stages, context, checkpoint_path=None):
            return WorkflowResult(success=True, stages={}, outputs={"out": 1})

        monkeypatch.setattr(
            "huginn.workflows.engine.WorkflowEngine.execute", fake_execute
        )
        monkeypatch.setattr(
            "huginn.workflows.engine.WorkflowEngine.resume", fake_execute
        )

        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "workflow",
                "symbolic_verify",
                'verify_type="derivative"',
                'expression="x**2"',
                'symbols=["x"]',
            ],
        )
        assert result.exit_code == 0, result.output
        assert "success" in result.output


class TestAutoresearchCli:
    def test_autoresearch_status(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        from huginn.types import ToolResult

        async def fake_call(self, inp, ctx):
            return ToolResult(success=True, data={"status": "clean"})

        monkeypatch.setattr(
            "huginn.plugins.autoresearch.AutoresearchTool.call", fake_call
        )
        result = CliRunner().invoke(
            cli,
            [
                "-w",
                str(tmp_path),
                "autoresearch",
                "status",
                "--workspace",
                str(tmp_path / "autoresearch"),
            ],
        )
        assert result.exit_code == 0, result.output
        assert "status" in result.output
