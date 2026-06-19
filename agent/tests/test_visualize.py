"""Tests for the unified framework visualization layer."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from huginn import visualize as evo_viz
from huginn.cli import cli
from huginn.tools.symbolic_math_tool import SymbolicMathInput, SymbolicMathTool
from huginn.types import ToolContext
from huginn.unified.models import heat_equation_fem
from huginn.unified.visualize import plot_solution, solve_and_plot

CTX = ToolContext(session_id="test", workspace=".")


def test_plot_solution_creates_file() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "sol.png"
        result = plot_solution(
            mesh=[0.0, 0.5, 1.0],
            solution=[0.0, 0.25, 1.0],
            output_path=path,
            title="Test",
        )
        assert result == path
        assert path.exists()


def test_solve_and_plot() -> None:
    problem = heat_equation_fem(k=1.0, f=1.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "heat.png"
        result = solve_and_plot(problem, method="fem", n=8, output_path=path)
        assert "plot_path" in result
        assert Path(result["plot_path"]).exists()
        assert result["residual"] < 1e-10


class TestUnifiedSymbolicVisualize:
    @pytest.fixture
    def tool(self):
        return SymbolicMathTool()

    def test_unified_solve_and_plot(self, tool):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = str(Path(tmpdir) / "unified.png")
            result = asyncio.run(
                tool.call(
                    SymbolicMathInput(
                        action="unified",
                        target="solve_and_plot",
                        expression="heat_equation_fem",
                        variable="fem",
                        order=6,
                    ),
                    CTX,
                )
            )
            # Default output path is used when target is not supplied
            assert result.success
            assert "plot_path" in result.data
            assert Path(result.data["plot_path"]).exists()

    def test_unified_solve_and_plot_custom_path(self, tool):
        with tempfile.TemporaryDirectory() as tmpdir:
            plot_path = str(Path(tmpdir) / "custom.png")
            result = asyncio.run(
                tool.call(
                    SymbolicMathInput(
                        action="unified",
                        target="solve_and_plot",
                        expression="heat_equation_fem",
                        variable="fem",
                        order=4,
                        output_path=plot_path,
                    ),
                    CTX,
                )
            )
            assert result.success
            assert result.data["plot_path"] == plot_path
            assert Path(plot_path).exists()


class TestEvolutionVisualization:
    def test_plot_benchmark_report(self):
        report = {
            "results": [
                {
                    "task_id": "t1",
                    "category": "math",
                    "passed": True,
                    "exec_time_seconds": 1.2,
                },
                {
                    "task_id": "t2",
                    "category": "math",
                    "passed": False,
                    "exec_time_seconds": 2.0,
                },
                {
                    "task_id": "t3",
                    "category": "materials",
                    "passed": True,
                    "exec_time_seconds": 0.8,
                },
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "bench.png"
            result = evo_viz.plot_benchmark_report(report, out)
            assert result == out
            assert out.exists()

    def test_plot_evolution_report(self):
        report = {
            "failure_rules": [{"confidence": 0.8}, {"confidence": 0.6}],
            "success_skills": [{"extraction_confidence": 0.7}],
            "prompt_patches": [{"confidence": 0.9}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "evo.png"
            result = evo_viz.plot_evolution_report(report, out)
            assert result == out
            assert out.exists()

    def test_plot_exploration_result_2d(self):
        result = {
            "pareto_front": [
                {"objectives": {"energy": 1.0, "cost": 2.0}},
                {"objectives": {"energy": 2.0, "cost": 1.0}},
            ],
            "best_branch": {"objectives": {"energy": 1.0, "cost": 2.0}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "exp.png"
            result_path = evo_viz.plot_exploration_result(result, out)
            assert result_path == out
            assert out.exists()

    def test_cli_visualize_bench(self):
        report = {
            "results": [
                {
                    "task_id": "t1",
                    "category": "math",
                    "passed": True,
                    "exec_time_seconds": 1.0,
                },
                {
                    "task_id": "t2",
                    "category": "math",
                    "passed": False,
                    "exec_time_seconds": 1.5,
                },
            ]
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            report_path = Path(tmpdir) / "report.json"
            report_path.write_text(json.dumps(report), encoding="utf-8")
            out_path = Path(tmpdir) / "report.png"
            runner = CliRunner()
            result = runner.invoke(
                cli, ["visualize", "bench", str(report_path), "-o", str(out_path)]
            )
            assert result.exit_code == 0, result.output
            assert out_path.exists()

    def test_plot_benchmark_pie(self):
        report = {
            "results": [
                {"task_id": "t1", "category": "math", "passed": True},
                {"task_id": "t2", "category": "math", "passed": False},
                {"task_id": "t3", "category": "materials", "passed": True},
            ]
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "bench_pie.png"
            result = evo_viz.plot_benchmark_report(report, out, plot_type="pie")
            assert result == out
            assert out.exists()

    def test_plot_exploration_3d(self):
        result = {
            "pareto_front": [
                {"objectives": {"a": 1.0, "b": 2.0, "c": 3.0}},
                {"objectives": {"a": 2.0, "b": 1.0, "c": 2.0}},
            ],
            "best_branch": {"objectives": {"a": 1.0, "b": 2.0, "c": 3.0}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "exp_3d.png"
            result_path = evo_viz.plot_exploration_result(result, out, plot_type="3d")
            assert result_path == out
            assert out.exists()

    def test_plot_exploration_parallel(self):
        result = {
            "pareto_front": [
                {"objectives": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}},
                {"objectives": {"a": 2.0, "b": 1.0, "c": 2.0, "d": 3.0}},
            ],
            "best_branch": {"objectives": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "exp_parallel.png"
            result_path = evo_viz.plot_exploration_result(
                result, out, plot_type="parallel"
            )
            assert result_path == out
            assert out.exists()

    def test_plot_exploration_radar(self):
        result = {
            "pareto_front": [
                {"objectives": {"a": 1.0, "b": 2.0, "c": 3.0}},
            ],
            "best_branch": {"objectives": {"a": 1.0, "b": 2.0, "c": 3.0}},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "exp_radar.png"
            result_path = evo_viz.plot_exploration_result(
                result, out, plot_type="radar"
            )
            assert result_path == out
            assert out.exists()

    def test_cli_visualize_explore_parallel(self):
        result = {
            "pareto_front": [
                {"objectives": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}},
                {"objectives": {"a": 2.0, "b": 1.0, "c": 2.0, "d": 3.0}},
            ],
            "best_branch": {"objectives": {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0}},
        }
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            result_path = Path(tmpdir) / "result.json"
            result_path.write_text(json.dumps(result), encoding="utf-8")
            out_path = Path(tmpdir) / "parallel.png"
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "visualize",
                    "explore",
                    str(result_path),
                    "--type",
                    "parallel",
                    "-o",
                    str(out_path),
                ],
            )
            assert result.exit_code == 0, result.output
            assert out_path.exists()

    def test_plot_evolution_report_confidence_only(self):
        report = {
            "failure_rules": [{"confidence": 0.8}, {"confidence": 0.6}],
            "success_skills": [{"extraction_confidence": 0.7}],
            "prompt_patches": [{"confidence": 0.9}],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "evo_conf.png"
            result = evo_viz.plot_evolution_report(report, out, plot_type="confidence")
            assert result == out
            assert out.exists()

    def test_plot_evolution_convergence(self):
        history = [
            {
                "total_rules": 1,
                "total_skills": 0,
                "avg_confidence": 0.5,
                "new_failure_rules": 1,
                "new_success_skills": 0,
                "new_prompt_patches": 0,
            },
            {
                "total_rules": 3,
                "total_skills": 1,
                "avg_confidence": 0.7,
                "new_failure_rules": 1,
                "new_success_skills": 1,
                "new_prompt_patches": 1,
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "evo_conv.png"
            result = evo_viz.plot_evolution_convergence(history, out)
            assert result == out
            assert out.exists()

    def test_cli_visualize_evolution_convergence(self):
        history = [
            {
                "total_rules": 1,
                "total_skills": 0,
                "avg_confidence": 0.5,
                "new_failure_rules": 1,
                "new_success_skills": 0,
                "new_prompt_patches": 0,
            },
            {
                "total_rules": 3,
                "total_skills": 1,
                "avg_confidence": 0.7,
                "new_failure_rules": 1,
                "new_success_skills": 1,
                "new_prompt_patches": 1,
            },
        ]
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            history_path = Path(tmpdir) / "history.json"
            history_path.write_text(json.dumps(history), encoding="utf-8")
            out_path = Path(tmpdir) / "conv.png"
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "visualize",
                    "evolution",
                    str(history_path),
                    "--type",
                    "convergence",
                    "-o",
                    str(out_path),
                ],
            )
            assert result.exit_code == 0, result.output
            assert out_path.exists()
