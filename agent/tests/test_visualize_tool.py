"""Tests for the visualize tool."""

from __future__ import annotations

import asyncio
import json

import pytest

from huginn.tools.registry import ToolRegistry
from huginn.tools.visualize_tool import VisualizeTool
from huginn.types import ToolContext


@pytest.fixture(autouse=True)
def _clear_registry():
    ToolRegistry.clear()
    ToolRegistry.register(VisualizeTool())
    yield
    ToolRegistry.clear()


class TestVisualizeTool:
    def test_benchmark_plot(self, tmp_path):
        tool = VisualizeTool()
        report_path = tmp_path / "benchmark.json"
        report_path.write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "task_id": "t1",
                            "category": "dft",
                            "passed": True,
                            "exec_time_seconds": 1.5,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "bench.png"

        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="benchmark",
                    report_path=str(report_path),
                    output_path=str(output_path),
                    plot_type="bar",
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is True
        assert output_path.exists()
        assert result.data["exists"] is True

    def test_evolution_plot(self, tmp_path):
        tool = VisualizeTool()
        report_path = tmp_path / "evolution.json"
        report_path.write_text(
            json.dumps(
                {
                    "failure_rules": [{"confidence": 0.8}],
                    "success_skills": [],
                    "prompt_patches": [],
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "evolution.png"

        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="evolution",
                    report_path=str(report_path),
                    output_path=str(output_path),
                    plot_type="summary",
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is True
        assert output_path.exists()

    def test_exploration_plot(self, tmp_path):
        tool = VisualizeTool()
        report_path = tmp_path / "exploration.json"
        report_path.write_text(
            json.dumps(
                {
                    "pareto_front": [{"objectives": {"energy": -1.0, "distance": 2.0}}],
                    "best_branch": {"objectives": {"energy": -1.0, "distance": 2.0}},
                }
            ),
            encoding="utf-8",
        )
        output_path = tmp_path / "exploration.png"

        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="exploration",
                    report_path=str(report_path),
                    output_path=str(output_path),
                    plot_type="2d",
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is True
        assert output_path.exists()

    def test_missing_report_fails(self, tmp_path):
        tool = VisualizeTool()
        output_path = tmp_path / "missing.png"

        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="benchmark",
                    report_path=str(tmp_path / "nope.json"),
                    output_path=str(output_path),
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is False
        assert "Failed to load report" in (result.error or "")

    def test_report_data_input(self, tmp_path):
        tool = VisualizeTool()
        output_path = tmp_path / "from_data.png"

        result = asyncio.run(
            tool.call(
                tool.input_schema(
                    action="benchmark",
                    report_data={"results": []},
                    output_path=str(output_path),
                    plot_type="pie",
                ),
                ToolContext(session_id="test", workspace=str(tmp_path)),
            )
        )

        assert result.success is False
        assert "No results" in (result.error or "")
