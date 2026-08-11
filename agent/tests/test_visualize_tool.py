"""Tests for the visualize tool."""

from __future__ import annotations

import asyncio
import json

import pytest

from huginn.core_types import ToolContext
from huginn.tools.registry import ToolRegistry
from huginn.tools.visualize_tool import VisualizeTool


@pytest.fixture(autouse=True)
def _isolate_registry():
    """隔离 ToolRegistry: 测试自用工具不影响全局注册表."""
    before = ToolRegistry.snapshot()
    yield
    ToolRegistry.restore(before)


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

    def test_benchmark_plot_has_visual_gate(self, tmp_path):
        """批次 C 后半: 成功生成图后附加 visual_gate 字段."""
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
        output_path = tmp_path / "bench_gate.png"

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
        assert "visual_gate" in result.data
        gate = result.data["visual_gate"]
        assert "action" in gate
        assert "image_path" in gate

    def test_figure_ir_numeric_extraction(self):
        """_figure_ir_numeric: 摊平 series data 为 {label: float|list}."""
        tool = VisualizeTool()
        ir = {
            "series": [
                {"label": "best_fitness", "data": [0.1, 0.2, 0.3]},
                {"label": "single", "data": [5.0]},
                {"label": "empty", "data": []},
                {"label": "nonnum", "data": ["a", "b"]},
            ]
        }
        out = tool._figure_ir_numeric(ir)
        assert out["best_fitness"] == [0.1, 0.2, 0.3]
        assert out["single"] == 5.0
        assert "empty" not in out
        assert "nonnum" not in out

    def test_figure_ir_numeric_malformed(self):
        tool = VisualizeTool()
        assert tool._figure_ir_numeric({}) == {}
        assert tool._figure_ir_numeric({"series": "nope"}) == {}
        assert tool._figure_ir_numeric(None) == {}
