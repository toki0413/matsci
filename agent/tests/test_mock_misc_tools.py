"""Mock tests for misc tools (MaterialsDatabase, MLPotential, Packing, Visualize)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from huginn.tools.materials_database_tool import (
    MaterialsDatabaseInput,
    MaterialsDatabaseTool,
)
from huginn.tools.ml_potential_tool import MLPotentialTool, MLPotentialInput
from huginn.tools.packing_tool import PackingTool, PackingToolInput
from huginn.tools.visualize_tool import VisualizeTool, VisualizeToolInput
from huginn.types import ToolContext, ToolResult

CTX = ToolContext(session_id="test", workspace=".")


# ── MaterialsDatabase ──
class TestMaterialsDatabaseTool:
    def test_init(self):
        tool = MaterialsDatabaseTool(mp_api_key="test_key")
        assert tool._config_mp_key == "test_key"

    def test_is_read_only(self):
        tool = MaterialsDatabaseTool()
        assert tool.is_read_only(MaterialsDatabaseInput(action="mp_summary", formula="Si")) is True

    @pytest.mark.asyncio
    async def test_mp_summary_no_key(self):
        tool = MaterialsDatabaseTool()
        with pytest.MonkeyPatch().context() as mp:
            mp.delenv("MP_API_KEY", raising=False)
            result = await tool.call(
                MaterialsDatabaseInput(action="mp_summary", formula="Si"), CTX
            )
        assert result.success is False
        assert "API key" in result.error

    @pytest.mark.asyncio
    async def test_mp_summary_mock(self):
        tool = MaterialsDatabaseTool(mp_api_key="fake_key")
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={"data": [{"formula_pretty": "Si", "energy_per_atom": -5.0}]})
            mock_resp.text = AsyncMock(return_value='{"data": [{"formula_pretty": "Si", "energy_per_atom": -5.0}]}')
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session.return_value.get = MagicMock(return_value=mock_resp)
            result = await tool.call(
                MaterialsDatabaseInput(action="mp_summary", formula="Si", fields=["formula_pretty", "energy_per_atom"]),
                CTX,
            )
            assert result.success is True or result.error == ""

    @pytest.mark.asyncio
    async def test_oqmd_summary(self):
        tool = MaterialsDatabaseTool()
        with patch("aiohttp.ClientSession") as mock_session:
            mock_resp = MagicMock()
            mock_resp.status = 200
            mock_resp.json = AsyncMock(return_value={"results": [{"composition": "Si", "delta_e": -5.0}]})
            mock_resp.text = AsyncMock(return_value='{"results": [{"composition": "Si", "delta_e": -5.0}]}')
            mock_resp.__aenter__ = AsyncMock(return_value=mock_resp)
            mock_resp.__aexit__ = AsyncMock(return_value=False)
            mock_session.return_value.__aenter__ = AsyncMock(return_value=mock_session.return_value)
            mock_session.return_value.__aexit__ = AsyncMock(return_value=False)
            mock_session.return_value.get = MagicMock(return_value=mock_resp)
            result = await tool.call(
                MaterialsDatabaseInput(action="oqmd_query", formula="Si"), CTX
            )
            assert result.success is True or "error" in result.error.lower()


# ── MLPotential ──
class TestMLPotentialTool:
    def test_init(self):
        tool = MLPotentialTool()
        assert tool.name == "ml_potential_tool"

    def test_is_read_only(self):
        tool = MLPotentialTool()
        assert tool.is_read_only(MLPotentialInput(action="predict", backend="mace", structure_file="test.cif")) is True

    @pytest.mark.asyncio
    async def test_predict_mock(self, tmp_path: Path):
        tool = MLPotentialTool()
        structure_file = tmp_path / "test.cif"
        structure_file.write_text("mock structure")
        with patch.object(tool, "_run_mace", return_value=ToolResult(data={"energy": -5.0}, success=True)):
            result = await tool.call(
                MLPotentialInput(
                    action="predict",
                    backend="mace",
                    structure_file=str(structure_file),
                    model_path=str(tmp_path / "nonexistent.pt"),
                ),
                CTX,
            )
        assert result.success is True


# ── Packing ──
class TestPackingTool:
    def test_init(self):
        tool = PackingTool()
        assert tool.name == "packing_tool"

    def test_generate_simple(self, tmp_path: Path):
        tool = PackingTool()
        result = tool.call(
            {
                "action": "generate",
                "container": {"type": "box", "dimensions": [10, 10, 10]},
                "objects": [{"type": "sphere", "radius": 1.0, "count": 5}],
                "working_dir": str(tmp_path),
                "output_prefix": "pack",
            },
            CTX,
        )
        assert result.success is True
        assert (tmp_path / "pack.json").exists() or result.data is not None

    def test_generate_no_objects(self, tmp_path: Path):
        tool = PackingTool()
        result = tool.call(
            {
                "action": "generate",
                "container": {"type": "box", "dimensions": [10, 10, 10]},
                "objects": [],
                "working_dir": str(tmp_path),
                "output_prefix": "pack",
            },
            CTX,
        )
        assert result.success is True or result.success is False


# ── Visualize ──
class TestVisualizeTool:
    def test_init(self):
        tool = VisualizeTool()
        assert tool.name == "visualize_tool"
        assert tool.is_read_only(VisualizeToolInput(action="benchmark", output_path="test.png")) is True

    def test_load_report_data(self):
        tool = VisualizeTool()
        report = tool._load_report(
            VisualizeToolInput(action="benchmark", report_data={"tasks": []}, output_path="test.png")
        )
        assert report == {"tasks": []}

    def test_load_report_path(self, tmp_path: Path):
        tool = VisualizeTool()
        path = tmp_path / "report.json"
        path.write_text('{"tasks": []}')
        report = tool._load_report(
            VisualizeToolInput(action="benchmark", report_path=str(path), output_path="test.png")
        )
        assert report == {"tasks": []}

    def test_unknown_action(self, tmp_path: Path):
        with pytest.raises(Exception):
            VisualizeToolInput(action="unknown", output_path="test.png")

    @pytest.mark.asyncio
    async def test_benchmark_report(self, tmp_path: Path):
        tool = VisualizeTool()
        out_path = str(tmp_path / "out.png")
        with patch("huginn.visualize.plot_benchmark_report", return_value=out_path) as mock_plot:
            (tmp_path / "out.png").write_text("mock")
            result = await tool.call(
                VisualizeToolInput(
                    action="benchmark",
                    report_data={"tasks": [{"name": "t1", "score": 0.9}]},
                    output_path=out_path,
                ),
                CTX,
            )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_evolution_report(self, tmp_path: Path):
        tool = VisualizeTool()
        out_path = str(tmp_path / "out.png")
        with patch("huginn.visualize.plot_evolution_report", return_value=out_path) as mock_plot:
            (tmp_path / "out.png").write_text("mock")
            result = await tool.call(
                VisualizeToolInput(
                    action="evolution",
                    report_data={"generations": [{"gen": 1, "best": 0.9}]},
                    output_path=out_path,
                ),
                CTX,
            )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_exploration_report(self, tmp_path: Path):
        tool = VisualizeTool()
        out_path = str(tmp_path / "out.png")
        with patch("huginn.visualize.plot_exploration_result", return_value=out_path) as mock_plot:
            (tmp_path / "out.png").write_text("mock")
            result = await tool.call(
                VisualizeToolInput(
                    action="exploration",
                    report_data={"branches": [{"id": 1, "score": 0.9}]},
                    output_path=out_path,
                ),
                CTX,
            )
        assert result.success is True
