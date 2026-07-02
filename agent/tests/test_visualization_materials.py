"""Phase 4c 可视化升级 (materials) 测试.

5 测:
  1. band_structure action 出 PNG + Arial 字体
  2. dos action 出 PNG
  3. phonon action 出 PNG
  4. structure_3d action 出 PNG
  5. 未知 action 报错
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # 不弹窗

from pathlib import Path

import pytest

from huginn.tools.visualize_tool import VisualizeTool, VisualizeToolInput


def _band_data(n_bands: int = 2, n_k: int = 10) -> list[dict]:
    """造 n 条带, 每条 n_k 个 k 点, 能量从 -2 到 +2 线性."""
    out = []
    for b in range(n_bands):
        kpoints = [i / (n_k - 1) for i in range(n_k)]
        energies = [-2.0 + b * 1.5 + 4.0 * (i / (n_k - 1)) for i in range(n_k)]
        out.append({"kpoints": kpoints, "energies": energies})
    return out


def _dos_data(n: int = 50) -> dict:
    """造 total + orbital_s 两条 DOS 曲线."""
    return {
        "total": [float(i) for i in range(n)],
        "orbital_s": [0.5 * float(i) for i in range(n)],
    }


def _branches(n_br: int = 3, n_q: int = 10) -> list[dict]:
    """造 n 条声子 branch."""
    out = []
    for b in range(n_br):
        qpoints = [i / (n_q - 1) for i in range(n_q)]
        freqs = [-5.0 + b * 3.0 + 10.0 * (i / (n_q - 1)) for i in range(n_q)]
        out.append({"qpoints": qpoints, "frequencies": freqs})
    return out


def _structure() -> dict:
    """Si 双原子 + 1 键."""
    return {
        "lattice": [[5.43, 0, 0], [0, 5.43, 0], [0, 0, 5.43]],
        "species": ["Si", "Si"],
        "coords": [[0.0, 0.0, 0.0], [1.3575, 1.3575, 1.3575]],
        "bonds": [[0, 1]],
    }


class TestMaterialsVisualization:
    """4 个 materials action + 1 个错误路径."""

    @pytest.mark.asyncio
    async def test_band_structure_action(self, tmp_path: Path) -> None:
        out = tmp_path / "band.png"
        tool = VisualizeTool()
        args = VisualizeToolInput(
            action="band_structure",
            output_path=str(out),
            bands_data=_band_data(),
            kpath=["Γ", "X", "M", "Γ"],
            fermi=0.0,
        )
        result = await tool.call(args, context=None)
        assert result.success, f"band_structure failed: {result.error}"
        assert out.exists()
        # 字体规则: Arial
        import matplotlib.pyplot as plt

        assert "Arial" in plt.rcParams["font.family"]

    @pytest.mark.asyncio
    async def test_dos_action(self, tmp_path: Path) -> None:
        out = tmp_path / "dos.png"
        tool = VisualizeTool()
        args = VisualizeToolInput(
            action="dos",
            output_path=str(out),
            dos_data=_dos_data(),
            energy=[-2.0 + 0.1 * i for i in range(50)],
            fermi=0.0,
        )
        result = await tool.call(args, context=None)
        assert result.success, f"dos failed: {result.error}"
        assert out.exists()

    @pytest.mark.asyncio
    async def test_phonon_action(self, tmp_path: Path) -> None:
        out = tmp_path / "phonon.png"
        tool = VisualizeTool()
        args = VisualizeToolInput(
            action="phonon",
            output_path=str(out),
            branches=_branches(),
            kpath=["Γ", "X", "M", "Γ"],
        )
        result = await tool.call(args, context=None)
        assert result.success, f"phonon failed: {result.error}"
        assert out.exists()

    @pytest.mark.asyncio
    async def test_structure_3d_action(self, tmp_path: Path) -> None:
        out = tmp_path / "struct3d.png"
        tool = VisualizeTool()
        args = VisualizeToolInput(
            action="structure_3d",
            output_path=str(out),
            structure=_structure(),
        )
        result = await tool.call(args, context=None)
        assert result.success, f"structure_3d failed: {result.error}"
        assert out.exists()

    @pytest.mark.asyncio
    async def test_unknown_action_returns_error(self, tmp_path: Path) -> None:
        """action 不在 Literal 里会被 Pydantic 拦; 这里直接测 report 路径的 fallback."""
        out = tmp_path / "err.png"
        tool = VisualizeTool()
        # 用 report_data 走 report 路径, action 给一个不在 plotters 里的值
        # Pydantic Literal 会拦, 所以这里绕过 schema 直接改 args
        args = VisualizeToolInput(
            action="benchmark",  # 合法值通过 schema
            output_path=str(out),
            report_data={"results": []},
        )
        args.action = "nonsense"  # 直接改绕过 Pydantic
        result = await tool.call(args, context=None)
        assert not result.success
        assert "Unknown" in (result.error or "") or "Failed" in (result.error or "")
