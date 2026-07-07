"""Tests for the cross-domain pipelines:

1. XRD inverse design — given target 2θ peaks, recover the lattice constant.
2. sim -> GP UQ hint — VASP / LAMMPS results carry a uq_hint nudging the agent
   to chain into numerical_tool's Gaussian Process action.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from huginn.tools.sci.xrd_sim_tool import XrdSimTool
from huginn.tools.sim.lammps_tool import LammpsTool, LammpsToolInput
from huginn.tools.sim.vasp_tool import VaspTool, VaspToolInput
from huginn.types import ToolContext

# Cu Kα, the tool default — kept explicit so the Bragg math below stays readable.
WAVELENGTH = 1.5406

# 立方晶系逆设计匹配的前三个反射, 必须和工具里的 _CUBIC_HKLS 一致
CUBIC_HKLS = [(1, 1, 1), (2, 0, 0), (2, 2, 0)]


def _two_theta(a: float, hkl: tuple[int, int, int], wl: float = WAVELENGTH) -> float:
    """Bragg 定律算 2θ, 和工具实现用同一套公式, 保证自洽."""
    h, k, l = hkl
    d = a / math.sqrt(h * h + k * k + l * l)
    return math.degrees(2.0 * math.asin(wl / (2.0 * d)))


@pytest.fixture
def xrd_tool():
    return XrdSimTool()


@pytest.fixture
def context(tmp_path):
    return ToolContext(session_id="test", workspace=str(tmp_path))


@pytest.mark.asyncio
class TestXrdInverseDesign:
    async def test_inverse_design_recovers_lattice_constant(self, xrd_tool, context):
        """自洽目标峰: 从已知 a 生成峰位, 再反推回来应该能找回 a."""
        pytest.importorskip("scipy", reason="scipy required for inverse_design")
        a_true = 3.15
        targets = [round(_two_theta(a_true, hkl), 4) for hkl in CUBIC_HKLS]

        result = await xrd_tool.call(
            {"action": "inverse_design", "target_peaks": targets, "wavelength": WAVELENGTH},
            context,
        )
        assert result.success, f"inverse_design failed: {result.error}"
        a_recovered = result.data["lattice_a"]
        # 自洽目标误差应趋近于 0, a 也应回到真值附近
        assert abs(a_recovered - a_true) < 0.02, (
            f"recovered a={a_recovered} far from true a={a_true}"
        )
        assert result.data["match_error"] < 0.1
        assert result.data["lattice_system"] == "cubic"
        # 全套晶格参数 [a, b, c, α, β, γ]
        lp = result.data["lattice_parameters"]
        assert lp[0] == lp[1] == lp[2]
        assert lp[3:] == [90.0, 90.0, 90.0]
        assert result.data["n_peaks"] == 3
        assert len(result.data["simulated_peaks"]) == 3

    async def test_inverse_design_respects_initial_guess(self, xrd_tool, context):
        """lattice_params_guess 应该作为优化的起点, 不影响最终收敛结果."""
        pytest.importorskip("scipy", reason="scipy required for inverse_design")
        a_true = 5.0
        targets = [round(_two_theta(a_true, hkl), 4) for hkl in CUBIC_HKLS]

        result = await xrd_tool.call(
            {
                "action": "inverse_design",
                "target_peaks": targets,
                "wavelength": WAVELENGTH,
                "lattice_params_guess": [3.0, 3.0, 3.0, 90.0, 90.0, 90.0],
            },
            context,
        )
        assert result.success
        assert abs(result.data["lattice_a"] - a_true) < 0.02

    async def test_inverse_design_with_literal_peaks(self, xrd_tool, context):
        """用户给的 [28.4, 47.3, 56.1] 这组峰对 (111)/(200)/(220) 并不自洽,
        优化器应收敛到唯一折中解 a≈4.52 (与初值无关)."""
        pytest.importorskip("scipy", reason="scipy required for inverse_design")
        result = await xrd_tool.call(
            {"action": "inverse_design", "target_peaks": [28.4, 47.3, 56.1]},
            context,
        )
        assert result.success, f"inverse_design failed: {result.error}"
        a = result.data["lattice_a"]
        # 这组峰的最优折中解 (Nelder-Mead 从多个初值都收敛到这里)
        assert 4.4 < a < 4.7, f"unexpected a={a} for literal peaks"
        assert result.data["n_peaks"] == 3
        assert result.data["match_error"] >= 0.0
        # 模拟峰列表带 hkl
        for peak in result.data["simulated_peaks"]:
            assert "hkl" in peak and "two_theta" in peak

    async def test_inverse_design_missing_target_peaks_fails_gracefully(self, xrd_tool, context):
        """缺 target_peaks 应优雅报错, 不抛异常."""
        result = await xrd_tool.call(
            {"action": "inverse_design"}, context
        )
        assert not result.success
        assert "target_peaks" in result.error.lower()


@pytest.mark.asyncio
class TestVaspUqHint:
    async def test_vasp_result_contains_uq_hint(self, tmp_path):
        """VASP (mock 模式) 结果应带 uq_hint, 指向 numerical_tool 的 gp action."""
        tool = VaspTool(vasp_executable=None)
        (tmp_path / "POSCAR").write_text(
            "Si\n1.0\n5.0 0 0\n0 5.0 0\n0 0 5.0\nSi\n1\n0 0 0\n"
        )
        (tmp_path / "INCAR").write_text("ENCUT = 520\n")
        ctx = ToolContext(session_id="test", workspace=str(tmp_path))

        result = await tool.call(
            VaspToolInput(action="relax", working_dir=str(tmp_path)), ctx
        )
        assert result.success
        assert result.data["status"] == "mock"
        hint = result.data["uq_hint"]
        assert hint["tool"] == "numerical_tool"
        assert hint["action"] == "gp"
        assert "Gaussian" in hint["suggestion"]
        assert "X" in hint["data_mapping"] and "y" in hint["data_mapping"]


@pytest.mark.asyncio
class TestLammpsUqHint:
    async def test_lammps_result_contains_uq_hint(self, tmp_path):
        """LAMMPS (mock 模式) 结果应带 uq_hint, 建议用 GP 拟合 MSD-vs-time."""
        tool = LammpsTool(lammps_executable=None)
        script = tmp_path / "lmp.in"
        script.write_text("units metal\natom_style atomic\n")
        ctx = ToolContext(session_id="test", workspace=str(tmp_path))

        result = await tool.call(
            LammpsToolInput(
                action="run", working_dir=str(tmp_path), input_script=str(script)
            ),
            ctx,
        )
        assert result.success
        hint = result.data["uq_hint"]
        assert hint["tool"] == "numerical_tool"
        assert hint["action"] == "gp"
        # 建议应该提到 MSD / 扩散相关的内容
        assert "msd" in hint["suggestion"].lower() or "diffusion" in hint["suggestion"].lower()
        assert hint["data_mapping"]["y"] == "msd (mean squared displacement)"
