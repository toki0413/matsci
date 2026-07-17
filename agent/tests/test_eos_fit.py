"""Tests for the EOS fitting pipeline (numerical_tool eos_fit + vasp_tool eos).

审计 14号报告指出旧版用同一公式生成数据再"验证"工具实现 (循环论证),
且 B0P_TRUE=4.0 恰好消掉 BM3 三次项, 无法检测 x³ 系数错误.
本文件用独立教科书公式 (Wikipedia / Poirier / Mouhat-Coudert) 生成参考数据,
B0P_TRUE 改为 5.5 (三次项非零, 强制测试 x³ 系数), 加 Vinet/Murnaghan
交叉一致性测试. 任何对工具 EOS 公式的回归都会被独立参考数据抓住.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from huginn.tools.numerical_tool import NumericalTool
from huginn.tools.sci.numerical_tool import NumericalTool as NumericalToolSci
from huginn.tools.sim.vasp_tool import VaspTool, VaspToolInput
from huginn.types import ToolContext

CTX = ToolContext(session_id="test", workspace=".")

# Known EOS parameters for generating synthetic data
B0_GPA_TRUE = 100.0
V0_TRUE = 50.0
E0_TRUE = -10.0
# B0P_TRUE 从 4.0 改为 5.5: BM3 三次项系数 (B0' x³) 在 B0'=4 时为零,
# 旧测试无法抓 x³ 系数错误 (代码写成 (B0'-4) 而非 B0').
# 5.5 让三次项非零, 任何对 x³ 系数的回归都会被独立参考数据抓住.
B0P_TRUE = 5.5
_EV_PER_A3_TO_GPA = 160.21766208
B0_EV_TRUE = B0_GPA_TRUE / _EV_PER_A3_TO_GPA


def _birch_murnaghan_ref(V, E0, V0, B0, B0p):
    """独立参考实现: 3rd-order Birch-Murnaghan EOS.

    来源: Wikipedia "Birch–Murnaghan isothermal equation of state";
    Poirier "Introduction to the Physics of the Earth's Interior";
    pymatgen.analysis.eos.BirchMurnaghan 实现.

    标准形式 (与 numerical_tool._get_eos_function 故意分开维护,
    避免任何一处错误传染到另一处):
        η = (V0/V)^(1/3),   x = η² - 1
        E = E0 + (9/16) * B0 * V0 * [B0' * x³ + 6 * x²]
    """
    eta = (V0 / V) ** (1.0 / 3.0)
    x = eta ** 2 - 1.0
    return E0 + (9.0 / 16.0) * B0 * V0 * (B0p * x ** 3 + 6.0 * x ** 2)


def _murnaghan_ref(V, E0, V0, B0, B0p):
    """独立参考实现: Murnaghan EOS.

    来源: Wikipedia "Murnaghan equation of state".
        E = E0 + B0*V0/B0p * [ (V0/V)^(B0p-1)/(B0p-1) - B0p/(B0p-1) + V/V0 ]
    """
    r = V0 / V
    return E0 + (B0 * V0 / B0p) * (
        r ** (B0p - 1.0) / (B0p - 1.0)
        - B0p / (B0p - 1.0)
        + V / V0
    )


def _vinet_ref(V, E0, V0, B0, B0p):
    """独立参考实现: Vinet EOS.

    来源: Vinet et al., J. Phys. Chem. Solids 47, 1011 (1986);
    Wikipedia "Vinet equation of state".
        η = (V/V0)^(1/3),  ξ = 1.5*(B0'-1)*(1-η)
        E = E0 + 4*B0*V0/(B0'-1)² * [1 - (1-ξ)*exp(ξ)]
    """
    eta = (V / V0) ** (1.0 / 3.0)
    xi = 1.5 * (B0p - 1.0) * (1.0 - eta)
    return E0 + 4.0 * B0 * V0 / (B0p - 1.0) ** 2 * (
        1.0 - (1.0 - xi) * np.exp(xi)
    )


def _make_outcar_content(volume: float, energy: float) -> str:
    """Bare-minimum OUTCAR text that the regex parser can extract V + E from."""
    return (
        " ENCUT  =  520.0 eV\n"
        " volume of cell :     {vol:.6f}\n"
        "  free  energy   TOTEN  =       {en:.6f} eV\n"
        "reached required accuracy - stopping structural energy minimisation\n"
    ).format(vol=volume, en=energy)


def _disable_accelerators(monkeypatch):
    """Force the pure-Python regex OUTCAR parser."""
    monkeypatch.setattr("huginn.tools.sim.vasp_tool._HAS_HUGINN_EXT", False)
    for mod in ("pymatgen", "pymatgen.io", "pymatgen.io.vasp"):
        monkeypatch.setitem(sys.modules, mod, None)


@pytest.fixture
def num_tool():
    return NumericalTool()


@pytest.fixture
def vasp_tool():
    return VaspTool(vasp_executable=None)


# ── 0. Independent reference vs tool implementation consistency ──────


class TestEosReferenceConsistency:
    """工具实现与独立教科书参考公式的逐点数值一致性.

    这层测试是循环论证破局的关键: 旧的 _birch_murnaghan() 直接复制
    numerical_tool 的错误公式, 工具错测试也错, 永远"通过". 现在用
    独立参考公式, 任何对工具实现的回归都会在这里暴露.
    """

    def test_bm3_tool_matches_reference(self):
        """工具 BM3 实现与独立参考公式逐点一致."""
        volumes = np.linspace(40, 60, 11)
        tool_fn = NumericalToolSci._get_eos_function("birch_murnaghan")
        tool_e = tool_fn(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
        ref_e = _birch_murnaghan_ref(
            volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )
        assert np.allclose(tool_e, ref_e, atol=1e-10), (
            f"BM3 tool != reference, max diff = {np.max(np.abs(tool_e - ref_e))}"
        )

    def test_murnaghan_tool_matches_reference(self):
        volumes = np.linspace(40, 60, 11)
        tool_fn = NumericalToolSci._get_eos_function("murnaghan")
        tool_e = tool_fn(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
        ref_e = _murnaghan_ref(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
        assert np.allclose(tool_e, ref_e, atol=1e-10), (
            f"Murnaghan tool != reference, max diff = {np.max(np.abs(tool_e - ref_e))}"
        )

    def test_vinet_tool_matches_reference(self):
        volumes = np.linspace(40, 60, 11)
        tool_fn = NumericalToolSci._get_eos_function("vinet")
        tool_e = tool_fn(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
        ref_e = _vinet_ref(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
        assert np.allclose(tool_e, ref_e, atol=1e-10), (
            f"Vinet tool != reference, max diff = {np.max(np.abs(tool_e - ref_e))}"
        )

    def test_bm3_minimum_at_v0(self):
        """物理约束: EOS 在 V=V0 处取最小值 E=E0, 导数为 0."""
        volumes = np.array([V0_TRUE])
        e_at_v0 = _birch_murnaghan_ref(
            volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )[0]
        assert math.isclose(e_at_v0, E0_TRUE, abs_tol=1e-12), (
            f"BM3 应在 V=V0 时 E=E0, got E={e_at_v0}"
        )
        # 数值导数 ≈ 0
        dv = 0.01
        e_plus = _birch_murnaghan_ref(
            np.array([V0_TRUE + dv]), E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )[0]
        e_minus = _birch_murnaghan_ref(
            np.array([V0_TRUE - dv]), E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )[0]
        slope = (e_plus - e_minus) / (2 * dv)
        assert abs(slope) < 1e-6, f"BM3 在 V0 处导数应 ≈ 0, got {slope}"


# ── 1. Pure-math EOS fitting ──────────────────────────────────────────

@pytest.mark.asyncio
class TestEosFitNumerical:
    async def test_birch_murnaghan_recovery(self, num_tool):
        # 用独立参考公式生成数据, 避免循环论证
        volumes = np.linspace(40, 60, 9)
        energies = _birch_murnaghan_ref(
            volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )
        result = await num_tool.call({
            "action": "eos_fit",
            "volumes": volumes.tolist(),
            "energies": energies.tolist(),
            "eos_type": "birch_murnaghan",
        })
        assert result.success
        d = result.data
        assert math.isclose(d["E0"], E0_TRUE, abs_tol=1e-6)
        assert math.isclose(d["V0"], V0_TRUE, abs_tol=1e-4)
        assert math.isclose(d["B0_GPa"], B0_GPA_TRUE, abs_tol=0.5)
        # B0P_TRUE=5.5 让三次项非零, 强制验证 x³ 系数
        assert math.isclose(d["B0_prime"], B0P_TRUE, abs_tol=0.05), (
            f"B0' recovery failed: got {d['B0_prime']}, expected {B0P_TRUE}. "
            f"旧代码 (B0p-4) 系数会让 B0'→4 当真值=5.5."
        )
        assert d["r_squared"] > 0.9999

    async def test_fit_curve_and_input_data_present(self, num_tool):
        volumes = np.linspace(42, 58, 7)
        energies = _birch_murnaghan_ref(
            volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )
        result = await num_tool.call({
            "action": "eos_fit",
            "volumes": volumes.tolist(),
            "energies": energies.tolist(),
        })
        assert result.success
        fc = result.data["fit_curve"]
        assert len(fc["volumes"]) == 200
        assert len(fc["energies"]) == 200
        inp = result.data["input_data"]
        assert len(inp["volumes"]) == 7

    async def test_murnaghan_recovery(self, num_tool):
        # 用独立参考公式, 不再用工具自己的 _get_eos_function 生成数据
        volumes = np.linspace(40, 60, 9)
        energies = _murnaghan_ref(
            volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )
        result = await num_tool.call({
            "action": "eos_fit",
            "volumes": volumes.tolist(),
            "energies": energies.tolist(),
            "eos_type": "murnaghan",
        })
        assert result.success
        assert math.isclose(result.data["E0"], E0_TRUE, abs_tol=1e-6)
        assert math.isclose(result.data["V0"], V0_TRUE, abs_tol=1e-4)
        assert math.isclose(result.data["B0_GPa"], B0_GPA_TRUE, abs_tol=0.5)

    async def test_vinet_recovery(self, num_tool):
        volumes = np.linspace(40, 60, 9)
        energies = _vinet_ref(
            volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )
        result = await num_tool.call({
            "action": "eos_fit",
            "volumes": volumes.tolist(),
            "energies": energies.tolist(),
            "eos_type": "vinet",
        })
        assert result.success
        assert math.isclose(result.data["E0"], E0_TRUE, abs_tol=1e-6)
        assert math.isclose(result.data["V0"], V0_TRUE, abs_tol=1e-4)
        assert math.isclose(result.data["B0_GPa"], B0_GPA_TRUE, abs_tol=0.5)

    async def test_with_noise(self, num_tool):
        rng = np.random.default_rng(42)
        volumes = np.linspace(42, 58, 11)
        energies = _birch_murnaghan_ref(
            volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )
        energies += rng.normal(0, 0.001, len(energies))
        result = await num_tool.call({
            "action": "eos_fit",
            "volumes": volumes.tolist(),
            "energies": energies.tolist(),
        })
        assert result.success
        assert math.isclose(result.data["V0"], V0_TRUE, abs_tol=0.5)
        assert math.isclose(result.data["B0_GPa"], B0_GPA_TRUE, abs_tol=5.0)
        assert result.data["r_squared"] > 0.99

    async def test_too_few_points(self, num_tool):
        result = await num_tool.call({
            "action": "eos_fit",
            "volumes": [45.0, 50.0, 55.0],
            "energies": [-9.5, -10.0, -9.5],
        })
        assert not result.success
        assert "4 points" in result.error

    async def test_missing_data(self, num_tool):
        result = await num_tool.call({"action": "eos_fit"})
        assert not result.success

    async def test_mismatched_lengths(self, num_tool):
        result = await num_tool.call({
            "action": "eos_fit",
            "volumes": [45.0, 50.0, 55.0, 60.0],
            "energies": [-9.5, -10.0, -9.5],
        })
        assert not result.success
        assert "same length" in result.error


# ── 2. VASP eos action ───────────────────────────────────────────────

@pytest.mark.asyncio
class TestVaspEosAction:
    async def test_eos_from_mock_outcars(self, vasp_tool, tmp_path, monkeypatch):
        _disable_accelerators(monkeypatch)
        volumes = np.linspace(42, 58, 7)
        energies = _birch_murnaghan_ref(
            volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )
        for i, (v, e) in enumerate(zip(volumes, energies)):
            sub = tmp_path / f"vol_{i:02d}"
            sub.mkdir()
            (sub / "OUTCAR").write_text(_make_outcar_content(v, e))

        result = await vasp_tool.call(
            VaspToolInput(action="eos", working_dir=str(tmp_path)), CTX
        )
        assert result.success
        assert result.data["n_points"] == 7
        ev = result.data["ev_points"]
        assert len(ev) == 7
        vols = [p["volume"] for p in ev]
        assert vols == sorted(vols)
        eos = result.data["eos_fit"]
        assert math.isclose(eos["V0"], V0_TRUE, abs_tol=0.5)
        assert math.isclose(eos["B0_GPa"], B0_GPA_TRUE, abs_tol=5.0)

    async def test_eos_insufficient_points(self, vasp_tool, tmp_path, monkeypatch):
        _disable_accelerators(monkeypatch)
        for i in range(2):
            sub = tmp_path / f"vol_{i}"
            sub.mkdir()
            (sub / "OUTCAR").write_text(_make_outcar_content(45 + i, -9.0 - i))

        result = await vasp_tool.call(
            VaspToolInput(action="eos", working_dir=str(tmp_path)), CTX
        )
        assert not result.success
        assert "4 points" in result.error

    async def test_eos_skips_dirs_without_outcar(self, vasp_tool, tmp_path, monkeypatch):
        _disable_accelerators(monkeypatch)
        volumes = np.linspace(42, 58, 7)
        energies = _birch_murnaghan_ref(
            volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )
        for i, (v, e) in enumerate(zip(volumes, energies)):
            sub = tmp_path / f"vol_{i:02d}"
            sub.mkdir()
            (sub / "OUTCAR").write_text(_make_outcar_content(v, e))
        (tmp_path / "junk_no_outcar").mkdir()
        (tmp_path / "junk_no_outcar" / "readme.txt").write_text("nothing useful")

        result = await vasp_tool.call(
            VaspToolInput(action="eos", working_dir=str(tmp_path)), CTX
        )
        assert result.success
        assert result.data["n_points"] == 7

    async def test_eos_with_eos_type_murnaghan(self, vasp_tool, tmp_path, monkeypatch):
        _disable_accelerators(monkeypatch)
        volumes = np.linspace(42, 58, 7)
        energies = _murnaghan_ref(
            volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE
        )
        for i, (v, e) in enumerate(zip(volumes, energies)):
            sub = tmp_path / f"vol_{i:02d}"
            sub.mkdir()
            (sub / "OUTCAR").write_text(_make_outcar_content(v, e))

        result = await vasp_tool.call(
            VaspToolInput(
                action="eos", working_dir=str(tmp_path), eos_type="murnaghan"
            ),
            CTX,
        )
        assert result.success
        assert result.data["eos_fit"]["eos_type"] == "murnaghan"
