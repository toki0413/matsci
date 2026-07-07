"""Tests for the EOS fitting pipeline (numerical_tool eos_fit + vasp_tool eos)."""

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
B0P_TRUE = 4.0
_EV_PER_A3_TO_GPA = 160.21766208
B0_EV_TRUE = B0_GPA_TRUE / _EV_PER_A3_TO_GPA


def _birch_murnaghan(V, E0, V0, B0, B0p):
    """3rd-order BM EOS (same formula as the tool)."""
    eta = (V / V0) ** (1.0 / 3.0)
    x = eta ** 2 - 1.0
    return E0 + (9.0 / 16.0) * B0 * V0 * (x ** 3 * (B0p - 4.0) + 2.0 * x ** 2)


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


# ── 1. Pure-math EOS fitting ──────────────────────────────────────────

@pytest.mark.asyncio
class TestEosFitNumerical:
    async def test_birch_murnaghan_recovery(self, num_tool):
        volumes = np.linspace(40, 60, 9)
        energies = _birch_murnaghan(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
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
        assert math.isclose(d["B0_prime"], B0P_TRUE, abs_tol=0.01)
        assert d["r_squared"] > 0.9999

    async def test_fit_curve_and_input_data_present(self, num_tool):
        volumes = np.linspace(42, 58, 7)
        energies = _birch_murnaghan(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
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
        eos_fn = NumericalToolSci._get_eos_function("murnaghan")
        volumes = np.linspace(40, 60, 9)
        energies = eos_fn(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
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
        eos_fn = NumericalToolSci._get_eos_function("vinet")
        volumes = np.linspace(40, 60, 9)
        energies = eos_fn(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
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
        energies = _birch_murnaghan(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
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
        energies = _birch_murnaghan(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
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
        energies = _birch_murnaghan(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
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
        eos_fn = NumericalToolSci._get_eos_function("murnaghan")
        volumes = np.linspace(42, 58, 7)
        energies = eos_fn(volumes, E0_TRUE, V0_TRUE, B0_EV_TRUE, B0P_TRUE)
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
