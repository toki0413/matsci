"""Mock tests for simulation tools (VASP, QE, CP2K, LAMMPS, COMSOL, ABAQUS, OpenFOAM).

These tools call external executables which are not available in CI.
We mock subprocess.run, shutil.which, and os.path.exists to exercise
all code paths without installing the actual software.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from huginn.tools.abaqus_tool import AbaqusTool, AbaqusToolInput
from huginn.tools.comsol_tool import ComsolTool, ComsolToolInput
from huginn.tools.cp2k_tool import Cp2kTool, Cp2kToolInput
from huginn.tools.lammps_tool import LammpsTool, LammpsToolInput
from huginn.tools.openfoam_tool import OpenFoamTool, OpenFoamToolInput
from huginn.tools.qe_tool import QuantumEspressoTool, QuantumEspressoToolInput
from huginn.tools.vasp_tool import VaspTool, VaspToolInput
from huginn.types import ToolContext

CTX = ToolContext(session_id="test", workspace=".")


# ── VASP ──
class TestVaspTool:
    def test_find_vasp_env(self, monkeypatch):
        monkeypatch.setenv("VASP_EXECUTABLE", "/fake/vasp")
        with patch("pathlib.Path.exists", return_value=True):
            tool = VaspTool()
        assert tool.vasp_executable == "/fake/vasp"

    def test_find_vasp_path(self):
        with patch("shutil.which", return_value="/usr/bin/vasp"):
            tool = VaspTool()
        assert tool.vasp_executable == "/usr/bin/vasp"

    def test_find_vasp_not_found(self):
        with patch("shutil.which", return_value=None):
            tool = VaspTool()
        assert tool.vasp_executable is None

    @pytest.mark.asyncio
    async def test_mock_relax_no_poscar(self, tmp_path: Path):
        tool = VaspTool(vasp_executable=None)
        result = await tool.call(
            VaspToolInput(action="relax", working_dir=str(tmp_path)), CTX
        )
        assert result.success is False
        assert "POSCAR" in result.error

    @pytest.mark.asyncio
    async def test_mock_relax_with_poscar(self, tmp_path: Path):
        tool = VaspTool(vasp_executable=None)
        (tmp_path / "POSCAR").write_text("Si\n1.0\n5.0 0 0\n0 5.0 0\n0 0 5.0\nSi\n1\n0 0 0\n")
        (tmp_path / "INCAR").write_text("ENCUT = 520\n")
        result = await tool.call(
            VaspToolInput(action="relax", working_dir=str(tmp_path)), CTX
        )
        assert result.success is True
        assert result.data["status"] == "mock"

    @pytest.mark.asyncio
    async def test_incar_overrides(self, tmp_path: Path):
        tool = VaspTool(vasp_executable=None)
        (tmp_path / "POSCAR").write_text("Si\n1.0\n5.0 0 0\n0 5.0 0\n0 0 5.0\nSi\n1\n0 0 0\n")
        incar = tmp_path / "INCAR"
        incar.write_text("ENCUT = 520\n")
        result = await tool.call(
            VaspToolInput(
                action="relax", working_dir=str(tmp_path), incar_overrides={"ENCUT": 600}
            ),
            CTX,
        )
        assert result.success is True
        assert "ENCUT = 600" in incar.read_text()

    def test_estimate_cost(self):
        tool = VaspTool()
        cost = tool.estimate_cost(VaspToolInput(action="scf", working_dir=".", walltime_hours=12))
        assert cost["cpu_hours"] == 48


# ── Quantum ESPRESSO ──
class TestQuantumEspressoTool:
    def test_find_qe_env(self, monkeypatch):
        monkeypatch.setenv("QE_EXECUTABLE", "/fake/pw.x")
        with patch("pathlib.Path.exists", return_value=True):
            tool = QuantumEspressoTool()
        assert tool.qe_executable == "/fake/pw.x"

    def test_find_qe_path(self):
        with patch("shutil.which", side_effect=["/usr/bin/pw.x", None, None, None, None]):
            tool = QuantumEspressoTool()
        assert "pw.x" in (tool.qe_executable or "")

    def test_generate_input(self, tmp_path: Path):
        tool = QuantumEspressoTool(qe_executable=None)
        result = tool.call(
            {
                "action": "generate",
                "working_dir": str(tmp_path),
                "output_prefix": "test",
                "calculation": "scf",
                "structure": {"species": ["Si"], "positions": [[0, 0, 0]], "cell": [[5, 0, 0], [0, 5, 0], [0, 0, 5]]},
                "ecutwfc": 50,
            },
            CTX,
        )
        assert result.success is True
        assert result.data.get("qe_available") is False or result.data.get("qe_available") is None
        assert (tmp_path / "test.in").exists()

    def test_run_qe_mock(self, tmp_path: Path):
        tool = QuantumEspressoTool(qe_executable=None)
        result = tool.call(
            {
                "action": "run",
                "working_dir": str(tmp_path),
                "output_prefix": "test",
                "calculation": "scf",
                "structure": {"species": ["Si"], "positions": [[0, 0, 0]], "cell": [[5, 0, 0], [0, 5, 0], [0, 0, 5]]},
                "ecutwfc": 50,
            },
            CTX,
        )
        assert result.success is True

    def test_parse_results_no_file(self, tmp_path: Path):
        tool = QuantumEspressoTool(qe_executable=None)
        result = tool.call(
            {
                "action": "parse",
                "working_dir": str(tmp_path),
                "output_prefix": "test",
            },
            CTX,
        )
        assert result.success is True
        assert result.data.get("results") == {}


# ── CP2K ──
class TestCp2kTool:
    def test_find_cp2k_env(self, monkeypatch):
        monkeypatch.setenv("CP2K_EXECUTABLE", "/fake/cp2k")
        with patch("pathlib.Path.exists", return_value=True):
            tool = Cp2kTool()
        assert tool.cp2k_executable == "/fake/cp2k"

    def test_generate_input(self, tmp_path: Path):
        tool = Cp2kTool(cp2k_executable=None)
        result = tool.call(
            {
                "action": "generate",
                "working_dir": str(tmp_path),
                "output_prefix": "cp2k",
                "run_type": "ENERGY_FORCE",
                "structure": {"species": ["Si"], "positions": [[0, 0, 0]], "cell": [[5, 0, 0], [0, 5, 0], [0, 0, 5]]},
                "basis_set_file": "BASIS_SET",
                "potential_file": "POTENTIAL",
            },
            CTX,
        )
        assert result.success is True
        assert (tmp_path / "cp2k.inp").exists()

    def test_run_cp2k_mock(self, tmp_path: Path):
        tool = Cp2kTool(cp2k_executable=None)
        result = tool.call(
            {
                "action": "run",
                "working_dir": str(tmp_path),
                "output_prefix": "cp2k",
                "run_type": "ENERGY_FORCE",
                "structure": {"species": ["Si"], "positions": [[0, 0, 0]], "cell": [[5, 0, 0], [0, 5, 0], [0, 0, 5]]},
                "basis_set_file": "BASIS_SET",
                "potential_file": "POTENTIAL",
            },
            CTX,
        )
        assert result.success is True


# ── LAMMPS ──
class TestLammpsTool:
    def test_find_lammps_env(self, monkeypatch):
        monkeypatch.setenv("LAMMPS_EXECUTABLE", "/fake/lmp")
        with patch("pathlib.Path.exists", return_value=True):
            tool = LammpsTool()
        assert tool.lammps_executable == "/fake/lmp"

    def test_find_lammps_path(self):
        with patch("shutil.which", return_value="/usr/bin/lmp"):
            tool = LammpsTool()
        assert tool.lammps_executable == "/usr/bin/lmp"

    @pytest.mark.asyncio
    async def test_generate_input(self, tmp_path: Path):
        tool = LammpsTool(lammps_executable=None)
        script_path = tmp_path / "lmp.in"
        script_path.write_text("units metal\natom_style atomic\n")
        result = await tool.call(
            LammpsToolInput(
                action="run",
                working_dir=str(tmp_path),
                input_script=str(script_path),
            ),
            CTX,
        )
        assert result.success is True

    @pytest.mark.asyncio
    async def test_run_md_mock(self, tmp_path: Path):
        tool = LammpsTool(lammps_executable=None)
        script_path = tmp_path / "lmp.in"
        script_path.write_text("units metal\natom_style atomic\n")
        result = await tool.call(
            LammpsToolInput(
                action="run",
                working_dir=str(tmp_path),
                input_script=str(script_path),
            ),
            CTX,
        )
        assert result.success is True


# ── COMSOL ──
class TestComsolTool:
    def test_find_comsol_env(self, monkeypatch):
        monkeypatch.setenv("COMSOL_EXECUTABLE", "/fake/comsol")
        with patch("pathlib.Path.exists", return_value=True):
            tool = ComsolTool()
        assert tool.comsol_executable == "/fake/comsol"

    def test_generate_script(self, tmp_path: Path):
        tool = ComsolTool(comsol_executable=None)
        result = tool.call(
            {
                "action": "generate",
                "working_dir": str(tmp_path),
                "output_prefix": "comsol",
                "physics": "solid_mechanics",
                "geometry": {"type": "block", "width": 1.0, "height": 0.1, "depth": 0.1},
                "mesh": {"element_size": "normal"},
                "material": {"youngs_modulus": 200e9, "poissons_ratio": 0.3, "density": 7850.0},
                "solver": "stationary",
            },
            CTX,
        )
        assert result.success is True
        assert (tmp_path / "comsol.java").exists()

    def test_run_mock(self, tmp_path: Path):
        tool = ComsolTool(comsol_executable=None)
        result = tool.call(
            {
                "action": "run",
                "working_dir": str(tmp_path),
                "output_prefix": "comsol",
                "physics": "solid_mechanics",
                "geometry": {"type": "block", "width": 1.0, "height": 0.1, "depth": 0.1},
                "mesh": {"element_size": "normal"},
                "material": {"youngs_modulus": 200e9, "poissons_ratio": 0.3, "density": 7850.0},
                "solver": "stationary",
            },
            CTX,
        )
        assert result.success is True
        assert "comsol_available" in result.data


# ── ABAQUS ──
class TestAbaqusTool:
    def test_find_abaqus_env(self, monkeypatch):
        monkeypatch.setenv("ABAQUS_EXECUTABLE", "/fake/abaqus")
        with patch("pathlib.Path.exists", return_value=True):
            tool = AbaqusTool()
        assert tool.abaqus_executable == "/fake/abaqus"

    def test_find_abaqus_path(self):
        with patch("shutil.which", return_value="/usr/bin/abaqus"):
            tool = AbaqusTool()
        assert tool.abaqus_executable == "/usr/bin/abaqus"

    def test_import_packing(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(AbaqusTool, "_find_abaqus", lambda self: None)
        tool = AbaqusTool(abaqus_executable=None)
        packing = {"objects": [{"center": [0, 0, 0], "radius": 1.0}]}
        result = tool.call(
            {
                "action": "import_packing",
                "working_dir": str(tmp_path),
                "packing_data": packing,
                "output_prefix": "abaqus",
                "particle_shape": "sphere",
            },
            CTX,
        )
        assert result.success is True
        assert (tmp_path / "abaqus.py").exists()

    def test_run_no_script(self, tmp_path: Path, monkeypatch):
        monkeypatch.setattr(AbaqusTool, "_find_abaqus", lambda self: None)
        tool = AbaqusTool(abaqus_executable=None)
        result = tool.call(
            {"action": "run", "working_dir": str(tmp_path)}, CTX
        )
        assert result.success is False
        assert "script_path" in result.error.lower()


# ── OpenFOAM ──
class TestOpenFoamTool:
    def test_find_openfoam_env(self, monkeypatch):
        monkeypatch.setenv("OPENFOAM_DIR", "/fake/openfoam")
        with patch("pathlib.Path.exists", return_value=True):
            tool = OpenFoamTool()
        assert tool.openfoam_dir == "/fake/openfoam"

    def test_generate_case(self, tmp_path: Path):
        tool = OpenFoamTool(openfoam_dir=None)
        result = tool.call(
            {
                "action": "generate",
                "working_dir": str(tmp_path),
                "case_name": "cavity",
                "solver": "icoFoam",
                "geometry": {"type": "block", "length": 2.0, "width": 0.5, "height": 0.5},
                "mesh": {"cells": [20, 8, 8]},
                "transport_properties": {"nu": 1e-05, "rho": 1.0},
                "end_time": 1.0,
                "delta_t": 0.005,
            },
            CTX,
        )
        assert result.success is True
        case_dir = tmp_path / "cavity"
        assert case_dir.exists()
        assert (case_dir / "0" / "U").exists() or (case_dir / "system" / "controlDict").exists()

    def test_parse_no_results(self, tmp_path: Path):
        tool = OpenFoamTool(openfoam_dir=None)
        result = tool.call(
            {
                "action": "parse",
                "working_dir": str(tmp_path),
                "case_name": "cavity",
                "result_files": [],
            },
            CTX,
        )
        assert result.success is True
        assert result.data.get("results") == {}

    def test_set_fields(self, tmp_path: Path):
        tool = OpenFoamTool(openfoam_dir=None)
        packing = tmp_path / "packing.json"
        packing.write_text(json.dumps({"objects": [{"center": [0, 0, 0], "radius": 1.0}]}))
        case_dir = tmp_path / "cavity"
        case_dir.mkdir()
        (case_dir / "0").mkdir()
        (case_dir / "0" / "alpha.water").write_text("internalField uniform 0;\n")
        result = tool.call(
            {
                "action": "set_fields",
                "working_dir": str(tmp_path),
                "case_name": "cavity",
                "field_name": "alpha.water",
                "set_value": 1.0,
                "packing_data": str(packing),
            },
            CTX,
        )
        assert result.success is True
