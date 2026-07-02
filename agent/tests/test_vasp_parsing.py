"""Tests for VASP parsing upgrade: pymatgen-first with regex/ElementTree fallback."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.tools.sim.vasp_tool import VaspTool


def _make_outcar(tmp_path: Path, content: str = "") -> Path:
    p = tmp_path / "OUTCAR"
    p.write_text(content, encoding="utf-8")
    return p


def _make_vasprun(tmp_path: Path) -> Path:
    # 最简 vasprun.xml, 够 ElementTree 解析
    p = tmp_path / "vasprun.xml"
    p.write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<modeling>\n'
        '  <calculation>\n'
        '    <energy>\n'
        '      <i name="e_wo_entrp">-100.5</i>\n'
        '    </energy>\n'
        '  </calculation>\n'
        '</modeling>\n',
        encoding="utf-8",
    )
    return p


class TestParseOutcarPymatgen:
    def test_pymatgen_path_fills_energy_and_forces(self, tmp_path, monkeypatch):
        outcar = _make_outcar(tmp_path, "fake outcar content")

        # 注入假 pymatgen.io.vasp.Outcar
        fake_outcar = MagicMock()
        fake_outcar.final_energy = -123.45
        fake_outcar.forces = [[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]]
        fake_outcar.magnetizations = [[1.5, -0.5]]
        fake_outcar.converged = True

        fake_module = types.ModuleType("pymatgen.io.vasp")
        fake_module.Outcar = MagicMock(return_value=fake_outcar)

        fake_pymatgen = types.ModuleType("pymatgen")
        fake_pymatgen_io = types.ModuleType("pymatgen.io")
        fake_pymatgen_io.vasp = fake_module
        monkeypatch.setitem(sys.modules, "pymatgen", fake_pymatgen)
        monkeypatch.setitem(sys.modules, "pymatgen.io", fake_pymatgen_io)
        monkeypatch.setitem(sys.modules, "pymatgen.io.vasp", fake_module)

        tool = VaspTool(vasp_executable="fake")
        result = tool._parse_outcar_python(outcar)

        assert result["energy"] == -123.45
        assert result["converged"] is True
        assert len(result["forces"]) == 2
        assert result["forces"][0]["force"] == [0.1, 0.2, 0.3]
        assert result["magnetic_moments"] == [1.5, -0.5]
        assert result["parse_source"] == "pymatgen"

    def test_regex_fallback_when_pymatgen_missing(self, tmp_path, monkeypatch):
        # 确保 pymatgen 不可导入
        monkeypatch.setitem(sys.modules, "pymatgen", None)
        monkeypatch.setitem(sys.modules, "pymatgen.io", None)
        monkeypatch.setitem(sys.modules, "pymatgen.io.vasp", None)

        outcar_content = (
            "ENCUT = 520.0\n"
            "ISPIN = 2\n"
            "NELM = 60\n"
            " free  energy   TOTEN  =      -100.1234\n"
            "reached required accuracy\n"
            "volume of cell :      50.0\n"
        )
        outcar = _make_outcar(tmp_path, outcar_content)

        tool = VaspTool(vasp_executable="fake")
        result = tool._parse_outcar_python(outcar)

        # 走 regex 路径
        assert result["energy"] == -100.1234
        assert result["converged"] is True
        assert result["encut"] == 520.0
        assert result["ispin"] == 2
        assert result["nelm"] == 60
        assert result["volume"] == 50.0
        assert "pymatgen" not in result.get("parse_source", "")

    def test_pymatgen_failure_falls_to_regex(self, tmp_path, monkeypatch):
        # pymatgen 导入成功但 Outcar 解析抛异常
        fake_module = types.ModuleType("pymatgen.io.vasp")
        fake_module.Outcar = MagicMock(side_effect=Exception("parse error"))

        fake_pymatgen = types.ModuleType("pymatgen")
        fake_pymatgen_io = types.ModuleType("pymatgen.io")
        fake_pymatgen_io.vasp = fake_module
        monkeypatch.setitem(sys.modules, "pymatgen", fake_pymatgen)
        monkeypatch.setitem(sys.modules, "pymatgen.io", fake_pymatgen_io)
        monkeypatch.setitem(sys.modules, "pymatgen.io.vasp", fake_module)

        outcar_content = " free  energy   TOTEN  =      -50.0\nreached required accuracy\n"
        outcar = _make_outcar(tmp_path, outcar_content)

        tool = VaspTool(vasp_executable="fake")
        result = tool._parse_outcar_python(outcar)

        # pymatgen 失败, regex 兜底
        assert result["energy"] == -50.0
        assert result["converged"] is True

    def test_band_gap_not_string_placeholder(self, tmp_path, monkeypatch):
        # 确保 band_gap 不再是 "see vasprun.xml..." 字符串
        monkeypatch.setitem(sys.modules, "pymatgen", None)
        monkeypatch.setitem(sys.modules, "pymatgen.io", None)
        monkeypatch.setitem(sys.modules, "pymatgen.io.vasp", None)

        outcar = _make_outcar(tmp_path, "E-fermi : 5.0\n")
        tool = VaspTool(vasp_executable="fake")
        result = tool._parse_outcar_python(outcar)

        # band_gap 要么是 None 要么是数字, 不能是 placeholder 字符串
        assert result["band_gap"] is None
        assert result["efermi"] == 5.0


class TestParseVasprunPymatgen:
    def test_pymatgen_vasprun_returns_band_gap(self, tmp_path, monkeypatch):
        vasprun = _make_vasprun(tmp_path)

        fake_vr = MagicMock()
        fake_vr.eigenvalue_band_properties = (1.23, 2.5, 1.27)
        fake_vr.efermi = 1.88

        fake_module = types.ModuleType("pymatgen.io.vasp")
        fake_module.Vasprun = MagicMock(return_value=fake_vr)

        fake_pymatgen = types.ModuleType("pymatgen")
        fake_pymatgen_io = types.ModuleType("pymatgen.io")
        fake_pymatgen_io.vasp = fake_module
        monkeypatch.setitem(sys.modules, "pymatgen", fake_pymatgen)
        monkeypatch.setitem(sys.modules, "pymatgen.io", fake_pymatgen_io)
        monkeypatch.setitem(sys.modules, "pymatgen.io.vasp", fake_module)

        tool = VaspTool(vasp_executable="fake")
        result = tool._parse_vasprun_quick(vasprun)

        assert result["band_gap"] == 1.23
        assert result["cbm"] == 2.5
        assert result["vbm"] == 1.27
        assert result["efermi"] == 1.88
        assert result["parse_source"] == "pymatgen_vasprun"

    def test_vasprun_falls_back_to_elementtree(self, tmp_path, monkeypatch):
        # pymatgen 不可用 → 走 ElementTree
        monkeypatch.setitem(sys.modules, "pymatgen", None)
        monkeypatch.setitem(sys.modules, "pymatgen.io", None)
        monkeypatch.setitem(sys.modules, "pymatgen.io.vasp", None)

        vasprun = _make_vasprun(tmp_path)
        tool = VaspTool(vasp_executable="fake")
        result = tool._parse_vasprun_quick(vasprun)

        assert result["parse_source"] == "vasprun.xml"
        assert result.get("energy_vasprun") == -100.5
