"""Tests for U-MLIP: GRACE backend, fine_tune action, UMLIP_REGISTRY."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.tools.sci.ml_potential_tool import UMLIP_REGISTRY, MLPotentialInput, MLPotentialTool


def _make_structure(tmp_path: Path) -> Path:
    # 最简 POSCAR, 够 ASE read 就行 (这里不真跑, 只测分发)
    p = tmp_path / "POSCAR"
    p.write_text(
        "Si\n1.0\n5.43 0 0\n0 5.43 0\n0 0 5.43\nSi\n2\ndirect\n0 0 0\n0.25 0.25 0.25\n",
        encoding="utf-8",
    )
    return p


class TestUmlipRegistry:
    def test_registry_has_four_backends(self):
        assert set(UMLIP_REGISTRY.keys()) == {"mace", "grace", "chgnet", "nep"}

    def test_registry_entries_have_required_fields(self):
        for name, entry in UMLIP_REGISTRY.items():
            assert "module" in entry
            assert "class" in entry
            assert "model" in entry


class TestGraceBackend:
    def test_grace_returns_not_available_when_fairchem_missing(self, tmp_path, monkeypatch):
        # 确保 fairchem.core 不可导入
        monkeypatch.setitem(sys.modules, "fairchem", None)
        monkeypatch.setitem(sys.modules, "fairchem.core", None)

        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        args = MLPotentialInput(
            backend="grace", action="predict", structure_file=str(struct)
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is False
        assert result.data["backend"] == "grace"
        assert result.data["status"] == "not_available"
        assert "fairchem-core" in result.error

    def test_grace_predict_when_available(self, tmp_path, monkeypatch):
        # 注入假的 fairchem.core 模块
        fake_atoms = MagicMock()
        fake_atoms.get_potential_energy.return_value = -10.5
        fake_atoms.get_forces.return_value = MagicMock(tolist=lambda: [[0.0, 0.0, 0.0]])
        fake_atoms.get_stress.return_value = MagicMock(tolist=lambda: [0.0] * 6)

        fake_calc = MagicMock()
        fake_read = MagicMock(return_value=fake_atoms)

        fake_module = types.ModuleType("fairchem.core")
        fake_module.FAIRChemCalculator = MagicMock(return_value=fake_calc)
        monkeypatch.setitem(sys.modules, "fairchem", types.ModuleType("fairchem"))
        monkeypatch.setitem(sys.modules, "fairchem.core", fake_module)

        monkeypatch.setattr("ase.io.read", fake_read)

        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        args = MLPotentialInput(
            backend="grace", action="predict", structure_file=str(struct)
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is True
        assert result.data["backend"] == "grace"
        assert result.data["energy"] == -10.5


class TestFineTune:
    def test_fine_tune_mace_returns_not_available_without_module(self, tmp_path, monkeypatch):
        # mace.finetune 不可导入
        monkeypatch.setitem(sys.modules, "mace.finetune", None)

        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        args = MLPotentialInput(
            backend="mace", action="fine_tune", structure_file=str(struct)
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is False
        assert result.data["status"] == "not_available"
        assert result.data["backend"] == "mace"

    def test_fine_tune_chgnet_returns_not_supported(self, tmp_path):
        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        args = MLPotentialInput(
            backend="chgnet", action="fine_tune", structure_file=str(struct)
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is False
        assert result.data["status"] == "not_supported"

    def test_fine_tune_nep_returns_not_supported(self, tmp_path):
        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        args = MLPotentialInput(
            backend="nep", action="fine_tune", structure_file=str(struct)
        )
        result = asyncio.run(tool.call(args, MagicMock()))
        assert result.data["status"] == "not_supported"
