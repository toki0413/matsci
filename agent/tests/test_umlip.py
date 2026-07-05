"""Tests for U-MLIP: GRACE/OMat24 backend, fine_tune, cross_validate, UMLIP_REGISTRY."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.tools.sci.ml_potential_tool import UMLIP_REGISTRY, MLPotentialInput, MLPotentialTool
from huginn.types import ToolResult


def _make_structure(tmp_path: Path) -> Path:
    # 最简 POSCAR, 够 ASE read 就行 (这里不真跑, 只测分发)
    p = tmp_path / "POSCAR"
    p.write_text(
        "Si\n1.0\n5.43 0 0\n0 5.43 0\n0 0 5.43\nSi\n2\ndirect\n0 0 0\n0.25 0.25 0.25\n",
        encoding="utf-8",
    )
    return p


class TestUmlipRegistry:
    def test_registry_has_five_backends(self):
        assert set(UMLIP_REGISTRY.keys()) == {
            "mace", "grace", "chgnet", "nep", "equiformer_v2_omat24",
        }

    def test_registry_entries_have_required_fields(self):
        for name, entry in UMLIP_REGISTRY.items():
            assert "module" in entry
            assert "class" in entry
            assert "model" in entry

    def test_all_entries_have_diversity_metadata(self):
        required_keys = {
            "training_source", "softening_tendency",
            "element_coverage_gaps", "failure_modes", "complementary_models",
        }
        for name, entry in UMLIP_REGISTRY.items():
            meta = entry.get("diversity_metadata")
            assert meta is not None, f"{name} missing diversity_metadata"
            assert required_keys <= set(meta.keys()), (
                f"{name} diversity_metadata missing keys: {required_keys - set(meta.keys())}"
            )
            assert meta["softening_tendency"] in ("high", "medium", "low")

    def test_complementary_models_are_registered(self):
        for name, entry in UMLIP_REGISTRY.items():
            for comp in entry["diversity_metadata"]["complementary_models"]:
                assert comp in UMLIP_REGISTRY, (
                    f"{name} lists complementary model '{comp}' not in registry"
                )

    def test_omat24_entry(self):
        entry = UMLIP_REGISTRY["equiformer_v2_omat24"]
        assert entry["module"] == "fairchem.core"
        assert entry["class"] == "OCCalculator"
        assert entry["model"] == "facebook/OMAT24"
        assert entry["diversity_metadata"]["training_source"] == "OMat24"


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

        # Inject a fake ase module so from-imports succeed without ase installed
        fake_io = types.ModuleType("ase.io")
        fake_io.read = MagicMock(return_value=fake_atoms)
        fake_io.write = MagicMock()
        fake_ase = types.ModuleType("ase")
        fake_ase.io = fake_io
        monkeypatch.setitem(sys.modules, "ase", fake_ase)
        monkeypatch.setitem(sys.modules, "ase.io", fake_io)

        fake_module = types.ModuleType("fairchem.core")
        fake_module.FAIRChemCalculator = MagicMock(return_value=fake_calc)
        monkeypatch.setitem(sys.modules, "fairchem", types.ModuleType("fairchem"))
        monkeypatch.setitem(sys.modules, "fairchem.core", fake_module)

        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        args = MLPotentialInput(
            backend="grace", action="predict", structure_file=str(struct)
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is True
        assert result.data["backend"] == "grace"
        assert result.data["energy"] == -10.5


class TestOMat24Backend:
    def test_omat24_returns_not_available_when_fairchem_missing(self, tmp_path, monkeypatch):
        monkeypatch.setitem(sys.modules, "fairchem", None)
        monkeypatch.setitem(sys.modules, "fairchem.core", None)

        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        args = MLPotentialInput(
            backend="equiformer_v2_omat24", action="predict", structure_file=str(struct)
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is False
        assert result.data["backend"] == "equiformer_v2_omat24"
        assert result.data["status"] == "not_available"
        assert "fairchem-core" in result.error

    def test_omat24_predict_when_available(self, tmp_path, monkeypatch):
        fake_atoms = MagicMock()
        fake_atoms.get_potential_energy.return_value = -20.0
        fake_atoms.get_forces.return_value = MagicMock(tolist=lambda: [[0.0, 0.0, 0.0]])
        fake_atoms.get_stress.return_value = MagicMock(tolist=lambda: [0.0] * 6)

        fake_calc = MagicMock()

        # Inject a fake ase module so from-imports succeed without ase installed
        fake_io = types.ModuleType("ase.io")
        fake_io.read = MagicMock(return_value=fake_atoms)
        fake_io.write = MagicMock()
        fake_ase = types.ModuleType("ase")
        fake_ase.io = fake_io
        monkeypatch.setitem(sys.modules, "ase", fake_ase)
        monkeypatch.setitem(sys.modules, "ase.io", fake_io)

        # Mock the fairchem.core.oc.calculator module chain
        fake_calc_mod = types.ModuleType("fairchem.core.oc.calculator")
        fake_calc_mod.OCCalculator = MagicMock(return_value=fake_calc)
        fake_oc = types.ModuleType("fairchem.core.oc")
        fake_oc.calculator = fake_calc_mod
        fake_core = types.ModuleType("fairchem.core")
        fake_core.oc = fake_oc
        fake_fairchem = types.ModuleType("fairchem")
        fake_fairchem.core = fake_core

        monkeypatch.setitem(sys.modules, "fairchem", fake_fairchem)
        monkeypatch.setitem(sys.modules, "fairchem.core", fake_core)
        monkeypatch.setitem(sys.modules, "fairchem.core.oc", fake_oc)
        monkeypatch.setitem(sys.modules, "fairchem.core.oc.calculator", fake_calc_mod)

        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        args = MLPotentialInput(
            backend="equiformer_v2_omat24", action="predict", structure_file=str(struct)
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is True
        assert result.data["backend"] == "equiformer_v2_omat24"
        assert result.data["energy"] == -20.0
        # Check OCCalculator was called with the HF repo id
        fake_calc_mod.OCCalculator.assert_called_once()
        call_kwargs = fake_calc_mod.OCCalculator.call_args
        assert call_kwargs.kwargs["model_path"] == "facebook/OMAT24"


class TestCrossValidate:
    def _mock_read_with_natoms(self, monkeypatch, n_atoms=2):
        """Inject a fake ase module so cross_validate's n_atoms calc works
        without ase installed."""
        fake_atoms = MagicMock()
        fake_atoms.__len__ = MagicMock(return_value=n_atoms)
        fake_io = types.ModuleType("ase.io")
        fake_io.read = MagicMock(return_value=fake_atoms)
        fake_ase = types.ModuleType("ase")
        fake_ase.io = fake_io
        monkeypatch.setitem(sys.modules, "ase", fake_ase)
        monkeypatch.setitem(sys.modules, "ase.io", fake_io)

    def test_cross_validate_energy_consistent(self, tmp_path, monkeypatch):
        """Models agree → std=0, consistency=1.0, no warning."""
        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        self._mock_read_with_natoms(monkeypatch, n_atoms=2)

        def fake_predict(model, args):
            return ToolResult(data={"backend": model, "energy": -10.0})

        monkeypatch.setattr(tool, "_predict_single", fake_predict)

        args = MLPotentialInput(
            backend="mace", action="cross_validate",
            structure_file=str(struct), property_type="energy",
            models=["mace", "chgnet"],
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is True
        assert result.data["std"] == 0.0
        assert result.data["consistency_score"] == 1.0
        assert len(result.data["warnings"]) == 0

    def test_cross_validate_energy_inconsistent(self, tmp_path, monkeypatch):
        """Models disagree → std > threshold, warning issued."""
        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        self._mock_read_with_natoms(monkeypatch, n_atoms=2)

        energies = {"mace": -10.0, "chgnet": -9.7}  # 0.3 eV / 2 atoms = 0.15 eV/atom

        def fake_predict(model, args):
            return ToolResult(data={"backend": model, "energy": energies[model]})

        monkeypatch.setattr(tool, "_predict_single", fake_predict)

        args = MLPotentialInput(
            backend="mace", action="cross_validate",
            structure_file=str(struct), property_type="energy",
            models=["mace", "chgnet"],
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is True
        assert result.data["std"] > 0.05  # > 50 meV/atom
        assert len(result.data["warnings"]) > 0
        assert "inconsistency" in result.data["warnings"][0].lower()

    def test_cross_validate_insufficient_models(self, tmp_path, monkeypatch):
        """Only one model succeeds → error returned."""
        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        self._mock_read_with_natoms(monkeypatch, n_atoms=2)

        def fake_predict(model, args):
            if model == "chgnet":
                return ToolResult(data=None, success=False, error="chgnet not installed")
            return ToolResult(data={"backend": model, "energy": -10.0})

        monkeypatch.setattr(tool, "_predict_single", fake_predict)

        args = MLPotentialInput(
            backend="mace", action="cross_validate",
            structure_file=str(struct), property_type="energy",
            models=["mace", "chgnet"],
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is False
        assert "insufficient" in (result.error or "").lower() or "at least 2" in (result.error or "").lower()

    def test_cross_validate_forces(self, tmp_path, monkeypatch):
        """Force cross-validation returns pairwise RMSE."""
        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        self._mock_read_with_natoms(monkeypatch, n_atoms=2)

        import numpy as np

        forces = {
            "mace": np.zeros((2, 3)),
            "chgnet": np.ones((2, 3)) * 0.2,  # 0.2 eV/Å per component
        }

        def fake_predict(model, args):
            return ToolResult(
                data={"backend": model, "forces": forces[model].tolist()}
            )

        monkeypatch.setattr(tool, "_predict_single", fake_predict)

        args = MLPotentialInput(
            backend="mace", action="cross_validate",
            structure_file=str(struct), property_type="forces",
            models=["mace", "chgnet"],
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is True
        assert "mean_rmse" in result.data
        assert result.data["mean_rmse"] > 0
        assert result.data["consistency_score"] < 1.0

    def test_cross_validate_auto_selects_complementary(self, tmp_path, monkeypatch):
        """Without args.models, picks complementary from diversity_metadata."""
        struct = _make_structure(tmp_path)
        tool = MLPotentialTool()
        self._mock_read_with_natoms(monkeypatch, n_atoms=2)

        called_models = []

        def fake_predict(model, args):
            called_models.append(model)
            return ToolResult(data={"backend": model, "energy": -10.0})

        monkeypatch.setattr(tool, "_predict_single", fake_predict)

        args = MLPotentialInput(
            backend="mace", action="cross_validate",
            structure_file=str(struct), property_type="energy",
        )
        result = asyncio.run(tool.call(args, MagicMock()))

        assert result.success is True
        # mace's complementary_models = ["chgnet", "nep"]
        assert "mace" in called_models
        assert "chgnet" in called_models


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
