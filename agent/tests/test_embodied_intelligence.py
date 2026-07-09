"""Tests for EI-1 (Bayesian Skill Evolution), EI-4 (Sim-to-Real Correction),
EI-5 (Physical Pre-check with force-proceed)."""

from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

# ── EI-1: Bayesian Skill Evolution Layer ───────────────────────


class TestToolBelief:
    def test_posterior_mean_uniform_prior(self):
        from huginn.skills.evolution import ToolBelief

        b = ToolBelief("vasp_tool", "encut", "520")
        # No observations: Beta(1,1) → mean = 0.5
        assert b.posterior_mean == pytest.approx(0.5)

    def test_posterior_mean_after_successes(self):
        from huginn.skills.evolution import ToolBelief

        b = ToolBelief("vasp_tool", "encut", "520")
        for _ in range(8):
            b.update(True)
        for _ in range(2):
            b.update(False)
        # Beta(1+8, 1+2) = Beta(9, 3) → mean = 9/12 = 0.75
        assert b.posterior_mean == pytest.approx(0.75)
        assert b.total == 10
        assert b.confidence == 1.0  # 10 samples → saturated

    def test_update_increments_correctly(self):
        from huginn.skills.evolution import ToolBelief

        b = ToolBelief("lammps_tool", "timestep", "1.0")
        b.update(True)
        assert b.successes == 1
        assert b.failures == 0
        b.update(False)
        assert b.successes == 1
        assert b.failures == 1
        assert b.total == 2


class TestSkillEvolutionLayer:
    def test_record_tool_call_updates_beliefs(self):
        from huginn.skills.evolution import SkillEvolutionLayer

        layer = SkillEvolutionLayer(persist_path=None)
        layer.clear()
        layer.record_tool_call("vasp_tool", {"action": "relax", "encut": 520}, True)
        layer.record_tool_call("vasp_tool", {"action": "relax", "encut": 520}, True)
        layer.record_tool_call("vasp_tool", {"action": "relax", "encut": 520}, False)

        b = layer.get_belief("vasp_tool", "encut", "520")
        assert b is not None
        assert b.successes == 2
        assert b.failures == 1
        assert b.posterior_mean == pytest.approx(3 / 5)  # Beta(3, 2)

    def test_get_skill_context_empty(self):
        from huginn.skills.evolution import SkillEvolutionLayer

        layer = SkillEvolutionLayer(persist_path=None)
        layer.clear()
        assert layer.get_skill_context() == ""

    def test_get_skill_context_with_data(self):
        from huginn.skills.evolution import SkillEvolutionLayer

        layer = SkillEvolutionLayer(persist_path=None)
        layer.clear()
        for _ in range(5):
            layer.record_tool_call("vasp_tool", {"encut": 520, "action": "relax"}, True)
        ctx = layer.get_skill_context()
        assert "Skill Evolution" in ctx
        assert "vasp_tool" in ctx
        assert "encut=520" in ctx

    def test_get_skill_context_filters_low_samples(self):
        from huginn.skills.evolution import SkillEvolutionLayer

        layer = SkillEvolutionLayer(persist_path=None)
        layer.clear()
        layer.record_tool_call("vasp_tool", {"encut": 520}, True)  # Only 1 sample
        ctx = layer.get_skill_context()
        assert ctx == ""  # Below _MIN_SAMPLES=2

    def test_update_from_trajectory_file(self, tmp_path):
        from huginn.skills.evolution import SkillEvolutionLayer

        # Create a fake trajectory file
        traj = {
            "version": "1.0",
            "tool_calls": [
                {"tool": "vasp_tool", "args": {"action": "relax", "encut": 520}, "success": True},
                {"tool": "vasp_tool", "args": {"action": "relax", "encut": 520}, "success": True},
                {"tool": "vasp_tool", "args": {"action": "static", "encut": 520}, "success": False},
                {"tool": "lammps_tool", "args": {"timestep": 1.0}, "success": True},
            ],
        }
        traj_path = tmp_path / "test_traj.json"
        traj_path.write_text(json.dumps(traj), encoding="utf-8")

        layer = SkillEvolutionLayer(persist_path=None)
        layer.clear()
        count = layer.update_from_trajectory(traj_path)
        assert count == 4

        b = layer.get_belief("vasp_tool", "encut", "520")
        assert b is not None
        assert b.successes == 2
        assert b.failures == 1

        b2 = layer.get_belief("lammps_tool", "timestep", "1.0")
        assert b2 is not None
        assert b2.successes == 1

    def test_update_from_directory(self, tmp_path):
        from huginn.skills.evolution import SkillEvolutionLayer

        for i in range(3):
            traj = {
                "version": "1.0",
                "tool_calls": [
                    {"tool": "vasp_tool", "args": {"encut": 400}, "success": True},
                ],
            }
            (tmp_path / f"traj_{i}.json").write_text(json.dumps(traj), encoding="utf-8")

        layer = SkillEvolutionLayer(persist_path=None)
        layer.clear()
        total = layer.update_from_directory(tmp_path)
        assert total == 3

        b = layer.get_belief("vasp_tool", "encut", "400")
        assert b is not None
        assert b.successes == 3

    def test_recommend_params(self):
        from huginn.skills.evolution import SkillEvolutionLayer

        layer = SkillEvolutionLayer(persist_path=None)
        layer.clear()
        # encut=520: 8 success, 2 fail → P=0.75
        for _ in range(8):
            layer.record_tool_call("vasp_tool", {"encut": 520}, True)
        for _ in range(2):
            layer.record_tool_call("vasp_tool", {"encut": 520}, False)
        # encut=400: 2 success, 8 fail → P=0.25
        for _ in range(2):
            layer.record_tool_call("vasp_tool", {"encut": 400}, True)
        for _ in range(8):
            layer.record_tool_call("vasp_tool", {"encut": 400}, False)

        recs = layer.recommend_params("vasp_tool", "encut")
        assert len(recs) == 2
        assert recs[0][0] == "520"  # Higher success rate first
        assert recs[0][1] > recs[1][1]

    def test_persistence_roundtrip(self, tmp_path):
        from huginn.skills.evolution import SkillEvolutionLayer

        path = tmp_path / "beliefs.json"
        layer1 = SkillEvolutionLayer(persist_path=path)
        layer1.clear()
        for _ in range(3):
            layer1.record_tool_call("vasp_tool", {"encut": 520}, True)
        layer1._save()

        # New instance loads from same file
        layer2 = SkillEvolutionLayer(persist_path=path)
        b = layer2.get_belief("vasp_tool", "encut", "520")
        assert b is not None
        assert b.successes == 3

    def test_summary(self):
        from huginn.skills.evolution import SkillEvolutionLayer

        layer = SkillEvolutionLayer(persist_path=None)
        layer.clear()
        layer.record_tool_call("vasp_tool", {"encut": 520}, True)
        s = layer.summary()
        assert s["total_beliefs"] == 1
        assert "vasp_tool" in s["tools"]


# ── EI-4: Sim-to-Real Correction Table ──────────────────────────


class TestCorrectionEntry:
    def test_correction_factor_underestimate(self):
        from huginn.provenance.correction import CorrectionEntry

        e = CorrectionEntry("Si", "band_gap", 0.61, 1.12, "PBE", "builtin (exp)")
        # exp/calc = 1.12/0.61 ≈ 1.836
        assert e.correction_factor == pytest.approx(1.836, abs=0.01)

    def test_correction_factor_overestimate(self):
        from huginn.provenance.correction import CorrectionEntry

        e = CorrectionEntry("Cu", "lattice_constant", 3.68, 3.61, "PBE", "builtin (exp)")
        # exp/calc = 3.61/3.68 ≈ 0.981
        assert e.correction_factor == pytest.approx(0.981, abs=0.01)

    def test_offset(self):
        from huginn.provenance.correction import CorrectionEntry

        e = CorrectionEntry("Si", "band_gap", 0.61, 1.12, "PBE", "builtin (exp)")
        assert e.offset == pytest.approx(0.51, abs=0.01)

    def test_zero_calc_value(self):
        from huginn.provenance.correction import CorrectionEntry

        e = CorrectionEntry("X", "prop", 0.0, 1.0, "PBE", "test")
        assert e.correction_factor == 1.0  # Safe default


class TestCorrectionTable:
    def test_builtin_corrections_loaded(self):
        from huginn.provenance.correction import CorrectionTable

        table = CorrectionTable(persist_path=None)
        # Should have Si band_gap PBE
        entries = table.get_corrections("Si", "band_gap", "PBE")
        assert len(entries) >= 1
        assert entries[0].calc_value == pytest.approx(0.61)
        assert entries[0].exp_value == pytest.approx(1.12)

    def test_register_user_correction(self):
        from huginn.provenance.correction import CorrectionTable

        table = CorrectionTable(persist_path=None)
        # Clear user entries to start fresh
        table.clear_user_entries()
        table.register("MoS2", "band_gap", 1.55, 1.80, "PBE", "user")
        entries = table.get_corrections("MoS2", "band_gap", "PBE")
        assert any(e.source == "user" for e in entries)

    def test_apply_correction_with_builtin(self):
        from huginn.provenance.correction import CorrectionTable

        table = CorrectionTable(persist_path=None)
        # Si PBE bandgap: 0.61 → 1.12, factor ≈ 1.836
        corrected = table.apply_correction("Si", "band_gap", 0.65, "PBE")
        assert corrected == pytest.approx(0.65 * 1.836, abs=0.05)

    def test_apply_correction_no_data_returns_raw(self):
        from huginn.provenance.correction import CorrectionTable

        table = CorrectionTable(persist_path=None)
        result = table.apply_correction("UnknownMaterial", "some_prop", 42.0, "PBE")
        assert result == 42.0

    def test_apply_correction_no_method_matches_any(self):
        from huginn.provenance.correction import CorrectionTable

        table = CorrectionTable(persist_path=None)
        # No method filter → returns all Si band_gap entries
        entries = table.get_corrections("Si", "band_gap", "")
        assert len(entries) >= 2  # PBE and HSE06

    def test_to_context_block(self):
        from huginn.provenance.correction import CorrectionTable

        table = CorrectionTable(persist_path=None)
        ctx = table.to_context_block()
        assert "Sim-to-Real" in ctx
        assert "Si" in ctx
        assert "band_gap" in ctx

    def test_list_materials(self):
        from huginn.provenance.correction import CorrectionTable

        table = CorrectionTable(persist_path=None)
        materials = table.list_materials()
        assert "Si" in materials
        assert "GaAs" in materials
        assert "Cu" in materials

    def test_list_properties(self):
        from huginn.provenance.correction import CorrectionTable

        table = CorrectionTable(persist_path=None)
        props = table.list_properties("Si")
        assert "band_gap" in props
        assert "lattice_constant" in props

    def test_persistence_roundtrip(self, tmp_path):
        from huginn.provenance.correction import CorrectionTable

        path = tmp_path / "corrections.json"
        t1 = CorrectionTable(persist_path=path)
        t1.clear_user_entries()
        t1.register("MoS2", "band_gap", 1.55, 1.80, "PBE", "user")
        t1._save()

        t2 = CorrectionTable(persist_path=path)
        entries = t2.get_corrections("MoS2", "band_gap", "PBE")
        assert any(e.source == "user" for e in entries)

    def test_summary(self):
        from huginn.provenance.correction import CorrectionTable

        table = CorrectionTable(persist_path=None)
        s = table.summary()
        assert s["total_entries"] > 0
        assert "Si" in s["materials"]


# ── EI-5: Physical Pre-check (warn + force-proceed) ─────────────


class TestPhysicalPrecheck:
    """Test that pre-checks warn and block, but allow force-proceed."""

    def _run_pre(self, hm, tool_name, args):
        """Helper: run pre hooks and return (allowed, args, ctx)."""
        return asyncio.run(hm.run_pre(tool_name, args))

    def test_band_before_scf_blocks_without_scf(self):
        from unittest.mock import MagicMock, patch
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks

        # Mock ProvenanceRegistry: no prior SCF found
        mock_reg = MagicMock()
        mock_reg.find_by_tool.return_value = []

        hm = HookManager()
        register_physical_prechecks(hm)
        with patch(
            "huginn.provenance.registry.ProvenanceRegistry.shared",
            return_value=mock_reg,
        ):
            allowed, _, ctx = self._run_pre(
                hm, "vasp_tool", {"action": "band", "encut": 520}
            )
        assert not allowed
        assert "physical_warning" in ctx.metadata
        assert "force_proceed_available" in ctx.metadata
        assert ctx.metadata["force_proceed_available"] is True
        assert "SCF" in ctx.metadata["block_reason"]

    def test_band_before_scf_force_proceed_skips(self):
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks

        hm = HookManager()
        register_physical_prechecks(hm)
        allowed, _, ctx = self._run_pre(
            hm, "vasp_tool", {"action": "band", "encut": 520, "force_proceed": True}
        )
        assert allowed
        assert "physical_warning" not in ctx.metadata

    def test_band_passes_when_scf_done(self):
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks
        from huginn.provenance.registry import ProvenanceRegistry

        # Register a prior SCF calculation
        reg = ProvenanceRegistry.shared()
        reg.register(
            "/fake/outcar", "vasp_tool",
            parameters={"action": "static"},
            key_properties={"converged": True},
        )

        hm = HookManager()
        register_physical_prechecks(hm)
        allowed, _, ctx = self._run_pre(
            hm, "vasp_tool", {"action": "band", "encut": 520}
        )
        assert allowed

    def test_elastic_without_relax_blocks(self):
        from unittest.mock import MagicMock, patch
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks

        # Mock: no prior relaxation found
        mock_reg = MagicMock()
        mock_reg.find_by_tool.return_value = []

        hm = HookManager()
        register_physical_prechecks(hm)
        with patch(
            "huginn.provenance.registry.ProvenanceRegistry.shared",
            return_value=mock_reg,
        ):
            allowed, _, ctx = self._run_pre(
                hm, "mechanical_tool", {"action": "elastic_constants"}
            )
        assert not allowed
        assert "弹性常数" in ctx.metadata["block_reason"]
        assert ctx.metadata["force_proceed_available"] is True

    def test_elastic_force_proceed_skips(self):
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks

        hm = HookManager()
        register_physical_prechecks(hm)
        allowed, _, ctx = self._run_pre(
            hm, "mechanical_tool",
            {"action": "elastic_constants", "force_proceed": True}
        )
        assert allowed

    def test_md_timestep_too_large_blocks(self):
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks

        hm = HookManager()
        register_physical_prechecks(hm)
        allowed, _, ctx = self._run_pre(
            hm, "lammps_tool", {"action": "md", "timestep": 10.0}
        )
        assert not allowed
        assert "时间步" in ctx.metadata["block_reason"]
        assert ctx.metadata["force_proceed_available"] is True

    def test_md_timestep_reasonable_passes(self):
        from unittest.mock import MagicMock, patch
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks

        # Mock: prior minimize found so md_without_minimize_hook passes
        mock_reg = MagicMock()
        mock_entry = MagicMock()
        mock_entry.parameters = {"action": "minimize"}
        mock_reg.find_by_tool.return_value = [mock_entry]

        hm = HookManager()
        register_physical_prechecks(hm)
        with patch(
            "huginn.provenance.registry.ProvenanceRegistry.shared",
            return_value=mock_reg,
        ):
            allowed, _, _ = self._run_pre(
                hm, "lammps_tool", {"action": "md", "timestep": 1.0}
            )
        assert allowed

    def test_low_encut_blocks(self):
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks

        hm = HookManager()
        register_physical_prechecks(hm)
        allowed, _, ctx = self._run_pre(
            hm, "vasp_tool", {"action": "relax", "encut": 200}
        )
        assert not allowed
        assert "截断能" in ctx.metadata["block_reason"]

    def test_normal_encut_passes(self):
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks

        hm = HookManager()
        register_physical_prechecks(hm)
        allowed, _, _ = self._run_pre(
            hm, "vasp_tool", {"action": "relax", "encut": 520}
        )
        assert allowed

    def test_non_relevant_tool_passes(self):
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks

        hm = HookManager()
        register_physical_prechecks(hm)
        allowed, _, _ = self._run_pre(
            hm, "rdkit_tool", {"action": "smiles_to_mol", "smiles": "CCO"}
        )
        assert allowed

    def test_register_is_idempotent(self):
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks

        hm = HookManager()
        register_physical_prechecks(hm)
        register_physical_prechecks(hm)  # Should not double-register
        # Count callbacks: should still be 5 (one per hook)
        assert len(hm._callbacks["pre_tool_use"]) == 5

    def test_block_reason_includes_force_proceed_instructions(self):
        from huginn.hooks import HookManager
        from huginn.hooks.physical_precheck import register_physical_prechecks

        hm = HookManager()
        register_physical_prechecks(hm)
        allowed, _, ctx = self._run_pre(
            hm, "vasp_tool", {"action": "band", "encut": 200}
        )
        assert not allowed
        reason = ctx.metadata["block_reason"]
        assert "force_proceed=True" in reason
        assert "Physical Pre-check" in reason
