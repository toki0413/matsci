"""Tests for VM-1 (fracture assessment skill + pipeline),
VM-2 (Abaqus Explicit Dynamic), VM-3 (LAMMPS DEM packing)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ── VM-1: Fracture Assessment Composite Skill ──────────────────


class TestFractureAssessmentSkill:
    def test_skill_registered(self):
        from huginn.skills.registry import SkillRegistry
        skill = SkillRegistry.get("fracture_assessment")
        assert skill is not None
        assert skill.category == "analysis"
        assert "image_analysis_tool" in skill.required_tools
        assert "specialty_analysis_tool" in skill.required_tools

    def test_skill_has_crack_detection_step(self):
        from huginn.skills.registry import SkillRegistry
        skill = SkillRegistry.get("fracture_assessment")
        step_names = [s.name for s in skill.steps]
        assert "crack_detection" in step_names
        assert "fracture_lefm" in step_names

    def test_skill_tags_include_fracture(self):
        from huginn.skills.registry import SkillRegistry
        skill = SkillRegistry.get("fracture_assessment")
        assert "fracture" in skill.tags
        assert "sem" in skill.tags

    def test_pipeline_rule_defect_detect_to_analysis(self):
        from huginn.provenance.pipeline import PIPELINE_RULES
        rules = [r for r in PIPELINE_RULES if r.tool_name == "image_analysis_tool"]
        assert len(rules) >= 2  # PROPERTIES + MECHANICAL stages
        defect_rules = [r for r in rules if r.action_matcher == "defect_detect"]
        assert len(defect_rules) >= 2
        for r in defect_rules:
            assert "specialty_analysis_tool" in r.next_tool_hints


# ── VM-2: Abaqus Explicit Dynamic ────────────────────────────────


class TestAbaqusExplicitDynamic:
    def test_explicit_dynamic_spec_creation(self):
        from huginn.tools.sim.abaqus_tool import ExplicitDynamicSpec
        spec = ExplicitDynamicSpec(
            youngs_modulus=210e9,
            poissons_ratio=0.3,
            density=7850.0,
            total_time=0.001,
        )
        assert spec.material_model == "plastic"  # default
        assert spec.mass_scaling == 1.0  # default

    def test_explicit_dynamic_spec_with_contact(self):
        from huginn.tools.sim.abaqus_tool import ExplicitDynamicSpec
        spec = ExplicitDynamicSpec(
            youngs_modulus=70e9,
            density=2700.0,
            total_time=0.005,
            contact_pairs=[
                {"master": "PLATE", "slave": "IMPACTOR", "type": "general", "friction": 0.3}
            ],
            initial_velocity=[0, 0, -10.0],
            material_model="johnson_cook",
            plastic_params={"A": 250e6, "B": 5e8, "n": 0.3, "C": 0.01, "m": 1.0,
                           "T_ref": 293.0, "T_melt": 1800.0},
        )
        assert spec.material_model == "johnson_cook"
        assert len(spec.contact_pairs) == 1
        assert spec.initial_velocity == [0, 0, -10.0]

    def test_explicit_dynamic_in_action_literal(self):
        from huginn.tools.sim.abaqus_tool import AbaqusToolInput
        # action should accept "explicit_dynamic"
        inp = AbaqusToolInput(action="explicit_dynamic", explicit_spec={
            "youngs_modulus": 210e9, "total_time": 0.001
        })
        assert inp.action == "explicit_dynamic"

    def test_explicit_dynamic_validates_spec(self):
        from huginn.tools.sim.abaqus_tool import AbaqusToolInput
        with pytest.raises(Exception):
            AbaqusToolInput(action="explicit_dynamic")  # missing explicit_spec

    def test_script_generator_produces_valid_python(self):
        from huginn.tools.sim.abaqus_tool import AbaqusTool, AbaqusToolInput, ExplicitDynamicSpec
        tool = AbaqusTool()
        spec = ExplicitDynamicSpec(
            youngs_modulus=210e9,
            density=7850.0,
            total_time=0.001,
            contact_pairs=[
                {"master": "SURF-1", "slave": "SURF-2", "type": "general", "friction": 0.2}
            ],
            initial_velocity=[0, 0, -5.0],
            loads=[{"type": "gravity", "value": 9.81}],
            boundary_conditions=[{"region": "SET-FIX", "dofs": [1, 2, 3], "value": 0.0}],
        )
        args = AbaqusToolInput(
            action="explicit_dynamic",
            explicit_spec=spec,
            output_prefix="test_explicit",
            base_model="Model-1",
        )
        script = tool._generate_explicit_dynamic_script(args)
        assert "ExplicitDynamicsStep" in script
        assert "JohnsonCook" in script or "Plastic" in script or "Elastic" in script
        assert "massScaling" in script or "MASS_SCALING" in script
        assert "ALLKE" in script  # kinetic energy output
        assert "ETOTAL" in script  # energy balance


# ── VM-3: LAMMPS DEM Packing ────────────────────────────────────


class TestLammpsDEMPacking:
    def test_dem_action_in_literal(self):
        from huginn.tools.sim.lammps_tool import LammpsToolInput
        inp = LammpsToolInput(action="dem_packing", dem_n_particles=100)
        assert inp.action == "dem_packing"
        assert inp.dem_n_particles == 100

    def test_dem_fields_have_defaults(self):
        from huginn.tools.sim.lammps_tool import LammpsToolInput
        inp = LammpsToolInput(action="dem_packing")
        assert inp.dem_n_particles == 1000
        assert inp.dem_radius == 5.0
        assert inp.dem_friction == 0.5
        assert inp.dem_restitution == 0.8

    def test_dem_script_generation(self):
        from huginn.tools.sim.lammps_tool import LammpsTool, LammpsToolInput
        tool = LammpsTool()
        args = LammpsToolInput(
            action="dem_packing",
            dem_n_particles=500,
            dem_radius=3.0,
            dem_density=2.5,
            dem_youngs=1e6,
            dem_friction=0.3,
            dem_restitution=0.9,
            dem_gravity=9.81,
            dem_n_steps=50000,
        )
        script = tool._generate_dem_input_script(args)
        assert "granular" in script
        assert "hertz/material" in script
        assert "mindlin" in script
        assert "nve/sphere" in script
        assert "gravity" in script  # gravity enabled
        assert "contact/atom" in script  # coordination number

    def test_dem_script_no_gravity(self):
        from huginn.tools.sim.lammps_tool import LammpsTool, LammpsToolInput
        tool = LammpsTool()
        args = LammpsToolInput(
            action="dem_packing",
            dem_gravity=0.0,
        )
        script = tool._generate_dem_input_script(args)
        assert "no gravity" in script

    def test_dem_script_polydisperse(self):
        from huginn.tools.sim.lammps_tool import LammpsTool, LammpsToolInput
        tool = LammpsTool()
        args = LammpsToolInput(
            action="dem_packing",
            dem_radius=5.0,
            dem_radius_std=1.0,
        )
        script = tool._generate_dem_input_script(args)
        assert "r_var normal" in script
        assert "diameter v_r_var" in script

    def test_dem_script_monodisperse(self):
        from huginn.tools.sim.lammps_tool import LammpsTool, LammpsToolInput
        tool = LammpsTool()
        args = LammpsToolInput(
            action="dem_packing",
            dem_radius=5.0,
            dem_radius_std=0.0,
        )
        script = tool._generate_dem_input_script(args)
        assert "r_var" not in script

    def test_dem_script_contains_packing_fraction(self):
        from huginn.tools.sim.lammps_tool import LammpsTool, LammpsToolInput
        tool = LammpsTool()
        args = LammpsToolInput(action="dem_packing")
        script = tool._generate_dem_input_script(args)
        assert "Packing fraction" in script
        assert "coordination number" in script

    @pytest.mark.asyncio
    async def test_dem_packing_handles_no_executable(self):
        from huginn.tools.sim.lammps_tool import LammpsTool, LammpsToolInput
        from huginn.types import ToolContext

        tool = LammpsTool()
        tool.lammps_executable = None
        # Mock resolve_executable to return a resolution request
        mock_req = MagicMock()
        mock_req.to_dict.return_value = {"tool": "lammps"}
        mock_req.install_hint = "Install LAMMPS"

        with patch(
            "huginn.tools.sim.executable_resolver.resolve_executable",
            return_value=mock_req,
        ):
            args = LammpsToolInput(action="dem_packing", dem_n_particles=10)
            ctx = ToolContext(session_id="test", workspace=".")
            result = await tool._handle_dem_packing(args, ctx)
            assert result.success is True
            assert "script_path" in result.data
            assert result.data.get("needs_resolution") is True
