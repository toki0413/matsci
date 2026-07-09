"""Tests for composite skill definitions and registration."""

from huginn.skills.composite import (
    BAND_STRUCTURE_ANALYSIS,
    MD_PIPELINE,
    MECHANICAL_PROPERTIES,
    MOLECULE_SCREENING,
    PHONON_ANALYSIS,
)
from huginn.skills.registry import SkillRegistry

ALL_COMPOSITE = [
    BAND_STRUCTURE_ANALYSIS,
    MECHANICAL_PROPERTIES,
    MD_PIPELINE,
    MOLECULE_SCREENING,
    PHONON_ANALYSIS,
]


class TestCompositeShape:
    """Every composite skill must satisfy the structural contract."""

    def test_names(self):
        names = [s.name for s in ALL_COMPOSITE]
        assert names == [
            "band_structure_analysis",
            "mechanical_properties",
            "md_pipeline",
            "molecule_screening",
            "phonon_analysis",
        ]

    def test_all_computation_category(self):
        assert all(s.category == "computation" for s in ALL_COMPOSITE)

    def test_at_least_three_steps(self):
        for s in ALL_COMPOSITE:
            assert len(s.steps) >= 3, f"{s.name} has too few steps"

    def test_has_required_tools(self):
        for s in ALL_COMPOSITE:
            assert len(s.required_tools) >= 1, f"{s.name} missing required_tools"

    def test_has_tags(self):
        for s in ALL_COMPOSITE:
            assert len(s.tags) >= 3, f"{s.name} missing tags"

    def test_has_estimated_cost(self):
        for s in ALL_COMPOSITE:
            assert "cpu_hours" in s.estimated_cost
            assert s.estimated_cost["cpu_hours"] > 0

    def test_has_applicable_systems(self):
        for s in ALL_COMPOSITE:
            systems = s.metadata.get("applicable_systems")
            assert systems is not None, f"{s.name} missing metadata.applicable_systems"
            assert len(systems) >= 2

    def test_first_param_is_structure_or_smiles(self):
        for s in ALL_COMPOSITE:
            first = s.parameters[0]
            assert first.required is True
            assert first.name in ("structure_file", "input_molecules", "smiles")


class TestBandStructureAnalysis:
    def test_four_steps(self):
        assert len(BAND_STRUCTURE_ANALYSIS.steps) == 4

    def test_step_sequence(self):
        names = [s.name for s in BAND_STRUCTURE_ANALYSIS.steps]
        assert names == [
            "structure_relaxation",
            "scf_calculation",
            "band_calculation",
            "dos_calculation",
        ]

    def test_all_steps_use_vasp(self):
        assert all(s.tool == "vasp_tool" for s in BAND_STRUCTURE_ANALYSIS.steps)

    def test_band_uses_icharg_11(self):
        band = BAND_STRUCTURE_ANALYSIS.steps[2]
        # icharg maps to "11" which ast.literal_eval turns into int 11
        assert band.input_mapping["icharg"] == "11"

    def test_defaults(self):
        params = {p.name: p for p in BAND_STRUCTURE_ANALYSIS.parameters}
        assert params["encut"].default == 520
        assert params["ediff"].default == 1e-6
        assert params["functional"].default == "PBE"


class TestMdPipeline:
    def test_five_steps(self):
        assert len(MD_PIPELINE.steps) == 5

    def test_step_sequence(self):
        names = [s.name for s in MD_PIPELINE.steps]
        assert names == [
            "build_initial_config",
            "energy_minimization",
            "npt_equilibration",
            "production_md",
            "trajectory_analysis",
        ]

    def test_tool_sequence(self):
        tools = [s.tool for s in MD_PIPELINE.steps]
        assert tools == ["packing_tool", "lammps_tool", "lammps_tool", "lammps_tool", "evaluation_tool"]

    def test_defaults(self):
        params = {p.name: p for p in MD_PIPELINE.parameters}
        assert params["temperature"].default == 300.0
        assert params["n_steps_production"].default == 500000


class TestRegistration:
    def test_all_registered_in_registry(self):
        for skill in ALL_COMPOSITE:
            assert SkillRegistry.get(skill.name) is skill

    def test_listed_by_name(self):
        names = SkillRegistry.list_skills()
        for skill in ALL_COMPOSITE:
            assert skill.name in names

    def test_searchable_by_tag(self):
        # every skill should be findable via at least one of its own tags
        for skill in ALL_COMPOSITE:
            hits = SkillRegistry.search(skill.tags[0])
            assert skill in hits, f"{skill.name} not found via tag '{skill.tags[0]}'"


class TestStepConditions:
    """Non-first steps must guard on a prior step's output_key."""

    def test_first_step_has_no_condition(self):
        for skill in ALL_COMPOSITE:
            assert skill.steps[0].condition is None, \
                f"{skill.name}: first step should not have a condition"

    def test_later_steps_have_conditions(self):
        for skill in ALL_COMPOSITE:
            for step in skill.steps[1:]:
                assert step.condition is not None, \
                    f"{skill.name}:{step.name} missing condition"

    def test_conditions_reference_prior_output_keys(self):
        for skill in ALL_COMPOSITE:
            seen = set()
            seen.add(skill.steps[0].output_key)
            for step in skill.steps[1:]:
                assert step.condition is not None
                referenced = any(key in step.condition for key in seen), \
                    f"{skill.name}:{step.name} condition doesn't reference a prior output_key"
                assert referenced
                seen.add(step.output_key)


class TestSafeEvalCompatibility:
    """Conditions and validations must parse and evaluate under safe_eval.

    safe_eval forbids attribute access and function calls, so these
    expressions use subscript + IfExp instead of dict.get().
    """

    MOCK = {
        "converged": True,
        "tensor": True,
        "structure": True,
        "equilibrated": True,
        "completed": True,
        "optimized": True,
        "energy": True,
        "homo": True,
        "logp": True,
        "force_constants": True,
        "free_energy": True,
        "bulk_modulus": True,
    }

    def test_conditions_eval_to_true_with_mock_data(self):
        from huginn.security import safe_eval

        for skill in ALL_COMPOSITE:
            ctx: dict = {}
            for step in skill.steps:
                if step.condition is not None:
                    result = safe_eval(step.condition, ctx)
                    assert result, f"{skill.name}:{step.name} condition False with mock data"
                ctx[step.output_key] = self.MOCK

    def test_validations_eval_to_true_with_mock_data(self):
        from huginn.security import safe_eval

        for skill in ALL_COMPOSITE:
            ctx: dict = {}
            for step in skill.steps:
                ctx[step.output_key] = self.MOCK
                if step.validation is not None:
                    result = safe_eval(step.validation, ctx)
                    assert result, f"{skill.name}:{step.name} validation False with mock data"

    def test_conditions_safe_when_field_missing(self):
        """When a prior output exists but lacks the field, condition is False.

        IfExp short-circuits: 'field' in {} is False, so the subscript
        never runs and we get False instead of a KeyError.
        """
        from huginn.security import safe_eval

        for skill in ALL_COMPOSITE:
            ctx: dict = {}
            for step in skill.steps:
                if step.condition is not None:
                    result = safe_eval(step.condition, ctx)
                    assert result is False, \
                        f"{skill.name}:{step.name} condition should be False when field missing"
                ctx[step.output_key] = {}  # output present, fields absent


class TestDefaultParameters:
    def test_encut_defaults_above_400ev(self):
        for skill in ALL_COMPOSITE:
            encut = next((p for p in skill.parameters if p.name == "encut"), None)
            if encut is not None:
                assert encut.default > 400, f"{skill.name} encut too low"

    def test_ediff_positive(self):
        for skill in ALL_COMPOSITE:
            ediff = next((p for p in skill.parameters if p.name == "ediff"), None)
            if ediff is not None:
                assert ediff.default > 0, f"{skill.name} ediff must be positive"

    def test_temperatures_positive(self):
        for skill in ALL_COMPOSITE:
            for p in skill.parameters:
                if p.name in ("temperature", "t_max"):
                    assert p.default > 0

    def test_str_defaults_are_strings(self):
        for skill in ALL_COMPOSITE:
            for p in skill.parameters:
                if p.type == "str" and p.default is not None:
                    assert isinstance(p.default, str)
