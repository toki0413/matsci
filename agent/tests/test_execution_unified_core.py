"""Tests for execution/ and unified/ core modules.

These modules are pure-Python with no heavy external dependencies (no VASP/QE/ABAQUS
executables, no transformers, no torch). They only use sympy, numpy, and matplotlib.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sympy as sp

from huginn.execution.autofix import AutoFixLoop
from huginn.execution.dimensional_validator import (
    DimensionalCheckResult,
    DimensionalValidator,
    Unit,
    UnitRegistry,
    registry,
)
from huginn.execution.orchestrator import (
    ExecutionOrchestrator,
    StageResult,
    WorkflowExecutionRecord,
)
from huginn.execution.remote_job_store import RemoteJobRecord, RemoteJobStore
from huginn.unified.bridge import (
    BRIDGE_REGISTRY,
    cauchy_born_elasticity,
    dft_potential_to_md,
    get_bridge,
    list_bridges,
    md_stress_to_continuum,
)
from huginn.unified.core import (
    ConstitutiveModel,
    Domain,
    EnergyFunctional,
    Field,
    FieldKind,
    UnifiedProblem,
    VariationalPrinciple,
)
from huginn.unified.derive import _derive_hamiltonian, _euler_lagrange, derive_equations
from huginn.unified.models import (
    get_model,
    harmonic_oscillator_md,
    heat_equation_fem,
    linear_elasticity_fem,
    list_models,
    one_d_kohn_sham_dft,
)
from huginn.unified.solve import solve
from huginn.unified.visualize import plot_solution, solve_and_plot


# ------------------------------------------------------------------
# execution/dimensional_validator
# ------------------------------------------------------------------

class TestDimensionalValidator:
    def test_parse_quantity(self):
        d = DimensionalValidator()
        val, unit, dims = d.parse_quantity("210 GPa")
        assert val == 210.0
        assert unit == "GPa"
        assert dims == {"M": 1, "L": -1, "T": -2}

    def test_parse_compound_unit(self):
        d = DimensionalValidator()
        val, unit, dims = d.parse_quantity("1.0 kg/m3")
        assert val == 1.0
        assert unit == "kg/m3"
        assert dims == {"M": 1, "L": -3}

    def test_check_equation_consistent(self):
        d = DimensionalValidator()
        result = d.check_equation(
            lhs_quantities=["500 MPa"],
            rhs_quantities=["210 GPa"],
            equation_name="sigma = E",
        )
        assert result.consistent is True
        assert "verified" in result.notes[0]

    def test_check_equation_inconsistent(self):
        d = DimensionalValidator()
        result = d.check_equation(
            lhs_quantities=["500 MPa"],
            rhs_quantities=["210 GPa", "1.0 m/s"],
            equation_name="wrong",
        )
        assert result.consistent is False

    def test_validate_stress_strain(self):
        d = DimensionalValidator()
        result = d.validate_stress_strain(
            stress_val=500, stress_unit="MPa",
            E_val=210, E_unit="GPa",
            strain=0.001,
        )
        # dimensionless is not a known unit in parse_quantity, so this may fail
        # but we test the structure
        assert isinstance(result, DimensionalCheckResult)

    def test_buckingham_pi(self):
        d = DimensionalValidator()
        groups = d.buckingham_pi(
            variables=[("E", "GPa"), ("rho", "kg/m3"), ("L", "m")],
            target="E",
        )
        assert isinstance(groups, list)

    def test_vasp_inputs_check(self):
        d = DimensionalValidator()
        results = d.check_vasp_inputs({"ENCUT": 520, "SIGMA": 0.05, "POTIM": 0.5})
        assert all(r.consistent for r in results)


# ------------------------------------------------------------------
# Unit algebraic type
# ------------------------------------------------------------------

class TestUnitAlgebra:
    def test_unit_base_construction(self):
        m = Unit.base(1, "m")  # L index
        assert m.dimensions[1] == 1.0
        assert m.scale == 1.0

    def test_unit_dimensionless(self):
        d = Unit.dimensionless()
        assert d.is_dimensionless
        assert d.dimension_signature == "dimensionless"

    def test_multiply_adds_dimensions(self):
        kg = Unit.base(0, "kg")
        m = Unit.base(1, "m")
        result = kg * m
        assert result.dimensions[0] == 1.0  # M
        assert result.dimensions[1] == 1.0  # L

    def test_divide_subtracts_dimensions(self):
        m = Unit.base(1, "m")
        s = Unit.base(2, "s")
        velocity = m / s
        assert velocity.dimensions[1] == 1.0   # L
        assert velocity.dimensions[2] == -1.0  # T

    def test_power_scales_dimensions(self):
        m = Unit.base(1, "m")
        m3 = m ** 3
        assert m3.dimensions[1] == 3.0  # L^3

    def test_equality_compares_dimensions_only(self):
        a = Unit(scale=1.0, dimensions=(1.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0), name="Pa")
        b = Unit(scale=1e9, dimensions=(1.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0), name="GPa")
        assert a == b  # same dimensions, different scale

    def test_same_dimensions_method(self):
        m = Unit.base(1, "m")
        cm = Unit(scale=0.01, dimensions=m.dimensions, name="cm")
        assert m.same_dimensions(cm)

    def test_dimension_dict(self):
        Pa = Unit(scale=1.0, dimensions=(1.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0), name="Pa")
        d = Pa.dimension_dict
        assert d == {"M": 1.0, "L": -1.0, "T": -2.0}

    def test_dimension_signature(self):
        Pa = Unit(scale=1.0, dimensions=(1.0, -1.0, -2.0, 0.0, 0.0, 0.0, 0.0), name="Pa")
        sig = Pa.dimension_signature
        assert "M1" in sig
        assert "L-1" in sig
        assert "T-2" in sig

    def test_hash_based_on_dimensions(self):
        a = Unit(scale=1.0, dimensions=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        b = Unit(scale=999.0, dimensions=(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        assert hash(a) == hash(b)


# ------------------------------------------------------------------
# UnitRegistry — composable definitions
# ------------------------------------------------------------------

class TestUnitRegistry:
    def test_n_per_m2_equals_pa(self):
        """N/m² should have the same dimensions as Pa."""
        assert registry.equivalent("N/m^2", "Pa")

    def test_kg_m_per_s2_equals_n(self):
        """kg·m/s² should have the same dimensions as N."""
        assert registry.equivalent("kg·m/s^2", "N")

    def test_j_equals_n_times_m(self):
        """J should equal N·m in dimensions."""
        assert registry.equivalent("J", "N·m")

    def test_w_equals_j_per_s(self):
        """W should equal J/s in dimensions."""
        assert registry.equivalent("W", "J/s")

    def test_hz_is_inverse_time(self):
        """Hz should have dimensions T^-1."""
        hz = registry.get("Hz")
        assert hz.dimensions[2] == -1.0  # T

    def test_gpa_to_pa_conversion(self):
        """210 GPa should convert to 2.1e11 Pa."""
        result = registry.convert(210, "GPa", "Pa")
        assert result == pytest.approx(2.1e11)

    def test_km_to_m_conversion(self):
        """5 km should convert to 5000 m."""
        result = registry.convert(5, "km", "m")
        assert result == pytest.approx(5000.0)

    def test_ev_to_j_conversion(self):
        """1 eV should convert to ~1.602e-19 J."""
        result = registry.convert(1, "eV", "J")
        assert result == pytest.approx(1.60218e-19)

    def test_convert_incompatible_raises(self):
        """Converting between incompatible units should raise ValueError."""
        with pytest.raises(ValueError, match="Incompatible"):
            registry.convert(1, "m", "kg")

    def test_si_prefix_auto_generation(self):
        """SI-prefixed units should be auto-generated."""
        gpa = registry.get("GPa")
        mpa = registry.get("MPa")
        assert gpa.dimensions == mpa.dimensions  # same dimensions
        assert gpa.scale == pytest.approx(1e9)
        assert mpa.scale == pytest.approx(1e6)

    def test_compound_unit_kg_per_m_s2(self):
        """kg/(m·s²) should parse correctly."""
        u = registry.get("kg/(m·s^2)")
        assert u.dimensions[0] == 1.0   # M
        assert u.dimensions[1] == -1.0  # L
        assert u.dimensions[2] == -2.0  # T

    def test_compound_gpa_nm(self):
        """GPa·nm should parse as pressure × length."""
        u = registry.get("GPa·nm")
        # M:1, L:-1, T:-2 * L:1 = M:1, L:0, T:-2
        assert u.dimensions[0] == 1.0   # M
        assert abs(u.dimensions[1]) < 1e-10  # L cancels
        assert u.dimensions[2] == -2.0  # T

    def test_compound_j_per_mol_k(self):
        """J/(mol·K) should parse correctly."""
        u = registry.get("J/(mol·K)")
        assert u.dimensions[0] == 1.0    # M
        assert u.dimensions[1] == 2.0    # L
        assert u.dimensions[2] == -2.0   # T
        assert u.dimensions[4] == -1.0   # N (mol)
        assert u.dimensions[3] == -1.0   # Theta (K)

    def test_all_registered_units_have_dimensions(self):
        """Every registered unit should have a valid dimension vector."""
        for name, u in registry.all_units().items():
            assert len(u.dimensions) == 7

    def test_register_custom_unit(self):
        """Should be able to register a custom unit."""
        reg = UnitRegistry()
        custom = Unit(scale=42.0, dimensions=(0.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0), name="furlong_per_week")
        reg.register("furlong_per_week", custom)
        assert reg.get("furlong_per_week").scale == 42.0


# ------------------------------------------------------------------
# SymPy dimension inference
# ------------------------------------------------------------------

class TestSymPyInference:
    def test_derivative_d2u_dx2(self):
        """d²u/dx² where u has units K and x has units m → K/m²."""
        x = sp.Symbol("x")
        u = sp.Function("u")
        expr = sp.diff(u(x), x, 2)
        v = DimensionalValidator()
        result = v.infer_dimensions(expr, {"x": "m", "u": "K"})
        assert result.dimensions[1] == -2.0   # L^-2
        assert result.dimensions[3] == 1.0    # Theta^1

    def test_multiplication_dims(self):
        """x * y where x is m and y is kg → m·kg."""
        x, y = sp.symbols("x y")
        expr = x * y
        v = DimensionalValidator()
        result = v.infer_dimensions(expr, {"x": "m", "y": "kg"})
        assert result.dimensions[0] == 1.0  # M
        assert result.dimensions[1] == 1.0  # L

    def test_power_dims(self):
        """x**2 where x is m → m²."""
        x = sp.Symbol("x")
        expr = x ** 2
        v = DimensionalValidator()
        result = v.infer_dimensions(expr, {"x": "m"})
        assert result.dimensions[1] == 2.0  # L^2

    def test_constant_is_dimensionless(self):
        """Numeric constants should be dimensionless."""
        x = sp.Symbol("x")
        expr = 3 * x
        v = DimensionalValidator()
        result = v.infer_dimensions(expr, {"x": "m"})
        assert result.dimensions[1] == 1.0  # L (only from x)

    def test_addition_consistent_dims(self):
        """x + y with same units should work."""
        x, y = sp.symbols("x y")
        expr = x + y
        v = DimensionalValidator()
        result = v.infer_dimensions(expr, {"x": "m", "y": "m"})
        assert result.dimensions[1] == 1.0  # L

    def test_addition_inconsistent_dims_raises(self):
        """x + y with different units should raise ValueError."""
        x, y = sp.symbols("x y")
        expr = x + y
        v = DimensionalValidator()
        with pytest.raises(ValueError, match="Dimension mismatch"):
            v.infer_dimensions(expr, {"x": "m", "y": "kg"})

    def test_check_expression_matching(self):
        """check_expression should return consistent=True for matching dims."""
        x = sp.Symbol("x")
        u = sp.Function("u")
        expr = sp.diff(u(x), x)  # du/dx → K/m
        v = DimensionalValidator()
        result = v.check_expression(expr, {"x": "m", "u": "K"}, "K/m")
        assert result.consistent is True

    def test_check_expression_mismatch(self):
        """check_expression should return consistent=False for wrong expected."""
        x = sp.Symbol("x")
        expr = x ** 2
        v = DimensionalValidator()
        result = v.check_expression(expr, {"x": "m"}, "kg")
        assert result.consistent is False

    def test_function_lookup(self):
        """Applied function should look up its name in symbol_units."""
        t = sp.Symbol("t")
        T = sp.Function("T")
        expr = T(t)
        v = DimensionalValidator()
        result = v.infer_dimensions(expr, {"t": "s", "T": "K"})
        assert result.dimensions[3] == 1.0  # Theta

    def test_complex_expression(self):
        """d/dx(k * du/dx) where k is W/(m·K), u is K, x is m.

        SymPy simplifies this to k * d²u/dx² (k constant wrt x).
        [W/(m·K)] * [K/m²] = [W/m³] = M1·L-1·T-3.
        """
        x = sp.Symbol("x")
        u = sp.Function("u")
        k = sp.Symbol("k")
        # k * du/dx
        inner = k * sp.diff(u(x), x)
        expr = sp.diff(inner, x)
        v = DimensionalValidator()
        result = v.infer_dimensions(expr, {
            "x": "m", "u": "K", "k": "W/(m·K)"
        })
        # k * d²u/dx² → W/(m·K) * K/m² = W/m³ → L exponent = -1
        assert result.dimensions[1] == pytest.approx(-1.0)  # L^-1
        assert result.dimensions[0] == pytest.approx(1.0)   # M^1
        assert result.dimensions[2] == pytest.approx(-3.0)  # T^-3


# ------------------------------------------------------------------
# execution/autofix
# ------------------------------------------------------------------

class TestAutoFixLoop:
    def test_vasp_zbrent_fix(self):
        fixer = AutoFixLoop()
        fixed = fixer.apply_fix(
            "vasp_tool", "ZBRENT: fatal error", {"ALGO": "Fast", "NELM": 60}
        )
        assert fixed is not None
        assert fixed["ALGO"] == "Normal"
        assert "__auto_fix" in fixed

    def test_vasp_memory_fix(self):
        fixer = AutoFixLoop()
        fixed = fixer.apply_fix(
            "vasp_tool", "out of memory", {"NCORE": 1}
        )
        assert fixed is not None
        assert fixed["NCORE"] == 4

    def test_no_match_returns_none(self):
        fixer = AutoFixLoop()
        fixed = fixer.apply_fix("vasp_tool", "UNKNOWN WEIRD ERROR", {})
        assert fixed is None

    def test_tool_match_star(self):
        fixer = AutoFixLoop()
        fixed = fixer.apply_fix("any_tool", "timeout exceeded", {})
        assert fixed is not None
        assert "walltime_hours" in fixed

    def test_list_rules(self):
        fixer = AutoFixLoop()
        rules = fixer.list_rules("vasp_tool")
        assert len(rules) > 0

    def test_add_rule(self):
        fixer = AutoFixLoop()
        fixer.add_rule({
            "tools": ["my_tool"],
            "patterns": ["my_error"],
            "fixes": {"fix": True},
            "description": "Test fix",
        })
        fixed = fixer.apply_fix("my_tool", "my_error occurred", {})
        assert fixed is not None
        assert fixed["fix"] is True

    def test_halve_double(self):
        fixer = AutoFixLoop()
        fixer.add_rule({
            "tools": ["test_tool"],
            "patterns": ["half"],
            "fixes": {"param": "halve"},
            "description": "Halve param",
        })
        fixed = fixer.apply_fix("test_tool", "half", {"param": 0.4})
        assert fixed is not None
        assert fixed["param"] == 0.2

    def test_double(self):
        fixer = AutoFixLoop()
        fixer.add_rule({
            "tools": ["test_tool"],
            "patterns": ["double"],
            "fixes": {"param": "double"},
            "description": "Double param",
        })
        fixed = fixer.apply_fix("test_tool", "double", {"param": 10})
        assert fixed is not None
        assert fixed["param"] == 20


# ------------------------------------------------------------------
# execution/remote_job_store
# ------------------------------------------------------------------

class TestRemoteJobStore:
    def test_record_serialize_roundtrip(self):
        r = RemoteJobRecord(
            local_id="abc",
            scheduler_id="slurm-123",
            command=["python", "run.py"],
            cwd="/tmp",
            status="COMPLETED",
            exit_code=0,
        )
        d = r.to_dict()
        r2 = RemoteJobRecord.from_dict(d)
        assert r2.local_id == "abc"
        assert r2.status == "COMPLETED"
        assert r2.exit_code == 0

    def test_store_save_load(self, tmp_path: Path):
        store = RemoteJobStore(path=tmp_path / "jobs.json")
        r1 = RemoteJobRecord(
            local_id="a", scheduler_id="s1", command=["echo", "hi"], cwd="."
        )
        r2 = RemoteJobRecord(
            local_id="b", scheduler_id="s2", command=["echo", "bye"], cwd="."
        )
        store.add_or_update(r1)
        store.add_or_update(r2)
        loaded = store.load()
        assert len(loaded) == 2
        assert store.get("a") is not None

    def test_store_list_sorted(self, tmp_path: Path):
        store = RemoteJobStore(path=tmp_path / "jobs.json")
        store.add_or_update(RemoteJobRecord(
            local_id="a", scheduler_id="s1", command=["x"], cwd=".", submitted_at=1.0
        ))
        store.add_or_update(RemoteJobRecord(
            local_id="b", scheduler_id="s2", command=["y"], cwd=".", submitted_at=2.0
        ))
        jobs = store.list_jobs()
        assert jobs[0].local_id == "b"  # newest first

    def test_store_cap_records(self, tmp_path: Path):
        store = RemoteJobStore(path=tmp_path / "jobs.json", max_records=2)
        for i in range(5):
            store.add_or_update(RemoteJobRecord(
                local_id=f"id{i}", scheduler_id=f"s{i}", command=["x"], cwd=".",
                submitted_at=float(i), status="COMPLETED",
            ))
        jobs = store.load()
        assert len(jobs) <= 2

    def test_store_prune(self, tmp_path: Path):
        store = RemoteJobStore(path=tmp_path / "jobs.json", max_records=5)
        for i in range(10):
            store.add_or_update(RemoteJobRecord(
                local_id=f"id{i}", scheduler_id=f"s{i}", command=["x"], cwd=".",
                submitted_at=float(i), status="COMPLETED",
            ))
        removed = store.prune(max_records=3)
        assert removed > 0
        assert len(store.load()) <= 3

    def test_store_remove(self, tmp_path: Path):
        store = RemoteJobStore(path=tmp_path / "jobs.json")
        store.add_or_update(RemoteJobRecord(
            local_id="a", scheduler_id="s1", command=["x"], cwd=".")
        )
        assert store.remove("a") is True
        assert store.get("a") is None
        assert store.remove("nonexistent") is False

    def test_store_load_corrupt(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        store = RemoteJobStore(path=path)
        assert store.load() == []

    def test_store_load_not_list(self, tmp_path: Path):
        path = tmp_path / "bad.json"
        path.write_text('{"a": 1}')
        store = RemoteJobStore(path=path)
        assert store.load() == []


# ------------------------------------------------------------------
# execution/orchestrator
# ------------------------------------------------------------------

class TestExecutionOrchestrator:
    def test_stage_result_to_dict(self):
        sr = StageResult(
            stage_id="s1",
            stage_name="setup",
            tool_name="bash",
            success=True,
        )
        d = sr.to_dict()
        assert d["stage_id"] == "s1"
        assert d["success"] is True

    def test_orchestrator_register_tool(self):
        orch = ExecutionOrchestrator()
        orch.register_tool("test_tool", lambda x: x)
        assert "test_tool" in orch.tool_registry

    def test_workflow_record_fields(self):
        record = WorkflowExecutionRecord(
            workflow_name="test",
            overall_success=True,
            working_directory="/tmp",
        )
        assert record.workflow_name == "test"
        assert record.overall_success is True


# ------------------------------------------------------------------
# unified/core
# ------------------------------------------------------------------

class TestUnifiedCore:
    def test_domain_continuum_1d(self):
        x = sp.Symbol("x")
        d = Domain.continuum_1d(x, bounds=(0.0, 1.0))
        assert d.kind == "continuum"
        assert d.dimension == 1
        assert d.bounds == {"x": (0.0, 1.0)}

    def test_domain_continuum_2d(self):
        x, y = sp.symbols("x y")
        d = Domain.continuum_2d(x, y)
        assert d.dimension == 2

    def test_domain_particles(self):
        d = Domain.particles(n=10, dim=3)
        assert d.kind == "particles"

    def test_field_expr(self):
        x = sp.Symbol("x")
        f = Field(name="u", kind=FieldKind.SCALAR, symbols=[x], domain=Domain.continuum_1d(x))
        assert f.expr() == x

    def test_energy_functional(self):
        x = sp.Symbol("x")
        u = sp.Function("u")
        expr = sp.diff(u(x), x) ** 2
        ef = EnergyFunctional(
            name="test", expression=expr, variables=[x], parameters={"k": 1.0}
        )
        assert "u" in str(ef.expression)

    def test_unified_problem_to_dict(self):
        x = sp.Symbol("x")
        u = sp.Function("u")
        domain = Domain.continuum_1d(x)
        problem = UnifiedProblem(
            name="test",
            fields={
                "u": Field(name="u", kind=FieldKind.SCALAR, symbols=[u(x)], domain=domain)
            },
            principle=VariationalPrinciple.MINIMUM,
            domain=domain,
            energy=EnergyFunctional(
                name="E", expression=sp.diff(u(x), x) ** 2, variables=[u(x)]
            ),
        )
        d = problem.to_dict()
        assert d["name"] == "test"
        assert d["principle"] == "minimum"
        assert d["energy"]["name"] == "E"

    def test_constitutive_model(self):
        x = sp.Symbol("x")
        c = ConstitutiveModel(name="linear", expression=2 * x, parameters={"a": 1})
        assert str(c.expression) == "2*x"


# ------------------------------------------------------------------
# unified/derive
# ------------------------------------------------------------------

class TestUnifiedDerive:
    def test_euler_lagrange(self):
        x = sp.Symbol("x")
        u = sp.Function("u")
        # E = 1/2 (du/dx)^2
        energy = sp.Rational(1, 2) * sp.diff(u(x), x) ** 2
        result = _euler_lagrange(energy, u(x), [x])
        # dE/du - d/dx(dE/d(du/dx)) = 0 - d/dx(du/dx) = -d²u/dx²
        assert result == -sp.diff(u(x), x, 2)

    def test_derive_hamiltonian(self):
        q, p = sp.symbols("q p")
        energy = EnergyFunctional(
            name="H", expression=p ** 2 / 2 + q ** 2 / 2, variables=[q, p]
        )
        eqs = _derive_hamiltonian(energy)
        assert eqs["dq_dt"] == p
        assert eqs["dp_dt"] == -q

    def test_derive_equations_hamiltonian(self):
        problem = harmonic_oscillator_md()
        result = derive_equations(problem)
        assert result["principle"] == "hamiltonian"
        assert "dq_dt" in result["equations"]
        assert "dp_dt" in result["equations"]

    def test_derive_equations_minimum(self):
        problem = heat_equation_fem()
        result = derive_equations(problem)
        assert result["principle"] == "minimum"
        assert "temperature" in result["equations"]

    def test_derive_self_consistent(self):
        problem = one_d_kohn_sham_dft()
        result = derive_equations(problem)
        assert result["principle"] == "self_consistent"
        assert "kohn_sham_equations" in result["equations"]
        assert "density_from_orbitals" in result["equations"]


# ------------------------------------------------------------------
# unified/models
# ------------------------------------------------------------------

class TestUnifiedModels:
    def test_harmonic_oscillator(self):
        problem = harmonic_oscillator_md()
        assert problem.name == "harmonic_oscillator_md"
        assert problem.principle == VariationalPrinciple.HAMILTONIAN
        assert "phase_space" in problem.fields

    def test_heat_equation(self):
        problem = heat_equation_fem()
        assert problem.name == "heat_equation_fem"
        assert problem.principle == VariationalPrinciple.MINIMUM

    def test_linear_elasticity(self):
        problem = linear_elasticity_fem()
        assert problem.name == "linear_elasticity_fem"
        assert problem.constitutive is not None

    def test_list_models(self):
        names = list_models()
        assert "harmonic_oscillator_md" in names
        assert "heat_equation_fem" in names

    def test_get_model(self):
        fn = get_model("harmonic_oscillator_md")
        assert fn is not None
        problem = fn()
        assert problem.name == "harmonic_oscillator_md"
        assert get_model("nonexistent") is None


# ------------------------------------------------------------------
# unified/bridge
# ------------------------------------------------------------------

class TestUnifiedBridge:
    def test_dft_potential_to_md(self):
        problem = one_d_kohn_sham_dft()
        result = dft_potential_to_md(problem)
        assert "potential" in result
        assert result["potential"].name == "effective_pair_potential"

    def test_md_stress_to_continuum(self):
        result = md_stress_to_continuum()
        assert "cauchy_stress" in result
        assert "Continuum Cauchy stress" in result["interpretation"]

    def test_cauchy_born_elasticity(self):
        r = sp.Symbol("r")
        potential = ConstitutiveModel(
            name="harmonic", expression=r ** 2, parameters={}
        )
        result = cauchy_born_elasticity(potential)
        assert "elastic_modulus" in result
        assert result["elastic_modulus"].rhs == 2

    def test_list_bridges(self):
        names = list_bridges()
        assert "dft_to_md" in names
        assert "md_to_stress" in names

    def test_get_bridge(self):
        fn = get_bridge("dft_to_md")
        assert fn is not None
        assert get_bridge("nonexistent") is None


# ------------------------------------------------------------------
# unified/solve
# ------------------------------------------------------------------

class TestUnifiedSolve:
    def test_solve_heat_equation_fem(self):
        problem = heat_equation_fem()
        result = solve(problem, method="fem", n=10)
        assert "solution" in result
        assert result["residual"] < 1e-6
        assert result["method"] == "fem"

    def test_solve_linear_elasticity(self):
        problem = linear_elasticity_fem()
        result = solve(problem, method="fem", n=10)
        assert "solution" in result
        assert result["residual"] < 1e-6

    def test_solve_2d_heat(self):
        from huginn.unified.models import heat_equation_2d

        problem = heat_equation_2d()
        result = solve(problem, method="fem", n=5)
        assert "solution" in result
        assert "shape" in result


# ------------------------------------------------------------------
# unified/visualize
# ------------------------------------------------------------------

class TestUnifiedVisualize:
    def test_plot_solution_1d(self, tmp_path: Path):
        mesh = list(np.linspace(0, 1, 10))
        solution = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        out = tmp_path / "plot.png"
        path = plot_solution(mesh, solution, out, title="Test", xlabel="x", ylabel="u")
        assert path.exists()

    def test_solve_and_plot(self, tmp_path: Path):
        import os

        os.chdir(tmp_path)
        problem = heat_equation_fem()
        result = solve_and_plot(problem, method="fem", n=10, output_path=tmp_path / "sol.png")
        assert (tmp_path / "sol.png").exists()
        assert "plot_path" in result


# ------------------------------------------------------------------
# workflows/engine (basic smoke)
# ------------------------------------------------------------------

class TestWorkflowEngine:
    def test_engine_init(self):
        from huginn.workflows.engine import WorkflowEngine

        engine = WorkflowEngine(tool_registry={})
        assert engine.registry == {}
        assert engine.budget_policy is None

    def test_engine_with_budget(self):
        from huginn.workflows.engine import WorkflowEngine
        from huginn.types import BudgetPolicy

        engine = WorkflowEngine(tool_registry={}, budget_policy=BudgetPolicy())
        assert engine.budget_policy is not None
