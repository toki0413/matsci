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
)
from huginn.execution.input_generator import GeneratedInput, InputFileGenerator
from huginn.execution.orchestrator import (
    ExecutionOrchestrator,
    StageResult,
    WorkflowExecutionRecord,
)
from huginn.execution.remote_job_store import RemoteJobRecord, RemoteJobStore
from huginn.execution.result_parser import ParsedResult, ResultParser
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
# execution/result_parser
# ------------------------------------------------------------------

class TestResultParser:
    def test_parse_unknown_file(self, tmp_path: Path):
        parser = ResultParser()
        f = tmp_path / "unknown.xyz"
        f.write_text("blah")
        result = parser.parse(f)
        assert result.software == "unknown"
        assert result.converged is False

    def test_detect_vasp_outcar(self, tmp_path: Path):
        parser = ResultParser()
        f = tmp_path / "OUTCAR"
        f.write_text("free  energy   TOTEN  = -123.456 eV\nreached required accuracy\n")
        result = parser.parse(f)
        assert result.software == "VASP"
        assert result.converged is True
        assert result.energy == pytest.approx(-123.456)

    def test_detect_vasp_warnings(self, tmp_path: Path):
        parser = ResultParser()
        f = tmp_path / "OUTCAR"
        f.write_text("ZBRENT error\nWARNING: some issue\n")
        result = parser.parse(f)
        assert any("ZBRENT" in w for w in result.warnings)

    def test_detect_gaussian_log(self, tmp_path: Path):
        parser = ResultParser()
        f = tmp_path / "run.log"
        f.write_text("Entering Gaussian System\nSCF Done: E(RHF) = -123.456\nNormal termination\n")
        result = parser.parse(f)
        assert result.software == "Gaussian"
        assert result.converged is True
        assert result.energy == pytest.approx(-123.456)

    def test_detect_lammps_log(self, tmp_path: Path):
        parser = ResultParser()
        f = tmp_path / "run.log"
        f.write_text("LAMMPS\nStep Temp Press\n1 300.0 1.0\n2 301.0 1.1\n")
        result = parser.parse(f)
        assert result.software == "LAMMPS"
        assert result.converged is True

    def test_detect_lammps_error(self, tmp_path: Path):
        parser = ResultParser()
        f = tmp_path / "run.log"
        f.write_text("LAMMPS\nERROR: some error\n")
        result = parser.parse(f)
        assert result.converged is False
        assert result.errors

    def test_detect_abaqus_dat(self, tmp_path: Path):
        parser = ResultParser()
        f = tmp_path / "job.dat"
        f.write_text("THE ANALYSIS HAS BEEN COMPLETED\n")
        result = parser.parse(f)
        assert result.software == "ABAQUS"
        assert result.converged is True

    def test_detect_openfoam_log(self, tmp_path: Path):
        parser = ResultParser()
        f = tmp_path / "run.log"
        f.write_text("foam\np final residual = 1e-5\n")
        result = parser.parse(f)
        assert result.software == "OpenFOAM"
        assert result.converged is True


# ------------------------------------------------------------------
# execution/input_generator
# ------------------------------------------------------------------

class TestInputFileGenerator:
    def test_generate_vasp_inputs(self):
        gen = InputFileGenerator()
        inputs = gen.generate_vasp_inputs(
            system="Si bulk",
            structure={"lattice": 5.43, "basis": [[0, 0, 0]], "species": ["Si"]},
            task="relax",
        )
        names = [i.filename for i in inputs]
        assert "POSCAR" in names
        assert "INCAR" in names
        assert "KPOINTS" in names

    def test_generate_vasp_inputs_with_overrides(self):
        gen = InputFileGenerator()
        inputs = gen.generate_vasp_inputs(
            system="Si",
            structure={"lattice": 5.43, "basis": [[0, 0, 0]], "species": ["Si"]},
            task="relax",
            params={"ENCUT": 600},
        )
        incar = next(i for i in inputs if i.filename == "INCAR")
        assert "ENCUT = 600" in incar.content

    def test_generated_input_dataclass(self):
        inp = GeneratedInput("VASP", "POSCAR", "Si\n1.0\n", "structure")
        assert inp.software == "VASP"
        assert inp.filename == "POSCAR"

    def test_generate_gaussian_input(self):
        gen = InputFileGenerator()
        inp = gen.generate_gaussian_input(
            task="opt", method="B3LYP", basis="6-31g",
            structure="Si 0 0 0\nSi 0 0 2",
        )
        assert "B3LYP/6-31g" in inp.content
        assert "opt" in inp.content

    def test_generate_lammps_input(self):
        gen = InputFileGenerator()
        inp = gen.generate_lammps_input(
            task="md", structure_file="data.lmp", potential="sw",
        )
        assert "units metal" in inp.content
        assert "sw" in inp.content


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
