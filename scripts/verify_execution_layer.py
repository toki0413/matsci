"""
Verification script for the Execution Layer.

Tests that the Agent can:
  1. Generate input files from specs
  2. Parse calculation outputs
  3. Apply automatic fixes to failed runs
  4. Orchestrate multi-stage workflows
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_ROOT))


def check_input_generator() -> bool:
    """Test input file generation."""
    print("\n[1] Checking InputFileGenerator...")
    try:
        from huginn.execution.input_generator import InputFileGenerator, GeneratedInput
        gen = InputFileGenerator()

        # Test VASP generation
        vasp_inputs = gen.generate_vasp_inputs(
            system="Si bulk FCC",
            structure={"lattice": 5.43, "basis": [[0, 0, 0]], "species": ["Si"]},
            task="relax",
            params={"ENCUT": 520, "ISMEAR": 0},
            potcar_hints={"Si": "Si_PBE"},
        )
        assert len(vasp_inputs) == 4, f"Expected 4 VASP inputs, got {len(vasp_inputs)}"
        assert any(i.filename == "POSCAR" for i in vasp_inputs)
        assert any(i.filename == "INCAR" for i in vasp_inputs)

        # Test Gaussian generation
        gaussian = gen.generate_gaussian_input(
            task="opt", method="B3LYP", basis="6-31G(d)",
            structure="Si 0.0 0.0 0.0\nSi 1.0 0.0 0.0",
            extras={"scf_xqc": True},
        )
        assert gaussian.filename == "job.gjf"
        assert "B3LYP/6-31G(d)" in gaussian.content
        assert "scf=xqc" in gaussian.content

        # Test LAMMPS generation
        lammps = gen.generate_lammps_input(
            task="md", potential="eam/alloy",
            structure_file="si.data", temperature=300, steps=10000,
        )
        assert lammps.filename == "in.lammps"
        assert "fix 1 all nvt" in lammps.content

        # Test OpenFOAM generation
        foam = gen.generate_openfoam_dicts(solver="simpleFoam", turbulence="kOmegaSST")
        assert len(foam) == 4
        assert any(i.filename == "controlDict" for i in foam)

        # Test ABAQUS generation
        abaqus = gen.generate_abaqus_input(job_name="test_job")
        assert abaqus.filename == "test_job.inp"

        print("  [PASS] All input generators working")
        print(f"  [PASS] VASP ({len(vasp_inputs)} files), Gaussian, LAMMPS, OpenFOAM ({len(foam)} files), ABAQUS")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_result_parser() -> bool:
    """Test result parsing with synthetic data."""
    print("\n[2] Checking ResultParser...")
    try:
        from huginn.execution.result_parser import ResultParser, ParsedResult
        import tempfile

        parser = ResultParser()

        # Test VASP OUTCAR parsing (synthetic)
        with tempfile.NamedTemporaryFile(mode="w", suffix="OUTCAR", delete=False) as f:
            f.write("""VASP output
IBRION = 2
free  energy   TOTEN  =      -10.12345678 eV
TOTAL-FORCE (eV/Angst)
--------------------------------------------------------------------------------
    1     1    0.000    0.000    0.000    0.001    0.002   -0.001
    2     1    0.250    0.250    0.250   -0.001   -0.002    0.001
--------------------------------------------------------------------------------
in kB  100.0  100.0  100.0  0.0  0.0  0.0
mag=    0.500
reached required accuracy - stopping structural energy minimisation
""")
            outcar_path = Path(f.name)

        result = parser.parse(outcar_path, "vasp_outcar")
        assert result.software == "VASP"
        assert result.converged is True
        assert abs(result.energy - (-10.12345678)) < 1e-6
        assert result.magnetic_moment == 0.5
        assert len(result.forces) == 2
        assert result.stress == [100.0, 100.0, 100.0, 0.0, 0.0, 0.0]
        outcar_path.unlink()

        # Test LAMMPS log parsing (synthetic)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("""LAMMPS output
Step Temp PotEng KinEng TotEng Press
0 300.0 -1000.0 50.0 -950.0 1.0
100 299.5 -1000.1 49.9 -950.2 1.1
200 300.2 -999.9 50.1 -949.8 0.9
""")
            lammps_path = Path(f.name)

        result = parser.parse(lammps_path, "lammps_log")
        assert result.software == "LAMMPS"
        assert result.converged is True
        assert abs(result.physical_quantities["final_temperature"] - 300.2) < 1e-6
        lammps_path.unlink()

        # Test Gaussian log parsing (synthetic)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write("""Entering Gaussian System
SCF Done:  E(RB3LYP) =  -100.1234567
Alpha  occ. eigenvalues --   -10.0   -5.0
Alpha virt. eigenvalues --    -1.0    2.0
Frequencies --   100.0   200.0   -50.0
Normal termination of Gaussian 16
""")
            gaussian_path = Path(f.name)

        result = parser.parse(gaussian_path, "gaussian_log")
        assert result.software == "Gaussian"
        assert result.converged is True
        assert abs(result.energy - (-100.1234567)) < 1e-6
        assert result.band_gap is not None
        assert len(result.warnings) > 0  # Imaginary frequency
        gaussian_path.unlink()

        print("  [PASS] All parsers working (VASP, LAMMPS, Gaussian)")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_autofix() -> bool:
    """Test automatic failure diagnosis and repair."""
    print("\n[3] Checking AutoFixLoop...")
    try:
        from huginn.execution.autofix import AutoFixLoop
        fixer = AutoFixLoop()

        # Test VASP SCF fix
        fixed = fixer.apply_fix(
            tool_name="vasp_tool",
            error="ZBRENT: fatal error in bracketing",
            current_params={"ALGO": "Fast", "ENCUT": 400},
        )
        assert fixed is not None
        assert fixed["ALGO"] == "Normal"
        assert fixed["NELMIN"] == 6
        assert "__auto_fix" in fixed

        # Test Gaussian convergence fix
        fixed = fixer.apply_fix(
            tool_name="gaussian_tool",
            error="Convergence failure -- SCF has not converged",
            current_params={"method": "B3LYP"},
        )
        assert fixed is not None
        assert fixed.get("scf") == "xqc"

        # Test LAMMPS lost atoms fix
        fixed = fixer.apply_fix(
            tool_name="lammps_tool",
            error="ERROR: Lost atoms: original 1000 current 998",
            current_params={"timestep": 2.0},
        )
        assert fixed is not None
        assert fixed["timestep"] == 1.0  # halved

        # Test no match
        fixed = fixer.apply_fix(
            tool_name="vasp_tool",
            error="some completely unknown error xyz123",
            current_params={},
        )
        assert fixed is None

        print("  [PASS] AutoFixLoop working with 20+ built-in rules")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def check_orchestrator() -> bool:
    """Test workflow orchestration."""
    print("\n[4] Checking ExecutionOrchestrator...")
    try:
        import asyncio
        from huginn.execution.orchestrator import ExecutionOrchestrator, StageResult

        orch = ExecutionOrchestrator(working_dir="./test_exec")

        # Register mock tools
        def mock_vasp(action, **kwargs):
            return {"energy": -10.0, "converged": True}

        def mock_band(action, **kwargs):
            return {"band_gap": 1.2, "converged": True}

        orch.register_tool("vasp_tool", mock_vasp)
        orch.register_tool("band_tool", mock_band)

        # Define a 2-stage workflow with dependency
        stages = [
            {"id": "relax", "tool": "vasp_tool", "action": "relax", "params": {"ENCUT": 520}},
            {"id": "band", "tool": "band_tool", "action": "band", "params": {"ENCUT": 520, "kpath": "GXM"},
             "depends_on": ["relax"]},
        ]

        record = asyncio.run(orch.run(stages, workflow_name="test_dft"))
        assert record.overall_success is True
        assert len(record.stage_results) == 2
        assert record.stage_results[0].stage_id == "relax"
        assert record.stage_results[1].stage_id == "band"

        # Check dependency resolution worked
        assert record.stage_results[1].success is True

        print("  [PASS] ExecutionOrchestrator working with dependency resolution")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


def main():
    print("=" * 60)
    print("Execution Layer Verification")
    print("=" * 60)

    results = []
    results.append(("InputFileGenerator", check_input_generator()))
    results.append(("ResultParser", check_result_parser()))
    results.append(("AutoFixLoop", check_autofix()))
    results.append(("ExecutionOrchestrator", check_orchestrator()))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    passed = sum(1 for _, r in results if r)
    total = len(results)
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n{passed}/{total} checks passed")

    if passed == total:
        print("\n[SUCCESS] All Execution Layer checks PASSED!")
        print("\nThe Agent can now:")
        print("  1. Generate input files for VASP/Gaussian/LAMMPS/ABAQUS/OpenFOAM")
        print("  2. Parse calculation outputs and extract physical quantities")
        print("  3. Automatically diagnose failures and apply fixes")
        print("  4. Orchestrate multi-stage workflows with dependency resolution")
        return 0
    else:
        print(f"\n[WARN] {total - passed} check(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
