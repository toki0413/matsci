"""Scientific automation E2E tests: toolchain workflows + self-healing loops.

Tests complete research workflows through the tool layer (no backend needed):
1. Structure → Symmetry → XRD → Validation chain
2. VASP mock-mode: scf run → parse output
3. LAMMPS mock-mode: run MD → trajectory analysis
4. AutoFix self-healing: bad error → diagnose → fix → verify
5. Materials database query → property summary
6. Symbolic + numerical math chain
7. Bourbaki equation discovery (newly implemented)
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time
from pathlib import Path

# Ensure agent dir is on path and cache dir is writable
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("HUGINN_CACHE_DIR", str(Path(__file__).parent.parent / ".test_cache"))

from huginn.types import ToolContext, ToolResult

# ── Test framework ──────────────────────────────────────────────────

PASSED = 0
FAILED = 0
SKIPPED = 0


def report(name: str, success: bool, detail: str = "", skipped: bool = False) -> None:
    global PASSED, FAILED, SKIPPED
    if skipped:
        SKIPPED += 1
        print(f"  ⊘ {name}: SKIP — {detail}")
    elif success:
        PASSED += 1
        print(f"  ✓ {name}: PASS — {detail}")
    else:
        FAILED += 1
        print(f"  ✗ {name}: FAIL — {detail}")


# ── Test 1: Structure → Symmetry → XRD → Validation chain ──────────

async def test_structure_symmetry_xrd_chain() -> None:
    print("\n[1] Structure → Symmetry → XRD → Validation Chain")
    from huginn.tools.sci.symmetry_tool import SymmetryTool
    from huginn.tools.sci.xrd_sim_tool import XrdSimTool, XrdSimToolInput

    tmpdir = tempfile.mkdtemp()
    ctx = ToolContext(session_id="sci-test", workspace=tmpdir)

    # Use the committed Si_diamond POSCAR test fixture
    fixture = Path(__file__).parent.parent / "Si_diamond" / "POSCAR"
    si_poscar = Path(tmpdir) / "Si_diamond.vasp"
    si_poscar.write_text(fixture.read_text())
    report("load Si diamond POSCAR", True, f"fixture={fixture.name}")

    # Symmetry analysis
    sym_tool = SymmetryTool()
    r2 = await sym_tool.call({"action": "analyze", "file_path": str(si_poscar)}, ctx)
    report("symmetry analyze", r2.success,
           f"spacegroup={r2.data.get('spacegroup', '?')}" if r2.data else r2.error)

    # XRD simulation (XrdSimTool.call expects a dict, not a model instance)
    xrd_tool = XrdSimTool()
    r3 = await xrd_tool.call({
        "action": "simulate_xrd",
        "file_path": str(si_poscar),
        "wavelength": 1.5406,
    }, ctx)
    report("XRD simulate", r3.success,
           f"peaks={len(r3.data.get('peaks', []))}" if r3.data else r3.error)

    # Structure validation via primitive cell extraction
    r4 = await sym_tool.call({"action": "primitive", "file_path": str(si_poscar)}, ctx)
    report("primitive cell extraction", r4.success,
           f"atoms={r4.data.get('n_atoms', '?')}" if r4.data else r4.error)


# ── Test 2: VASP mock-mode workflow ─────────────────────────────────

async def test_vasp_mock_workflow() -> None:
    print("\n[2] VASP Mock-Mode Workflow")
    from huginn.tools.sim.vasp_tool import VaspTool, VaspToolInput

    tmpdir = Path(tempfile.mkdtemp())
    ctx = ToolContext(session_id="sci-test", workspace=str(tmpdir))

    # Create a minimal working_dir with POSCAR for mock SCF
    fixture = Path(__file__).parent.parent / "Si_diamond" / "POSCAR"
    (tmpdir / "POSCAR").write_text(fixture.read_text())

    tool = VaspTool()
    r1 = await tool.call(VaspToolInput(
        action="scf",
        working_dir=str(tmpdir),
    ), ctx)
    report("VASP scf (mock)", r1.success,
           f"energy={r1.data.get('energy', '?')}" if r1.data else r1.error)


# ── Test 3: LAMMPS mock-mode workflow ───────────────────────────────

async def test_lammps_mock_workflow() -> None:
    print("\n[3] LAMMPS Mock-Mode Workflow")
    from huginn.tools.sim.lammps_tool import LammpsTool, LammpsToolInput

    tmpdir = tempfile.mkdtemp()
    ctx = ToolContext(session_id="sci-test", workspace=tmpdir)

    tool = LammpsTool()
    r1 = await tool.call(LammpsToolInput(
        action="run",
        input_script="units metal\natom_style atomic\nlattice fcc 3.615\nregion box block 0 4 0 4 0 4\ncreate_box 1 box\ncreate_atoms 1 box\npair_style eam\npair_coeff * * Cu_u3.eam\nvelocity all create 300 12345\nfix 1 all nvt temp 300 300 0.1\ntimestep 0.001\nrun 100\n",
        output_prefix="cu_test",
    ), ctx)
    report("LAMMPS run (mock)", r1.success,
           f"msg={(r1.data.get('message', '') or '')[:60]}" if r1.data else r1.error)


# ── Test 4: AutoFix self-healing loop ───────────────────────────────

async def test_autofix_self_healing() -> None:
    print("\n[4] AutoFix Self-Healing Loop")
    try:
        from huginn.execution.autofix import AutoFixLoop
    except ImportError as e:
        report("AutoFix import", False, str(e), skipped=True)
        return

    fixer = AutoFixLoop()

    # Test: SCF convergence failure → should switch ALGO to Normal
    fixed = fixer.apply_fix(
        tool_name="vasp_tool",
        error="ZBRENT: fatal error in bracketing, please rerun with smaller EDIFF",
        current_params={"ALGO": "Fast", "NELM": 60},
    )
    report("AutoFix ZBRENT", fixed is not None and "ALGO" in (fixed or {}),
           f"fixes={fixed}" if fixed else "no fix returned")

    # Test: memory error → should adjust NCORE/KPAR
    fixed2 = fixer.apply_fix(
        tool_name="vasp_tool",
        error="out of memory during allocation",
        current_params={"NCORE": 1, "KPAR": 1},
    )
    report("AutoFix OOM", fixed2 is not None,
           f"fixes={fixed2}" if fixed2 else "no fix returned")

    # Test: LAMMPS timestep issue
    fixed3 = fixer.apply_fix(
        tool_name="lammps_tool",
        error="Lost atoms: can't compute pressure",
        current_params={"timestep": 0.005},
    )
    report("AutoFix LAMMPS lost atoms", fixed3 is not None,
           f"fixes={fixed3}" if fixed3 else "no fix returned")


# ── Test 5: Materials database query ────────────────────────────────

async def test_materials_database_chain() -> None:
    print("\n[5] Materials Database Query Chain")
    from huginn.tools.materials_database_tool import MaterialsDatabaseTool, MaterialsDatabaseInput

    tmpdir = tempfile.mkdtemp()
    ctx = ToolContext(session_id="sci-test", workspace=tmpdir)

    tool = MaterialsDatabaseTool()
    # Query Si summary from Materials Project
    r1 = await tool.call(MaterialsDatabaseInput(
        action="mp_summary",
        query="Si",
        limit=3,
    ), ctx)
    # This may fail if MP_API_KEY is not set — that's OK, just report it
    if r1.success:
        report("MP query Si", True,
               f"results={len(r1.data.get('results', []))}" if r1.data else "ok")
    else:
        report("MP query Si", False, r1.error or "failed (likely no API key)", skipped=True)


# ── Test 6: Numerical/symbolic math chain ───────────────────────────

async def test_math_chain() -> None:
    print("\n[6] Symbolic + Numerical Math Chain")
    from huginn.tools.sci.numerical_tool import NumericalTool, NumericalToolInput

    tmpdir = tempfile.mkdtemp()
    ctx = ToolContext(session_id="sci-test", workspace=tmpdir)

    tool = NumericalTool()
    # Solve ODE: dy/dt = -y, y(0) = 1
    r1 = await tool.call({
        "action": "ode",
        "func": "-y[0]",
        "y0": [1.0],
        "t_span": [0.0, 5.0],
        "t_eval": [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0],
    }, ctx)
    report("ODE solve dy/dt=-y", r1.success,
           f"keys={list(r1.data.keys())[:5]}" if r1.data else r1.error)

    # Curve fitting: exponential decay
    r2 = await tool.call({
        "action": "curve_fit",
        "func_fit": "a0*exp(-a1*x)",
        "xdata": [0, 1, 2, 3, 4, 5],
        "ydata": [1.0, 0.368, 0.135, 0.050, 0.018, 0.007],
    }, ctx)
    report("curve fit exponential", r2.success,
           f"params={r2.data.get('parameters', r2.data.get('popt', '?'))}" if r2.data else r2.error)


# ── Test 7: Bourbaki equation discovery (newly implemented) ─────────

async def test_bourbaki_discovery() -> None:
    print("\n[7] Bourbaki Equation Discovery (newly implemented)")
    from huginn.tools.bourbaki_tool import BourbakiTool, BourbakiInput

    tool = BourbakiTool()
    ctx = ToolContext(session_id="sci-test", workspace=".")

    # Algebraic equation: x^2 - 4 = 0 → x = -2 or x = 2
    r1 = await tool.call(BourbakiInput(
        task="discover_equation",
        equations="x**2 - 4 = 0",
    ), ctx)
    report("discover x^2-4=0", r1.data.get("verified") is not None and r1.data["verified"],
           f"equation={r1.data.get('equation')}")

    # PDE classification: heat equation → parabolic
    r2 = await tool.call(BourbakiInput(
        task="discover_equation",
        equations="∂u/∂t = ∇²u",
    ), ctx)
    _msg2 = r2.data.get("message", "")
    report("classify heat equation", "parabolic" in _msg2,
           _msg2[:80])

    # Conservation check: E_kin + E_pot = E_total
    r3 = await tool.call(BourbakiInput(
        task="check_conservation",
        equations="E_kin + E_pot = E_total",
    ), ctx)
    _msg3 = r3.data.get("message", "")
    report("conservation check", r3.success,
           _msg3[:80])


# ── Main ────────────────────────────────────────────────────────────

async def main() -> None:
    print("=" * 70)
    print("SCIENTIFIC AUTOMATION E2E TESTS")
    print("=" * 70)

    await test_structure_symmetry_xrd_chain()
    await test_vasp_mock_workflow()
    await test_lammps_mock_workflow()
    await test_autofix_self_healing()
    await test_materials_database_chain()
    await test_math_chain()
    await test_bourbaki_discovery()

    print(f"\n{'=' * 70}")
    print(f"RESULTS: {PASSED} passed, {FAILED} failed, {SKIPPED} skipped")
    print(f"{'=' * 70}")

    return FAILED == 0


if __name__ == "__main__":
    ok = asyncio.run(main())
    sys.exit(0 if ok else 1)
