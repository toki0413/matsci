"""Comprehensive end-to-end demo across all 6 mathematical domains.

Validates that the symbolic -> formal verification pipeline works correctly
for every phase of huginn-Lean.

Usage:
    cd agent
    PYTHONIOENCODING=utf-8 python demo_comprehensive.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from huginn.tools.lean_tool import LeanTool, LeanToolInput
from huginn.tools.symbolic_math_tool import SymbolicMathInput, SymbolicMathTool
from huginn.types import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(session_id="demo", workspace=".")


async def run_symbolic_then_lean(
    name: str,
    action: str,
    target: str,
    symbols: list[str],
    expression: str = "",
    equations: list[str] | None = None,
    matrix: list[list[str]] | None = None,
    auto_verify: str = "",
) -> bool:
    """Generic wrapper: symbolic derivation -> Lean verification."""
    sym_tool = SymbolicMathTool()
    sym_in = SymbolicMathInput(
        action=action,
        target=target,
        symbols=symbols,
        expression=expression,
        equations=equations or [],
        matrix=matrix,
    )
    sym_res = await sym_tool.call(sym_in, _ctx())
    if not sym_res.success:
        print(f"  [{name}] Symbolic FAILED: {sym_res.error}")
        return False

    lean_tool = LeanTool()
    lean_in = LeanToolInput(
        action="auto_verify",
        auto_verify_action=auto_verify or action,
        symbolic_result=sym_res.data,
        symbols=symbols,
    )
    lean_res = await lean_tool.call(lean_in, _ctx())
    ok = (lean_res.data or {}).get("verified", False)
    elapsed = (lean_res.data or {}).get("elapsed_seconds", 0)
    print(f"  [{name}] {'PASS' if ok else 'FAIL'} ({elapsed:.2f}s)")
    if not ok and (lean_res.data or {}).get("stderr"):
        print(f"    stderr: {lean_res.data['stderr'][:200]}")
    return ok


async def main() -> int:
    print("=" * 60)
    print("Huginn Comprehensive Pipeline Demo (6 Phases)")
    print("=" * 60)

    results: dict[str, bool] = {}

    # Phase 1: FEM Weak Forms
    print("\n--- Phase 1: FEM Weak Forms ---")
    results["heat_conduction"] = await run_symbolic_then_lean(
        "Heat Conduction",
        "weak_form",
        "heat_conduction",
        ["u", "v", "x", "k", "f"],
        auto_verify="fem",
    )
    results["linear_elasticity"] = await run_symbolic_then_lean(
        "Linear Elasticity",
        "weak_form",
        "linear_elasticity",
        ["ux", "uy", "vx", "vy", "x", "y", "E", "nu"],
        auto_verify="fem",
    )
    results["bar_element"] = await run_symbolic_then_lean(
        "Bar Element",
        "weak_form",
        "assemble_element_matrix",
        ["E", "A", "h"],
        expression="bar",
        auto_verify="fem",
    )

    # Phase 2: Tensor Algebra
    print("\n--- Phase 2: Tensor Algebra ---")
    results["tensor"] = await run_symbolic_then_lean(
        "Matrix Invariants",
        "tensor_ops",
        "",
        ["a", "b", "c"],
        matrix=[["a", "b"], ["b", "c"]],
        auto_verify="tensor_ops",
    )

    # Phase 3: Numerical Linear Algebra
    print("\n--- Phase 3: Numerical Linear Algebra ---")
    results["la"] = await run_symbolic_then_lean(
        "Cholesky",
        "linear_algebra",
        "cholesky",
        ["A", "L"],
        matrix=[["4", "2"], ["2", "3"]],
        auto_verify="linear_algebra",
    )

    # Phase 4: DFT
    print("\n--- Phase 4: DFT ---")
    results["dft_fermi"] = await run_symbolic_then_lean(
        "Free Electron Fermi Energy",
        "dft",
        "fermi_energy",
        ["n", "kF", "EF"],
        expression="n=0.05",
        auto_verify="dft",
    )

    # Phase 5: Thermodynamics
    print("\n--- Phase 5: Thermodynamics ---")
    results["thermo_ideal_gas"] = await run_symbolic_then_lean(
        "Ideal Gas Pressure",
        "thermodynamics",
        "ideal_gas",
        ["n", "T", "V", "P"],
        expression="n=1.0,T=273.15,V=0.022414",
        auto_verify="thermodynamics",
    )

    # Phase 6: Probability
    print("\n--- Phase 6: Probability ---")
    results["prob_normal"] = await run_symbolic_then_lean(
        "Normal PDF",
        "probability",
        "normal_pdf",
        ["mu", "sigma", "x", "pdf"],
        expression="mu=0.0,sigma=1.0,x=0.0",
        auto_verify="probability",
    )

    # Summary
    print("\n" + "=" * 60)
    total = len(results)
    passed = sum(results.values())
    print(f"Results: {passed}/{total} pipelines passed")
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        code = 130
    sys.exit(code)
