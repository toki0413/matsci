"""Parallel pipeline demo — run all 8 verification pipelines concurrently.

Demonstrates that independent symbolic -> Lean workflows can execute in
parallel using a thread pool, reducing wall-clock time significantly.

Usage:
    cd agent
    PYTHONIOENCODING=utf-8 python demo_parallel.py
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from matsci_agent.tools.symbolic_math_tool import SymbolicMathTool, SymbolicMathInput
from matsci_agent.tools.lean_tool import LeanTool, LeanToolInput
from matsci_agent.types import ToolContext


def _ctx() -> ToolContext:
    return ToolContext(session_id="demo", workspace=".")


def run_pipeline_sync(name: str, action: str, target: str, symbols: list[str],
                      expression: str = "", equations: list[str] | None = None,
                      matrix: list[list[str]] | None = None,
                      auto_verify: str = "") -> tuple[str, bool, float]:
    """Synchronous wrapper for a single pipeline."""
    t0 = time.time()

    async def _inner():
        sym_tool = SymbolicMathTool()
        sym_in = SymbolicMathInput(
            action=action, target=target, symbols=symbols,
            expression=expression, equations=equations or [],
            matrix=matrix,
        )
        sym_res = await sym_tool.call(sym_in, _ctx())
        if not sym_res.success:
            return False

        lean_tool = LeanTool()
        lean_in = LeanToolInput(
            action="auto_verify",
            auto_verify_action=auto_verify or action,
            symbolic_result=sym_res.data,
            symbols=symbols,
        )
        lean_res = await lean_tool.call(lean_in, _ctx())
        return (lean_res.data or {}).get("verified", False)

    # Each thread gets its own event loop
    ok = asyncio.run(_inner())
    return name, ok, time.time() - t0


def main() -> int:
    print("=" * 60)
    print("MatSci-Agent Parallel Pipeline Demo")
    print("=" * 60)

    tasks = [
        ("Heat Conduction", "weak_form", "heat_conduction",
         ["u", "v", "x", "k", "f"], "", None, None, "fem"),
        ("Linear Elasticity", "weak_form", "linear_elasticity",
         ["ux", "uy", "vx", "vy", "x", "y", "E", "nu"], "", None, None, "fem"),
        ("Bar Element", "weak_form", "assemble_element_matrix",
         ["E", "A", "h"], "bar", None, None, "fem"),
        ("Tensor Invariants", "tensor_ops", "",
         ["a", "b", "c"], "", None, [["a", "b"], ["b", "c"]], "tensor_ops"),
        ("Cholesky", "linear_algebra", "cholesky",
         ["A", "L"], "", None, [["4", "2"], ["2", "3"]], "linear_algebra"),
        ("Free Electron Fermi Energy", "dft", "fermi_energy",
         ["n", "kF", "EF"], "n=0.05", None, None, "dft"),
        ("Ideal Gas Pressure", "thermodynamics", "ideal_gas",
         ["n", "T", "V", "P"], "n=1.0,T=273.15,V=0.022414", None, None, "thermodynamics"),
        ("Normal PDF", "probability", "normal_pdf",
         ["mu", "sigma", "x", "pdf"], "mu=0.0,sigma=1.0,x=0.0", None, None, "probability"),
    ]

    print(f"\nLaunching {len(tasks)} pipelines in thread pool...\n")
    overall_t0 = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(run_pipeline_sync, *t) for t in tasks]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]

    overall_elapsed = time.time() - overall_t0

    # Sort results by name for consistent display
    results.sort(key=lambda x: x[0])
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)

    print("\n" + "=" * 60)
    print("Results")
    print("=" * 60)
    for name, ok, elapsed in results:
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status:<8} {name:<35} ({elapsed:.2f}s)")

    print("-" * 60)
    print(f"  Total: {passed}/{total} passed")
    print(f"  Wall-clock time: {overall_elapsed:.2f}s")
    seq_estimate = sum(elapsed for _, _, elapsed in results)
    print(f"  Sequential estimate: {seq_estimate:.2f}s")
    print(f"  Speedup: {seq_estimate / max(overall_elapsed, 0.01):.1f}x")
    print("=" * 60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    try:
        code = main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
        code = 130
    sys.exit(code)
