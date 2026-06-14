"""End-to-end demo: FEM weak form derivation -> Lean formal verification.

This script manually orchestrates the pipeline that the Agent would normally
drive via LLM planning, allowing us to verify integration without needing
a running LLM.

Usage:
    cd agent
    python demo_fem_pipeline.py
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

# Ensure imports work when run from agent/ directory
sys.path.insert(0, str(Path(__file__).parent))

from huginn.tools.symbolic_math_tool import SymbolicMathTool, SymbolicMathInput
from huginn.tools.lean_tool import LeanTool, LeanToolInput
from huginn.types import ToolContext


def _make_ctx() -> ToolContext:
    """Create a minimal tool context for standalone demos."""
    return ToolContext(session_id="demo-session", workspace=".")


async def main() -> int:
    print("=" * 60)
    print("Huginn End-to-End Demo: FEM -> Lean Verification")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Stage 1: Symbolic derivation of 1D heat conduction weak form
    # ------------------------------------------------------------------
    print("\n[Stage 1] SymbolicMathTool: weak_form (heat_conduction)")
    sym_tool = SymbolicMathTool()

    sym_input = SymbolicMathInput(
        action="weak_form",
        target="heat_conduction",
        symbols=["u", "v", "x", "k", "f"],
        expression="",  # use default strong form
    )

    sym_result = await sym_tool.call(sym_input, _make_ctx())
    if not sym_result.success:
        print(f"  FAILED: {sym_result.error}")
        return 1

    print(f"  strong_form: {sym_result.data.get('strong_form', 'N/A')}")
    print(f"  bilinear_form: {sym_result.data.get('bilinear_form', 'N/A')[:80]}...")
    print(f"  linear_functional: {sym_result.data.get('linear_functional', 'N/A')[:80]}...")
    print(f"  element_type: {sym_result.data.get('element_type', 'N/A')}")

    # ------------------------------------------------------------------
    # Stage 2: Lean formal verification
    # ------------------------------------------------------------------
    print("\n[Stage 2] LeanTool: auto_verify (fem)")
    lean_tool = LeanTool()

    lean_input = LeanToolInput(
        action="auto_verify",
        auto_verify_action="fem",
        symbolic_result=sym_result.data,
        symbols=["u", "v", "x", "k", "f"],
    )

    lean_result = await lean_tool.call(lean_input, _make_ctx())

    print(f"  verified: {lean_result.data.get('verified', False)}")
    print(f"  elapsed: {lean_result.data.get('elapsed_seconds', 0):.2f}s")

    if lean_result.data.get("stdout"):
        stdout = lean_result.data["stdout"]
        print(f"  stdout:\n{stdout[:500]}")
        if len(stdout) > 500:
            print("  ... (truncated)")

    if lean_result.data.get("stderr"):
        stderr = lean_result.data["stderr"]
        print(f"  stderr:\n{stderr[:500]}")
        if len(stderr) > 500:
            print("  ... (truncated)")

    if not lean_result.success:
        print(f"  FAILED: {lean_result.error}")
        return 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if lean_result.data.get("verified"):
        print("Demo PASSED: Symbolic derivation -> Lean verification succeeded.")
    else:
        print("Demo FAILED: Lean verification did not succeed.")
    print("=" * 60)

    return 0 if lean_result.data.get("verified") else 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
    except KeyboardInterrupt:
        print("\nInterrupted.")
        exit_code = 130
    sys.exit(exit_code)
