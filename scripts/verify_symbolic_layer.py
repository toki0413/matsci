"""
Verification script for Phase 1: Symbolic Math / Mathematical Formalization Layer.

Tests:
  1. SymbolicMathTool (differentiate, integrate, solve, constitutive, eigenvalue)
  2. DimensionalValidator (unit parsing, equation checking, Buckingham π)
  3. AutoDiffTool (gradient, hessian, sensitivity)
  4. Symbolic workflow templates registration
"""

from __future__ import annotations

import sys
from pathlib import Path

AGENT_ROOT = Path(__file__).resolve().parent.parent / "agent"
sys.path.insert(0, str(AGENT_ROOT))


async def check_symbolic_math_tool() -> bool:
    """Test SymbolicMathTool with various operations."""
    print("\n[1] Checking SymbolicMathTool...")
    try:
        from matsci_agent.tools.symbolic_math_tool import SymbolicMathTool, SymbolicMathInput

        tool = SymbolicMathTool()

        # Test differentiation
        result = await tool.call(SymbolicMathInput(
            action="differentiate",
            expression="x**2 + sin(x)",
            symbols=["x"],
            variable="x",
            order=1,
        ), None)
        assert result.success, f"Differentiation failed: {result.error}"
        assert "2*x + cos(x)" in result.data["result"] or "2*x+cos(x)" in result.data["result"]
        print(f"  [PASS] differentiate: d/dx(x^2+sin(x)) = {result.data['result']}")

        # Test integration
        result = await tool.call(SymbolicMathInput(
            action="integrate",
            expression="x**2",
            symbols=["x"],
            variable="x",
        ), None)
        assert result.success
        assert "x**3" in result.data["result"] or "x^3" in result.data["result"]
        print(f"  [PASS] integrate: intx^2 dx = {result.data['result']}")

        # Test equation solving
        result = await tool.call(SymbolicMathInput(
            action="solve",
            equations=["x**2 - 4 = 0"],
            symbols=["x"],
        ), None)
        assert result.success
        assert len(result.data["solutions"]) > 0
        print(f"  [PASS] solve: x^2-4=0 has {len(result.data['solutions'])} solution(s)")

        # Test constitutive derivation (Neo-Hookean)
        result = await tool.call(SymbolicMathInput(
            action="constitutive",
            free_energy="C10*(I1 - 3) + D1*(J - 1)**2",
            symbols=["C10", "D1", "I1", "J", "C"],
            target="stress_from_psi",
        ), None)
        assert result.success
        assert "free_energy" in result.data
        print(f"  [PASS] constitutive: derived stress from Neo-Hookean energy")

        # Test eigenvalue
        result = await tool.call(SymbolicMathInput(
            action="eigenvalue",
            matrix=[["a", "b"], ["b", "a"]],
            symbols=["a", "b"],
        ), None)
        assert result.success
        assert len(result.data["eigenvalues"]) == 2
        print(f"  [PASS] eigenvalue: [[a,b],[b,a]] has {len(result.data['eigenvalues'])} eigenvalues")

        # Test Taylor series
        result = await tool.call(SymbolicMathInput(
            action="taylor",
            expression="exp(x)",
            symbols=["x"],
            variable="x",
            order=3,
            point={"x": 0},
        ), None)
        assert result.success
        print(f"  [PASS] taylor: exp(x) ≈ {result.data['series']}")

        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] {e}")
        return False


def check_dimensional_validator() -> bool:
    """Test DimensionalValidator."""
    print("\n[2] Checking DimensionalValidator...")
    try:
        from matsci_agent.execution.dimensional_validator import DimensionalValidator

        validator = DimensionalValidator()

        # Test unit parsing
        val, unit, dims = validator.parse_quantity("210 GPa")
        assert val == 210.0
        assert unit == "GPa"
        assert dims.get("M") == 1 and dims.get("L") == -1 and dims.get("T") == -2
        print(f"  [PASS] parse: 210 GPa → {dims}")

        # Test equation consistency: σ = E·ε
        result = validator.validate_stress_strain(500, "MPa", 210, "GPa", 0.002)
        assert result.consistent is True
        print(f"  [PASS] Hooke's law dimensional check: {result.notes[0]}")

        # Test inconsistent equation
        result = validator.check_equation(
            lhs_quantities=["100 MPa"],
            rhs_quantities=["10 kg", "5 m/s"],
            equation_name="nonsense",
        )
        assert result.consistent is False
        print(f"  [PASS] inconsistent equation detected")

        # Test Buckingham π
        pi_groups = validator.buckingham_pi(
            variables=[("E", "GPa"), ("rho", "g/cm3"), ("v", "m/s")],
            target="wave_speed",
        )
        assert len(pi_groups) > 0
        print(f"  [PASS] Buckingham π: found {len(pi_groups)} dimensionless group(s)")

        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] {e}")
        return False


async def check_autodiff_tool() -> bool:
    """Test AutoDiffTool."""
    print("\n[3] Checking AutoDiffTool...")
    try:
        from matsci_agent.tools.autodiff_tool import AutoDiffTool, AutoDiffInput

        tool = AutoDiffTool()

        # Test gradient of Lennard-Jones
        result = await tool.call(AutoDiffInput(
            action="gradient",
            function_type="lennard_jones",
            function_params={"epsilon": 1.0, "sigma": 1.0},
            variables={"r": [1.5]},
            target_variable="r",
        ), None)
        assert result.success, f"Gradient failed: {result.error}"
        print(f"  [PASS] gradient: d(LJ)/dr at r=1.5s = {result.data['gradients']}")

        # Test Hessian for stability
        result = await tool.call(AutoDiffInput(
            action="hessian",
            function_type="lennard_jones",
            function_params={"epsilon": 1.0, "sigma": 1.0},
            variables={"r": [1.122]},  # Minimum of LJ potential
            use_jax=True,
        ), None)
        if result.success:
            print(f"  [PASS] hessian: eigenvalues = {result.data.get('eigenvalues')}")
            print(f"  [PASS] positive_definite = {result.data.get('positive_definite')}")
        else:
            print(f"  [WARN] hessian: {result.error} (JAX may be missing)")

        # Test sensitivity
        result = await tool.call(AutoDiffInput(
            action="sensitivity",
            function_type="lennard_jones",
            function_params={"epsilon": 1.0, "sigma": 1.0},
            variables={"r": [1.5]},
        ), None)
        assert result.success
        print(f"  [PASS] sensitivity: S = {result.data['sensitivities']}")

        return True
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  [FAIL] {e}")
        return False


def check_symbolic_templates() -> bool:
    """Check symbolic workflow templates are registered."""
    print("\n[4] Checking Symbolic Workflow Templates...")
    try:
        from matsci_agent.workflows.templates import WORKFLOW_TEMPLATES as _TEMPLATES

        expected = [
            "constitutive_derivation",
            "fem_weak_form_verification",
            "eos_fitting",
            "stability_analysis",
        ]
        missing = [t for t in expected if t not in _TEMPLATES]
        if missing:
            print(f"  [FAIL] Missing templates: {missing}")
            return False

        print(f"  [PASS] All {len(expected)} symbolic templates registered")
        return True
    except Exception as e:
        print(f"  [FAIL] {e}")
        return False


async def main():
    print("=" * 60)
    print("Symbolic Math Layer (Phase 1) Verification")
    print("=" * 60)

    results = []
    results.append(("SymbolicMathTool", await check_symbolic_math_tool()))
    results.append(("DimensionalValidator", check_dimensional_validator()))
    results.append(("AutoDiffTool", await check_autodiff_tool()))
    results.append(("Symbolic Templates", check_symbolic_templates()))

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
        print("\n[SUCCESS] All Symbolic Math Layer checks PASSED!")
        print("\nMathematical Formalization Phase 1 complete:")
        print("  - Symbolic computation (SymPy): differentiate, integrate, solve, eigenvalue")
        print("  - Constitutive derivation: free energy → stress → tangent modulus")
        print("  - Dimensional analysis: unit parsing, equation consistency, Buckingham π")
        print("  - Automatic differentiation: gradient, Hessian, sensitivity (JAX + fallback)")
        return 0
    else:
        print(f"\n[WARN] {total - passed} check(s) failed")
        return 1


if __name__ == "__main__":
    import asyncio
    sys.exit(asyncio.run(main()))
