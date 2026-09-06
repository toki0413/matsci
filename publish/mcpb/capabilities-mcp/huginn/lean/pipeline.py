"""Automated pipeline: Python elastic constants → Lean 4 stability verification.

Usage:
    from huginn.lean.pipeline import StabilityPipeline
    pipe = StabilityPipeline()
    result = pipe.verify_cubic({"C11": 230.0, "C12": 135.0, "C44": 117.0})
    # result.success == True, result.stdout contains "true"
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from huginn.lean.interface import LeanInterface


class StabilityPipeline:
    """Bridge between Python numerical data and Lean 4 formal verification."""

    def __init__(self, project_path: str | Path | None = None):
        if project_path is None:
            candidates = [
                Path(__file__).parent.parent.parent / "lean" / "HuginnLean",
                Path.cwd() / "lean" / "HuginnLean",
            ]
            for c in candidates:
                if (c / "lakefile.toml").exists():
                    project_path = c.resolve()
                    break
        if project_path is None:
            raise RuntimeError("HuginnLean project not found")
        self._lean = LeanInterface(project_path)

    # ------------------------------------------------------------------
    # Cubic
    # ------------------------------------------------------------------
    def verify_cubic(self, constants: dict[str, float], timeout: int = 60) -> Any:
        """Verify Born stability for a cubic crystal.

        Args:
            constants: dict with keys C11, C12, C44.
        """
        c11 = constants.get("C11", 0.0)
        c12 = constants.get("C12", 0.0)
        c44 = constants.get("C44", 0.0)
        code = f"""open HuginnLean

def mat : CubicElastic := ⟨{c11}, {c12}, {c44}⟩
#eval cubicBornStable mat
#eval cubicHillBoundsHold mat
#eval cubicZenerRatio mat
#eval cubicUniversalAnisotropy mat
"""
        return self._lean.eval_lean_code(
            code, imports=["HuginnLean.Elasticity"], timeout=timeout
        )

    # ------------------------------------------------------------------
    # Hexagonal
    # ------------------------------------------------------------------
    def verify_hexagonal(self, constants: dict[str, float], timeout: int = 60) -> Any:
        """Verify Born stability for a hexagonal crystal.

        Args:
            constants: dict with keys C11, C12, C13, C33, C44, C66.
        """
        c11 = constants.get("C11", 0.0)
        c12 = constants.get("C12", 0.0)
        c13 = constants.get("C13", 0.0)
        c33 = constants.get("C33", 0.0)
        c44 = constants.get("C44", 0.0)
        c66 = constants.get("C66", 0.0)
        code = f"""open HuginnLean

def mat : HexElastic := ⟨{c11}, {c12}, {c13}, {c33}, {c44}, {c66}⟩
#eval hexBornStable mat
#eval hexBulkModulusVoigt mat
#eval hexBulkModulusReuss mat
#eval hexShearModulusVoigt mat
#eval hexShearModulusReuss mat
#eval hexUniversalAnisotropy mat
"""
        return self._lean.eval_lean_code(
            code, imports=["HuginnLean.BornStability"], timeout=timeout
        )

    # ------------------------------------------------------------------
    # Orthorhombic
    # ------------------------------------------------------------------
    def verify_orthorhombic(
        self, constants: dict[str, float], timeout: int = 60
    ) -> Any:
        """Verify Born stability for an orthorhombic crystal.

        Args:
            constants: dict with keys C11, C22, C33, C44, C55, C66, C12, C13, C23.
        """
        fields = ["C11", "C22", "C33", "C44", "C55", "C66", "C12", "C13", "C23"]
        vals = [constants.get(f, 0.0) for f in fields]
        code = f"""open HuginnLean

def mat : OrthoElastic := ⟨{', '.join(str(v) for v in vals)}⟩
#eval orthoBornStable mat
#eval orthoBulkModulusVoigt mat
#eval orthoBulkModulusReuss mat
#eval orthoShearModulusVoigt mat
#eval orthoShearModulusReuss mat
#eval orthoUniversalAnisotropy mat
"""
        return self._lean.eval_lean_code(
            code, imports=["HuginnLean.BornStability"], timeout=timeout
        )

    # ------------------------------------------------------------------
    # Generic dispatcher
    # ------------------------------------------------------------------
    def verify(
        self, crystal_system: str, constants: dict[str, float], timeout: int = 60
    ) -> Any:
        """Dispatch to the appropriate verifier by crystal system."""
        system = crystal_system.lower()
        if system == "cubic":
            return self.verify_cubic(constants, timeout)
        if system == "hexagonal":
            return self.verify_hexagonal(constants, timeout)
        if system in ("orthorhombic", "ortho"):
            return self.verify_orthorhombic(constants, timeout)
        raise ValueError(f"Unsupported crystal system: {crystal_system}")
