"""Mechanical property calculations and stability criteria.

Elastic constants, Born stability criteria, and derived moduli.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass
class ElasticTensor:
    """Elastic stiffness tensor (Voigt notation, 6x6 matrix)."""

    C: np.ndarray  # 6x6 matrix in GPa

    def __post_init__(self):
        if self.C.shape != (6, 6):
            raise ValueError("Elastic tensor must be 6x6")

    def voigt_moduli(self) -> dict[str, float]:
        """Calculate elastic moduli using Voigt average (upper bound)."""
        C = self.C

        # Bulk modulus (Voigt)
        Kv = (C[0, 0] + C[1, 1] + C[2, 2] + 2 * (C[0, 1] + C[1, 2] + C[2, 0])) / 9.0

        # Shear modulus (Voigt)
        Gv = (
            C[0, 0]
            + C[1, 1]
            + C[2, 2]
            - C[0, 1]
            - C[1, 2]
            - C[2, 0]
            + 3 * (C[3, 3] + C[4, 4] + C[5, 5])
        ) / 15.0

        # Young's modulus (approximate from Voigt)
        Ev = 9 * Kv * Gv / (3 * Kv + Gv)

        # Poisson's ratio
        nu = (3 * Kv - 2 * Gv) / (2 * (3 * Kv + Gv))

        return {
            "bulk_modulus_voigt": float(Kv),
            "shear_modulus_voigt": float(Gv),
            "youngs_modulus": float(Ev),
            "poisson_ratio": float(nu),
        }

    def reuss_moduli(self, S: np.ndarray | None = None) -> dict[str, float]:
        """Calculate elastic moduli using Reuss average (lower bound)."""
        if S is None:
            S = np.linalg.inv(self.C)

        Kr = 1.0 / (S[0, 0] + S[1, 1] + S[2, 2] + 2 * (S[0, 1] + S[1, 2] + S[2, 0]))
        Gr = 15.0 / (
            4 * (S[0, 0] + S[1, 1] + S[2, 2])
            - 4 * (S[0, 1] + S[1, 2] + S[2, 0])
            + 3 * (S[3, 3] + S[4, 4] + S[5, 5])
        )

        Er = 9 * Kr * Gr / (3 * Kr + Gr)
        nu = (3 * Kr - 2 * Gr) / (2 * (3 * Kr + Gr))

        return {
            "bulk_modulus_reuss": float(Kr),
            "shear_modulus_reuss": float(Gr),
            "youngs_modulus": float(Er),
            "poisson_ratio": float(nu),
        }

    def hill_moduli(self) -> dict[str, float]:
        """Calculate Hill average moduli (Voigt-Reuss average)."""
        voigt = self.voigt_moduli()
        reuss = self.reuss_moduli()

        return {
            "bulk_modulus_hill": (
                voigt["bulk_modulus_voigt"] + reuss["bulk_modulus_reuss"]
            )
            / 2,
            "shear_modulus_hill": (
                voigt["shear_modulus_voigt"] + reuss["shear_modulus_reuss"]
            )
            / 2,
            "youngs_modulus": (voigt["youngs_modulus"] + reuss["youngs_modulus"]) / 2,
            "poisson_ratio": (voigt["poisson_ratio"] + reuss["poisson_ratio"]) / 2,
        }


class BornStabilityChecker:
    """Check mechanical stability using Born criteria for various crystal systems."""

    @classmethod
    def check(
        cls,
        C: np.ndarray,
        crystal_system: Literal[
            "cubic",
            "hexagonal",
            "trigonal",
            "tetragonal",
            "orthorhombic",
            "monoclinic",
            "triclinic",
            "auto",
        ] = "auto",
    ) -> dict:
        """Check mechanical stability.

        Returns dict with:
            - stable: bool
            - criteria: list of (name, value, threshold, passed)
            - crystal_system: detected or provided
        """
        if crystal_system == "auto":
            crystal_system = cls._detect_system(C)

        checker = getattr(cls, f"_check_{crystal_system}", None)
        if checker is None:
            # 未知晶系不能默认判 unstable — 这会让真实存在的 tetragonal/
            # trigonal/monoclinic 材料被错误拒绝. 改为 None + warning,
            # 让调用方知道"未实现"而非"不稳定". 升级路径是补 _check_tetragonal
            # 等方法 (Mouhat & Coudert PRB 2014 给出了所有晶系的判据).
            return {
                "stable": None,
                "error": (
                    f"Born criteria for {crystal_system} not implemented; "
                    f"cannot judge stability. "
                    f"Implemented: cubic, hexagonal, orthorhombic, triclinic. "
                    f"See Mouhat & Coudert PRB 90, 224104 (2014) for full table."
                ),
                "crystal_system": crystal_system,
            }

        return checker(C)

    @classmethod
    def _detect_system(cls, C: np.ndarray) -> str:
        """Auto-detect crystal system from elastic tensor pattern."""
        # Simple heuristic based on symmetry of C matrix
        # This is approximate — user should ideally specify

        # Check if cubic-like (C11=C22=C33, C12=C13=C23, C44=C55=C66, others=0)
        tol = 5.0  # GPa tolerance

        if (
            abs(C[0, 0] - C[1, 1]) < tol
            and abs(C[1, 1] - C[2, 2]) < tol
            and abs(C[0, 1] - C[0, 2]) < tol
            and abs(C[0, 2] - C[1, 2]) < tol
            and abs(C[3, 3] - C[4, 4]) < tol
            and abs(C[4, 4] - C[5, 5]) < tol
        ):
            # Check off-diagonals are near zero
            off_diag_zeros = all(
                abs(C[i, j]) < tol
                for i in range(6)
                for j in range(6)
                if i != j and not (i < 3 and j < 3)
            )
            if off_diag_zeros:
                return "cubic"

        # Default fallback
        return "triclinic"

    @classmethod
    def _check_cubic(cls, C: np.ndarray) -> dict:
        """Born criteria for cubic crystals."""
        C11, C12, C44 = C[0, 0], C[0, 1], C[3, 3]

        criteria = [
            ("C11 > |C12|", C11, abs(C12), abs(C12) < C11),
            ("C11 + 2*C12 > 0", C11 + 2 * C12, 0, C11 + 2 * C12 > 0),
            ("C44 > 0", C44, 0, C44 > 0),
        ]

        stable = all(passed for _, _, _, passed in criteria)

        return {
            "stable": stable,
            "criteria": [
                {
                    "name": name,
                    "value": float(val),
                    "threshold": float(th),
                    "passed": passed,
                }
                for name, val, th, passed in criteria
            ],
            "crystal_system": "cubic",
        }

    @classmethod
    def _check_hexagonal(cls, C: np.ndarray) -> dict:
        """Born criteria for hexagonal crystals.

        Ref: Mouhat & Coudert, PRB 90, 224104 (2014), Eq. (60).
        Necessary and sufficient conditions for hexagonal/trigonal (with 6-fold
        along c-axis):
            C11 > |C12|,  C44 > 0,  C66 > 0,  C11 + C12 > 0,  C33 > 0,
            (C11 + C12) * C33 > 2 * C13^2

        旧实现 `C11*C33 > C13^2` 是错误形式 (近似/早期文献), 会漏判部分
        不稳定六方结构. 正确判据含 (C11+C12) 因子, 阈值 2*C13^2.
        """
        C11, C12, C13, C33, C44, C66 = (
            C[0, 0],
            C[0, 1],
            C[0, 2],
            C[2, 2],
            C[3, 3],
            C[5, 5],
        )

        criteria = [
            ("C11 > |C12|", C11, abs(C12), abs(C12) < C11),
            ("C11 + C12 > 0", C11 + C12, 0, (C11 + C12) > 0),
            ("C44 > 0", C44, 0, C44 > 0),
            ("C66 > 0", C66, 0, C66 > 0),
            ("C33 > 0", C33, 0, C33 > 0),
            (
                "(C11+C12)*C33 > 2*C13^2",
                (C11 + C12) * C33,
                2 * C13**2,
                (C11 + C12) * C33 > 2 * C13**2,
            ),
        ]

        stable = all(passed for _, _, _, passed in criteria)

        return {
            "stable": stable,
            "criteria": [
                {
                    "name": name,
                    "value": float(val),
                    "threshold": float(th),
                    "passed": passed,
                }
                for name, val, th, passed in criteria
            ],
            "crystal_system": "hexagonal",
        }

    @classmethod
    def _check_orthorhombic(cls, C: np.ndarray) -> dict:
        """Born criteria for orthorhombic crystals."""
        C11, C22, C33 = C[0, 0], C[1, 1], C[2, 2]
        C12, C13, C23 = C[0, 1], C[0, 2], C[1, 2]
        C44, C55, C66 = C[3, 3], C[4, 4], C[5, 5]

        criteria = [
            ("C11 > 0", C11, 0, C11 > 0),
            ("C22 > 0", C22, 0, C22 > 0),
            ("C33 > 0", C33, 0, C33 > 0),
            ("C44 > 0", C44, 0, C44 > 0),
            ("C55 > 0", C55, 0, C55 > 0),
            ("C66 > 0", C66, 0, C66 > 0),
            (
                "C11 + C22 + C33 + 2(C12+C13+C23) > 0",
                C11 + C22 + C33 + 2 * (C12 + C13 + C23),
                0,
                C11 + C22 + C33 + 2 * (C12 + C13 + C23) > 0,
            ),
            ("C11*C22 > C12^2", C11 * C22, C12**2, C11 * C22 > C12**2),
            ("C11*C33 > C13^2", C11 * C33, C13**2, C11 * C33 > C13**2),
            ("C22*C33 > C23^2", C22 * C33, C23**2, C22 * C33 > C23**2),
            (
                "det(C_3x3) > 0",
                np.linalg.det(C[:3, :3]),
                0,
                np.linalg.det(C[:3, :3]) > 0,
            ),
        ]

        stable = all(passed for _, _, _, passed in criteria)

        return {
            "stable": stable,
            "criteria": [
                {
                    "name": name,
                    "value": float(val),
                    "threshold": float(th),
                    "passed": passed,
                }
                for name, val, th, passed in criteria
            ],
            "crystal_system": "orthorhombic",
        }

    @classmethod
    def _check_triclinic(cls, C: np.ndarray) -> dict:
        """Born criteria for triclinic crystals (most general)."""
        # For triclinic: all principal minors of C must be positive
        # Check leading principal minors
        criteria = []
        for k in range(1, 7):
            minor = np.linalg.det(C[:k, :k])
            criteria.append((f"M{k} > 0", minor, 0, minor > 0))

        stable = all(passed for _, _, _, passed in criteria)

        return {
            "stable": stable,
            "criteria": [
                {
                    "name": name,
                    "value": float(val),
                    "threshold": float(th),
                    "passed": passed,
                }
                for name, val, th, passed in criteria
            ],
            "crystal_system": "triclinic",
        }
