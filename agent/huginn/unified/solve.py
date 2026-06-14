"""Solver layer for the unified scientific computing framework.

Connects the discretization layer to numerical linear algebra:
  1. Discretize a UnifiedProblem into K u = F.
  2. Solve the linear system with NumPy/SciPy.
  3. Return the solution field on the mesh.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from huginn.unified.core import UnifiedProblem
from huginn.unified.discretize import discretize


def solve(
    problem: UnifiedProblem,
    method: str = "fem",
    n: int = 10,
) -> dict[str, Any]:
    """Discretize and solve a variational unified problem.

    Args:
        problem: UnifiedProblem with a variational / minimum principle.
        method: "fem" or "fd".
        n: Number of elements (FEM 1D) or points (FD 1D/2D).

    Returns:
        dict with mesh, solution, residual norm, and discretization metadata.
    """
    disc = discretize(problem, method=method, n=n)
    K = np.array(disc["stiffness_matrix"], dtype=float)
    F = np.array(disc["load_vector"], dtype=float)

    if K.shape[0] == 0:
        raise ValueError("Empty stiffness matrix")

    method_used = disc.get("method", method)

    # For 1D FEM without Dirichlet constraints the matrix is singular.
    # Apply a simple Dirichlet fix: u = 0 at both ends.
    if method_used == "fem":
        for idx in (0, -1):
            K[idx, :] = 0.0
            K[:, idx] = 0.0
            K[idx, idx] = 1.0
            F[idx] = 0.0

    try:
        u = np.linalg.solve(K, F)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Linear solve failed: {e}")

    residual = np.linalg.norm(K @ u - F)
    result = {
        "method": method_used,
        "n": n,
        "mesh": disc["mesh"],
        "solution": u.tolist(),
        "residual": residual,
        "n_dof": disc["n_dof"],
        "stiffness_matrix": disc["stiffness_matrix"],
        "load_vector": disc["load_vector"],
    }
    if "shape" in disc:
        result["shape"] = disc["shape"]
    return result
