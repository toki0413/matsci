"""Discretization layer for the unified scientific computing framework.

Turns a continuous UnifiedProblem into a discrete algebraic system:
  - FEM: element stiffness matrix and load vector
  - FD: finite-difference stencil matrix and right-hand side

This is the first step toward connecting the symbolic derivation layer to
actual solvers (linear algebra, HPC, etc.).
"""

from __future__ import annotations

from typing import Any

import sympy as sp

from huginn.unified.core import UnifiedProblem, VariationalPrinciple


def _bounds(problem: UnifiedProblem) -> tuple[float, float]:
    if problem.domain and problem.domain.bounds:
        return next(iter(problem.domain.bounds.values()))
    return (0.0, 1.0)


def _material_coefficient(problem: UnifiedProblem) -> float:
    """Extract the leading coefficient (thermal conductivity / Young modulus)."""
    if problem.energy is None:
        return 1.0
    params = problem.energy.parameters or {}
    return float(params.get("k", params.get("E", 1.0)))


def _source_term(problem: UnifiedProblem) -> float:
    if problem.energy is None:
        return 0.0
    params = problem.energy.parameters or {}
    return float(params.get("f", 0.0))


def _discretize_fem_1d(problem: UnifiedProblem, n: int) -> dict[str, Any]:
    """1D linear FEM discretization for energy-minimization problems."""
    a, b = _bounds(problem)
    h = (b - a) / n
    coeff = _material_coefficient(problem)
    f = _source_term(problem)

    n_dof = n + 1
    K = [[0.0] * n_dof for _ in range(n_dof)]
    F = [0.0] * n_dof

    k_local = (coeff / h) * 1.0
    f_local = (f * h) / 2.0

    for e in range(n):
        i, j = e, e + 1
        # Local stiffness: coeff/h * [[1, -1], [-1, 1]]
        K[i][i] += k_local
        K[i][j] -= k_local
        K[j][i] -= k_local
        K[j][j] += k_local
        # Local load: f*h/2 * [1, 1]
        F[i] += f_local
        F[j] += f_local

    mesh = [a + i * h for i in range(n_dof)]
    return {
        "method": "fem",
        "n_elements": n,
        "n_dof": n_dof,
        "stiffness_matrix": K,
        "load_vector": F,
        "mesh": mesh,
        "dof_map": list(range(n_dof)),
    }


def _discretize_fd_1d(problem: UnifiedProblem, n: int) -> dict[str, Any]:
    """1D finite-difference discretization for -coeff u'' = f."""
    a, b = _bounds(problem)
    if n < 3:
        raise ValueError("FD discretization needs at least 3 points")
    h = (b - a) / (n - 1)
    coeff = _material_coefficient(problem)
    f = _source_term(problem)

    A = [[0.0] * n for _ in range(n)]
    rhs = [0.0] * n

    # Dirichlet boundaries: u = 0
    A[0][0] = 1.0
    A[-1][-1] = 1.0

    c = coeff / (h * h)
    for i in range(1, n - 1):
        A[i][i - 1] = -c
        A[i][i] = 2.0 * c
        A[i][i + 1] = -c
        rhs[i] = f

    mesh = [a + i * h for i in range(n)]
    return {
        "method": "fd",
        "n_points": n,
        "n_dof": n,
        "stiffness_matrix": A,
        "load_vector": rhs,
        "mesh": mesh,
        "dof_map": list(range(n)),
    }


def discretize(
    problem: UnifiedProblem,
    method: str = "fem",
    n: int = 10,
) -> dict[str, Any]:
    """Discretize a unified problem into a linear algebraic system.

    Args:
        problem: UnifiedProblem with a variational / minimum principle.
        method: "fem" or "fd".
        n: Number of elements (FEM) or points (FD).

    Returns:
        dict with stiffness_matrix, load_vector, mesh, etc.
    """
    method = method.lower()
    if problem.principle not in {
        VariationalPrinciple.STATIONARY,
        VariationalPrinciple.MINIMUM,
        VariationalPrinciple.MAXIMUM,
    }:
        raise ValueError(
            f"Discretization currently supports variational principles, got {problem.principle}"
        )

    if problem.domain is None or len(problem.domain.coordinates) != 1:
        raise ValueError("Only 1D problems are supported for now")

    if method == "fem":
        return _discretize_fem_1d(problem, n)
    if method == "fd":
        return _discretize_fd_1d(problem, n)

    raise ValueError(f"Unknown discretization method: {method}")
