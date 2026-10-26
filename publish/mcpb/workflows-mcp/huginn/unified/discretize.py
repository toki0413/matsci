"""Discretization layer for the unified scientific computing framework.

Turns a continuous UnifiedProblem into a discrete algebraic system:
  - FEM: element stiffness matrix and load vector
  - FD: finite-difference stencil matrix and right-hand side

This is the first step toward connecting the symbolic derivation layer to
actual solvers (linear algebra, HPC, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from huginn.unified.core import UnifiedProblem, VariationalPrinciple

# ====================================================================
# Discretization metadata — lightweight annotation
# ====================================================================


@dataclass
class DiscretizationMetadata:
    """Metadata attached to discretization output matrices.

    Provides semantic context for downstream consumers (solvers, HPC,
    visualization) without modifying the raw matrix data.
    """

    spatial_dimension: int
    dof_type: str  # e.g. "temperature", "displacement", "u"
    dof_kind: str  # "scalar", "vector", "tensor"
    dof_units: str  # e.g. "K", "m", ""
    domain_bounds: dict[str, tuple[float, float]]
    material_coefficient: float
    source_term: float
    bc_type: str  # "dirichlet", "none", "mixed"
    bc_indices: list[int]  # DOF indices where BCs are applied
    bc_values: list[float]  # prescribed values at bc_indices
    interior_indices: list[int]  # DOF indices NOT constrained
    element_type: str  # "linear_segment", "linear_triangle", "5pt_stencil"
    matrix_structure: str  # "symmetric_positive_definite", "tridiagonal", "dense"

    def to_dict(self) -> dict[str, Any]:
        return {
            "spatial_dimension": self.spatial_dimension,
            "dof_type": self.dof_type,
            "dof_kind": self.dof_kind,
            "dof_units": self.dof_units,
            "domain_bounds": self.domain_bounds,
            "material_coefficient": self.material_coefficient,
            "source_term": self.source_term,
            "bc_type": self.bc_type,
            "bc_indices": self.bc_indices,
            "bc_values": self.bc_values,
            "interior_indices": self.interior_indices,
            "element_type": self.element_type,
            "matrix_structure": self.matrix_structure,
        }


def _dof_descriptor(problem: UnifiedProblem) -> tuple[str, str, str]:
    # Extract DOF type from the first field in the problem
    dof_type, dof_kind, dof_units = "u", "scalar", ""
    if problem.fields:
        first_field = next(iter(problem.fields.values()))
        dof_type = first_field.name
        dof_kind = (
            str(first_field.kind.value)
            if hasattr(first_field.kind, "value")
            else str(first_field.kind)
        )
        dof_units = first_field.units or ""
    return dof_type, dof_kind, dof_units


def _detect_dirichlet_dofs(
    result: dict[str, Any], n_dof: int
) -> tuple[list[int], list[float]]:
    """Detect Dirichlet DOFs from identity rows with zero right-hand side."""
    bc_indices: list[int] = []
    bc_values: list[float] = []

    # Detect boundary DOFs from the stiffness matrix structure:
    # A row that is [0, ..., 0, 1, 0, ..., 0] with rhs=0 is a Dirichlet BC row
    K = result.get("stiffness_matrix", [])
    F = result.get("load_vector", [])
    for i in range(min(n_dof, len(K))):
        row = K[i]
        if len(row) == n_dof and abs(row[i] - 1.0) < 1e-12:
            off_diag_sum = sum(abs(row[j]) for j in range(n_dof) if j != i)
            if off_diag_sum < 1e-12 and i < len(F) and abs(F[i]) < 1e-12:
                bc_indices.append(i)
                bc_values.append(0.0)
    return bc_indices, bc_values


def _element_profile(dim: int, method: str) -> tuple[str, str]:
    # Element type label
    if dim == 1 and method == "fem":
        return "linear_segment", "symmetric_positive_definite"
    if dim == 1 and method == "fd":
        return "3pt_stencil", "tridiagonal"
    if dim == 2 and method == "fem":
        return "linear_triangle", "symmetric_positive_definite"
    if dim == 2 and method == "fd":
        return "5pt_stencil", "symmetric_positive_definite"
    return "unknown", "dense"


def _attach_metadata(
    result: dict[str, Any],
    problem: UnifiedProblem,
    method: str,
) -> dict[str, Any]:
    """Attach DiscretizationMetadata to the result dict (non-invasive)."""
    dim = len(problem.domain.coordinates) if problem.domain else 1
    bounds = dict(problem.domain.bounds) if problem.domain else {}

    dof_type, dof_kind, dof_units = _dof_descriptor(problem)

    coeff = _material_coefficient(problem)
    source = _source_term(problem)

    n_dof: int = result.get("n_dof", 0)
    bc_indices, bc_values = _detect_dirichlet_dofs(result, n_dof)

    all_indices = set(range(n_dof))
    bc_set = set(bc_indices)
    interior = sorted(all_indices - bc_set)

    bc_type = "dirichlet" if bc_indices else "none"

    element_type, matrix_structure = _element_profile(dim, method)

    metadata = DiscretizationMetadata(
        spatial_dimension=dim,
        dof_type=dof_type,
        dof_kind=dof_kind,
        dof_units=dof_units,
        domain_bounds=bounds,
        material_coefficient=coeff,
        source_term=source,
        bc_type=bc_type,
        bc_indices=bc_indices,
        bc_values=bc_values,
        interior_indices=interior,
        element_type=element_type,
        matrix_structure=matrix_structure,
    )
    result["metadata"] = metadata
    return result


def _bounds_1d(problem: UnifiedProblem) -> tuple[float, float]:
    if problem.domain and problem.domain.bounds:
        return next(iter(problem.domain.bounds.values()))
    return (0.0, 1.0)


def _bounds_2d(
    problem: UnifiedProblem,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if problem.domain and problem.domain.bounds:
        bounds = list(problem.domain.bounds.values())
        if len(bounds) >= 2:
            return bounds[0], bounds[1]
    return (0.0, 1.0), (0.0, 1.0)


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
    a, b = _bounds_1d(problem)
    h = (b - a) / n
    coeff = _material_coefficient(problem)
    f = _source_term(problem)

    n_dof = n + 1
    K = [[0.0] * n_dof for _ in range(n_dof)]
    F = [0.0] * n_dof

    k_local = coeff / h
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
    a, b = _bounds_1d(problem)
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


def _node_index_2d(i: int, j: int, ny: int) -> int:
    return i * ny + j


def _node_coord_2d(
    i: int, j: int, x0: float, y0: float, hx: float, hy: float
) -> tuple[float, float]:
    return (x0 + i * hx, y0 + j * hy)


def _triangle_area_2d(tri: list[tuple[float, float]]) -> float:
    (x_a, y_a), (x_b, y_b), (x_c, y_c) = tri
    return 0.5 * abs((x_b - x_a) * (y_c - y_a) - (x_c - x_a) * (y_b - y_a))


def _triangle_stiffness_2d(tri: list[tuple[float, float]], coeff: float) -> np.ndarray:
    (x_a, y_a), (x_b, y_b), (x_c, y_c) = tri
    area = _triangle_area_2d(tri)
    if area == 0.0:
        return np.zeros((3, 3))
    # Shape function gradients (constant over element)
    dndx = np.array([y_b - y_c, y_c - y_a, y_a - y_b]) / (2.0 * area)
    dndy = np.array([x_c - x_b, x_a - x_c, x_b - x_a]) / (2.0 * area)
    bmat = np.vstack([dndx, dndy])
    return coeff * area * (bmat.T @ bmat)


def _triangle_load_2d(area: float, f: float) -> np.ndarray:
    # Consistent load: f * area / 3 for each node of a linear triangle
    return (f * area / 3.0) * np.ones(3)


def _cell_triangles_2d(
    i: int, j: int, ny: int, x0: float, y0: float, hx: float, hy: float
) -> list[tuple[list[tuple[float, float]], list[int]]]:
    p00 = _node_coord_2d(i, j, x0, y0, hx, hy)
    p10 = _node_coord_2d(i + 1, j, x0, y0, hx, hy)
    p01 = _node_coord_2d(i, j + 1, x0, y0, hx, hy)
    p11 = _node_coord_2d(i + 1, j + 1, x0, y0, hx, hy)

    return [
        (
            [p00, p10, p01],
            [
                _node_index_2d(i, j, ny),
                _node_index_2d(i + 1, j, ny),
                _node_index_2d(i, j + 1, ny),
            ],
        ),
        (
            [p10, p11, p01],
            [
                _node_index_2d(i + 1, j, ny),
                _node_index_2d(i + 1, j + 1, ny),
                _node_index_2d(i, j + 1, ny),
            ],
        ),
    ]


def _assemble_fem_2d(
    n: int,
    nx: int,
    ny: int,
    x0: float,
    y0: float,
    hx: float,
    hy: float,
    coeff: float,
    f: float,
) -> tuple[np.ndarray, np.ndarray]:
    N = nx * ny
    K = np.zeros((N, N), dtype=float)
    F = np.zeros(N, dtype=float)

    for i in range(n):
        for j in range(n):
            for tri, nodes in _cell_triangles_2d(i, j, ny, x0, y0, hx, hy):
                ke = _triangle_stiffness_2d(tri, coeff)
                area = _triangle_area_2d(tri)
                fe = _triangle_load_2d(area, f)
                for a in range(3):
                    F[nodes[a]] += fe[a]
                    for b in range(3):
                        K[nodes[a], nodes[b]] += ke[a, b]

    return K, F


def _apply_dirichlet_fem_2d(
    K: np.ndarray, F: np.ndarray, nx: int, ny: int, n: int
) -> None:
    # Dirichlet boundary: u = 0 on rectangle edges
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == n or j == 0 or j == n:
                k = _node_index_2d(i, j, ny)
                K[k, :] = 0.0
                K[:, k] = 0.0
                K[k, k] = 1.0
                F[k] = 0.0


def _discretize_fem_2d(problem: UnifiedProblem, n: int) -> dict[str, Any]:
    """2D linear triangular FEM discretization for -coeff Δu = f.

    Each axis-aligned cell is split into two triangles. Linear shape
    functions give a constant gradient per element, so the element
    stiffness is K_e = area * B^T B.
    """
    (x0, x1), (y0, y1) = _bounds_2d(problem)
    if n < 1:
        raise ValueError("FEM discretization needs at least 1 element per dimension")
    hx = (x1 - x0) / n
    hy = (y1 - y0) / n
    coeff = _material_coefficient(problem)
    f = _source_term(problem)

    nx = n + 1
    ny = n + 1
    N = nx * ny

    K, F = _assemble_fem_2d(n, nx, ny, x0, y0, hx, hy, coeff, f)

    _apply_dirichlet_fem_2d(K, F, nx, ny, n)

    mesh = [
        list(_node_coord_2d(i, j, x0, y0, hx, hy)) for i in range(nx) for j in range(ny)
    ]
    return {
        "method": "fem_2d",
        "n_elements_per_dim": n,
        "n_dof": N,
        "stiffness_matrix": K.tolist(),
        "load_vector": F.tolist(),
        "mesh": mesh,
        "shape": [nx, ny],
    }


def _discretize_fd_2d(problem: UnifiedProblem, n: int) -> dict[str, Any]:
    """2D finite-difference discretization for -coeff Δu = f on a rectangle.

    Uses the classic 5-point stencil with n points per dimension.
    """
    (x0, x1), (y0, y1) = _bounds_2d(problem)
    if n < 3:
        raise ValueError("2D FD discretization needs at least 3 points per dimension")
    hx = (x1 - x0) / (n - 1)
    hy = (y1 - y0) / (n - 1)
    coeff = _material_coefficient(problem)
    f = _source_term(problem)

    N = n * n
    A = np.zeros((N, N), dtype=float)
    rhs = np.full(N, f, dtype=float)

    cx = coeff / (hx * hx)
    cy = coeff / (hy * hy)

    def idx(i: int, j: int) -> int:
        return i * n + j

    for i in range(n):
        for j in range(n):
            k = idx(i, j)
            # Boundary nodes: Dirichlet u = 0
            if i == 0 or i == n - 1 or j == 0 or j == n - 1:
                A[k, k] = 1.0
                rhs[k] = 0.0
                continue
            A[k, k] = 2.0 * (cx + cy)
            A[k, idx(i - 1, j)] = -cx
            A[k, idx(i + 1, j)] = -cx
            A[k, idx(i, j - 1)] = -cy
            A[k, idx(i, j + 1)] = -cy

    mesh = [[x0 + i * hx, y0 + j * hy] for i in range(n) for j in range(n)]
    return {
        "method": "fd_2d",
        "n_points_per_dim": n,
        "n_dof": N,
        "stiffness_matrix": A.tolist(),
        "load_vector": rhs.tolist(),
        "mesh": mesh,
        "shape": [n, n],
    }


def _dispatch_discretize(
    problem: UnifiedProblem, method: str, dim: int, n: int
) -> dict[str, Any]:
    if dim == 1:
        if method == "fem":
            return _discretize_fem_1d(problem, n)
        if method == "fd":
            return _discretize_fd_1d(problem, n)
        raise ValueError(f"Unknown discretization method for 1D: {method}")
    if dim == 2:
        if method == "fem":
            return _discretize_fem_2d(problem, n)
        if method == "fd":
            return _discretize_fd_2d(problem, n)
        raise ValueError(f"Unknown discretization method for 2D: {method}")
    raise ValueError(f"Unsupported domain dimension: {dim}")


def discretize(
    problem: UnifiedProblem,
    method: str = "fem",
    n: int = 10,
) -> dict[str, Any]:
    """Discretize a unified problem into a linear algebraic system.

    Args:
        problem: UnifiedProblem with a variational / minimum principle.
        method: "fem" or "fd".
        n: Number of elements (FEM 1D) or points (FD 1D/2D).

    Returns:
        dict with stiffness_matrix, load_vector, mesh, and ``metadata``
        (:class:`DiscretizationMetadata`) containing DOF type, spatial
        dimension, boundary condition labels, and matrix structure info.
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

    if problem.domain is None:
        raise ValueError("Problem must have a domain")

    dim = len(problem.domain.coordinates)
    result: dict[str, Any] = _dispatch_discretize(problem, method, dim, n)

    return _attach_metadata(result, problem, method)
