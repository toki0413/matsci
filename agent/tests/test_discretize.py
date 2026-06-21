"""Tests for the unified framework discretization layer."""

from __future__ import annotations

import asyncio

import pytest

from huginn.tools.symbolic_math_tool import SymbolicMathInput, SymbolicMathTool
from huginn.types import ToolContext
from huginn.unified import discretize
from huginn.unified.models import heat_equation_fem

CTX = ToolContext(session_id="test", workspace=".")


def test_fem_heat_discretization() -> None:
    problem = heat_equation_fem(k=2.0, f=1.0)
    result = discretize(problem, method="fem", n=4)
    assert result["method"] == "fem"
    assert result["n_dof"] == 5
    K = result["stiffness_matrix"]
    F = result["load_vector"]
    # 1D linear Laplacian stiffness is tridiagonal with 2/h and -/h
    assert K[0][0] > 0
    assert K[0][1] < 0
    assert sum(F) == pytest.approx(1.0, abs=1e-9)


def test_fd_heat_discretization() -> None:
    problem = heat_equation_fem(k=1.0, f=0.0)
    result = discretize(problem, method="fd", n=5)
    assert result["method"] == "fd"
    assert result["n_dof"] == 5
    A = result["stiffness_matrix"]
    # Boundary rows are identity
    assert A[0][0] == 1.0
    assert A[-1][-1] == 1.0
    # Interior row: [-c, 2c, -c]
    assert A[2][1] == A[2][3]
    assert A[2][1] < 0
    assert A[2][2] == -2.0 * A[2][1]


def test_discretize_unsupported_principle() -> None:
    from huginn.unified.models import harmonic_oscillator_md

    problem = harmonic_oscillator_md()
    with pytest.raises(ValueError, match="variational principles"):
        discretize(problem, method="fem", n=4)


class TestUnifiedSymbolicDiscretize:
    @pytest.fixture
    def tool(self):
        return SymbolicMathTool()

    def test_unified_discretize_fem(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="discretize",
                    expression="linear_elasticity_fem",
                    variable="fem",
                    order=4,
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["method"] == "fem"
        assert result.data["n_dof"] == 5
        assert len(result.data["stiffness_matrix"]) == 5

    def test_unified_discretize_fd(self, tool):
        result = asyncio.run(
            tool.call(
                SymbolicMathInput(
                    action="unified",
                    target="discretize",
                    expression="heat_equation_fem",
                    variable="fd",
                    order=5,
                ),
                CTX,
            )
        )
        assert result.success
        assert result.data["method"] == "fd"
        assert result.data["n_dof"] == 5


# ------------------------------------------------------------------
# DiscretizationMetadata tests
# ------------------------------------------------------------------

class TestDiscretizationMetadata:
    """Verify that discretize() attaches correct metadata annotations."""

    def test_fem_1d_metadata_present(self):
        problem = heat_equation_fem(k=2.0, f=1.0)
        result = discretize(problem, method="fem", n=4)
        assert "metadata" in result
        meta = result["metadata"]
        from huginn.unified.discretize import DiscretizationMetadata
        assert isinstance(meta, DiscretizationMetadata)

    def test_fem_1d_spatial_dimension(self):
        problem = heat_equation_fem(k=2.0, f=1.0)
        result = discretize(problem, method="fem", n=4)
        meta = result["metadata"]
        assert meta.spatial_dimension == 1

    def test_fem_1d_dof_type(self):
        problem = heat_equation_fem(k=2.0, f=1.0)
        result = discretize(problem, method="fem", n=4)
        meta = result["metadata"]
        assert meta.dof_type == "temperature"
        assert meta.dof_kind == "scalar"
        assert meta.dof_units == "K"  # heat equation uses temperature

    def test_fem_1d_no_bc(self):
        """1D FEM does not apply BCs at discretization time."""
        problem = heat_equation_fem(k=2.0, f=1.0)
        result = discretize(problem, method="fem", n=4)
        meta = result["metadata"]
        assert meta.bc_type == "none"
        assert meta.bc_indices == []
        assert meta.interior_indices == list(range(5))

    def test_fem_1d_element_type(self):
        problem = heat_equation_fem(k=2.0, f=1.0)
        result = discretize(problem, method="fem", n=4)
        meta = result["metadata"]
        assert meta.element_type == "linear_segment"
        assert meta.matrix_structure == "symmetric_positive_definite"

    def test_fem_1d_material_coefficient(self):
        problem = heat_equation_fem(k=2.0, f=1.0)
        result = discretize(problem, method="fem", n=4)
        meta = result["metadata"]
        assert meta.material_coefficient == 2.0
        assert meta.source_term == 1.0

    def test_fd_1d_dirichlet_bc(self):
        """1D FD applies Dirichlet BCs at first and last DOF."""
        problem = heat_equation_fem(k=1.0, f=0.0)
        result = discretize(problem, method="fd", n=5)
        meta = result["metadata"]
        assert meta.bc_type == "dirichlet"
        assert meta.bc_indices == [0, 4]
        assert meta.bc_values == [0.0, 0.0]
        assert meta.interior_indices == [1, 2, 3]
        assert meta.element_type == "3pt_stencil"
        assert meta.matrix_structure == "tridiagonal"

    def test_fd_1d_spatial_dimension(self):
        problem = heat_equation_fem(k=1.0, f=0.0)
        result = discretize(problem, method="fd", n=5)
        meta = result["metadata"]
        assert meta.spatial_dimension == 1

    def test_fem_2d_metadata(self):
        from huginn.unified.models import heat_equation_2d
        problem = heat_equation_2d()
        result = discretize(problem, method="fem", n=3)
        meta = result["metadata"]
        assert meta.spatial_dimension == 2
        assert meta.element_type == "linear_triangle"
        assert meta.bc_type == "dirichlet"
        assert len(meta.bc_indices) > 0
        assert len(meta.interior_indices) > 0
        # boundary + interior = all DOFs
        all_bc = set(meta.bc_indices)
        all_interior = set(meta.interior_indices)
        assert all_bc | all_interior == set(range(result["n_dof"]))
        assert all_bc & all_interior == set()

    def test_fd_2d_metadata(self):
        from huginn.unified.models import heat_equation_2d
        problem = heat_equation_2d()
        result = discretize(problem, method="fd", n=4)
        meta = result["metadata"]
        assert meta.spatial_dimension == 2
        assert meta.element_type == "5pt_stencil"
        assert meta.bc_type == "dirichlet"
        assert len(meta.bc_indices) > 0
        # perimeter DOFs for 4x4 grid: 4*4 - 2*2 = 12 boundary
        assert len(meta.bc_indices) == 12
        assert len(meta.interior_indices) == 4

    def test_metadata_to_dict(self):
        problem = heat_equation_fem(k=2.0, f=1.0)
        result = discretize(problem, method="fem", n=4)
        meta = result["metadata"]
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["spatial_dimension"] == 1
        assert d["dof_type"] == "temperature"
        assert "bc_indices" in d
        assert "interior_indices" in d

    def test_metadata_domain_bounds(self):
        problem = heat_equation_fem(k=2.0, f=1.0)
        result = discretize(problem, method="fem", n=4)
        meta = result["metadata"]
        assert isinstance(meta.domain_bounds, dict)
        assert len(meta.domain_bounds) >= 1

    def test_existing_keys_unchanged(self):
        """Verify that adding metadata does not alter existing result keys."""
        problem = heat_equation_fem(k=2.0, f=1.0)
        result = discretize(problem, method="fem", n=4)
        assert "method" in result
        assert "stiffness_matrix" in result
        assert "load_vector" in result
        assert "mesh" in result
        assert "n_dof" in result
        assert "metadata" in result
        # The stiffness matrix should still work as before
        K = result["stiffness_matrix"]
        assert K[0][0] > 0
        assert K[0][1] < 0
