"""Tests for 2D unified framework workflows."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from huginn.unified import discretize, solve
from huginn.unified.models import heat_equation_2d
from huginn.unified.visualize import plot_solution, solve_and_plot


def test_2d_discretize_shape() -> None:
    problem = heat_equation_2d(k=1.0, f=1.0)
    result = discretize(problem, method="fd", n=4)
    assert result["method"] == "fd_2d"
    assert result["n_dof"] == 16
    assert result["shape"] == [4, 4]
    assert len(result["mesh"]) == 16
    assert len(result["mesh"][0]) == 2


def test_2d_solve_residual() -> None:
    problem = heat_equation_2d(k=1.0, f=1.0)
    result = solve(problem, method="fd", n=5)
    assert result["method"] == "fd_2d"
    assert result["n_dof"] == 25
    assert result["residual"] < 1e-10
    u = np.array(result["solution"]).reshape(5, 5)
    # Boundary values are zero due to Dirichlet conditions.
    assert np.allclose(u[0, :], 0.0)
    assert np.allclose(u[-1, :], 0.0)
    assert np.allclose(u[:, 0], 0.0)
    assert np.allclose(u[:, -1], 0.0)
    # Interior values are positive for positive source.
    assert np.all(u[1:-1, 1:-1] > 0)


def test_2d_plot() -> None:
    problem = heat_equation_2d(k=1.0, f=1.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "heat2d.png"
        result = solve_and_plot(problem, method="fd", n=5, output_path=path)
        assert Path(result["plot_path"]).exists()
        assert result["shape"] == [5, 5]


def test_plot_solution_2d_directly() -> None:
    n = 4
    mesh = [[i / (n - 1), j / (n - 1)] for i in range(n) for j in range(n)]
    solution = [i * j for i in range(n) for j in range(n)]
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "direct2d.png"
        result = plot_solution(mesh, solution, path, shape=(n, n))
        assert result.exists()


def test_2d_fem_discretize_shape() -> None:
    problem = heat_equation_2d(k=1.0, f=1.0)
    result = discretize(problem, method="fem", n=3)
    assert result["method"] == "fem_2d"
    assert result["n_dof"] == 16
    assert result["shape"] == [4, 4]


def test_2d_fem_solve_residual() -> None:
    problem = heat_equation_2d(k=1.0, f=1.0)
    result = solve(problem, method="fem", n=4)
    assert result["method"] == "fem_2d"
    assert result["n_dof"] == 25
    assert result["residual"] < 1e-10
    u = np.array(result["solution"]).reshape(5, 5)
    # Dirichlet boundary values are zero
    assert np.allclose(u[0, :], 0.0)
    assert np.allclose(u[-1, :], 0.0)
    assert np.allclose(u[:, 0], 0.0)
    assert np.allclose(u[:, -1], 0.0)
    # Interior values are positive for positive source
    assert np.all(u[1:-1, 1:-1] > 0)


def test_2d_fem_plot() -> None:
    problem = heat_equation_2d(k=1.0, f=1.0)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "fem2d.png"
        result = solve_and_plot(problem, method="fem", n=4, output_path=path)
        assert Path(result["plot_path"]).exists()
