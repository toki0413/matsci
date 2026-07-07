"""Tests for FenicsTool, ElmerTool, GromacsTool — solver tools with mock sandbox."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from huginn.tools.sim.fenics_tool import FenicsTool, FenicsToolInput
from huginn.tools.sim.elmer_tool import ElmerTool, ElmerToolInput
from huginn.tools.sim.gromacs_tool import GromacsTool, GromacsToolInput
from huginn.types import ToolContext


# ── fixtures ──


@pytest.fixture
def ctx(tmp_path):
    return ToolContext(session_id="test", workspace=str(tmp_path))


def _mock_subprocess(returncode=0, stdout="", stderr=""):
    """Create a mock sandbox result (fields match SandboxResult closely enough)."""
    return type("R", (), {
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    })()


# ════════════════════════════════════════════════════════
#  FenicsTool
# ════════════════════════════════════════════════════════


class TestFenicsTool:
    @pytest.fixture
    def tool(self):
        return FenicsTool()

    def test_solve_pde_no_script(self, tool, ctx):
        """solve_pde without script → error."""
        result = tool.call({"action": "solve_pde", "working_dir": ctx.workspace}, ctx)
        assert result.success is False
        assert "script" in result.error.lower()

    def test_solve_pde_fenics_missing(self, tool, ctx, tmp_path):
        """FEniCS not installed → script saved, status=script_generated."""
        with patch("huginn.tools.sim.fenics_tool._fenics_available", return_value=False):
            result = tool.call(
                {
                    "action": "solve_pde",
                    "script": "from dolfin import *; print('hello')",
                    "working_dir": ctx.workspace,
                },
                ctx,
            )
        assert result.success is True
        assert result.data["status"] == "script_generated"
        assert "script_path" in result.data

    def test_solve_pde_fenics_available(self, tool, ctx):
        """FEniCS installed → sandbox.run called."""
        with patch("huginn.tools.sim.fenics_tool._fenics_available", return_value=True), \
             patch.object(tool.sandbox, "run") as mock_run:
            mock_run.return_value = _mock_subprocess(0, "Solved!", "")
            result = tool.call(
                {
                    "action": "solve_pde",
                    "script": "from dolfin import *",
                    "working_dir": ctx.workspace,
                },
                ctx,
            )
        assert result.success is True
        assert result.data["returncode"] == 0
        mock_run.assert_called_once()

    def test_mesh_info_missing_file(self, tool, ctx):
        """mesh_info without mesh_file → error."""
        result = tool.call({"action": "mesh_info", "working_dir": ctx.workspace}, ctx)
        assert result.success is False
        assert "mesh_file" in result.error

    def test_mesh_info_file_not_found(self, tool, ctx):
        """mesh_info with nonexistent file → error."""
        result = tool.call(
            {"action": "mesh_info", "mesh_file": "nope.xml", "working_dir": ctx.workspace},
            ctx,
        )
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_convergence_check_needs_two(self, tool, ctx):
        """convergence_check with <2 files → error."""
        result = tool.call(
            {"action": "convergence_check", "solution_files": ["a.pvd"], "working_dir": ctx.workspace},
            ctx,
        )
        assert result.success is False
        assert "2" in result.error


# ════════════════════════════════════════════════════════
#  ElmerTool
# ════════════════════════════════════════════════════════


class TestElmerTool:
    @pytest.fixture
    def tool(self):
        return ElmerTool()

    def test_solve_sif_no_content(self, tool, ctx):
        """solve_sif without sif_content/sif_file → error."""
        result = tool.call({"action": "solve_sif", "working_dir": ctx.workspace}, ctx)
        assert result.success is False
        assert "sif" in result.error.lower()

    def test_solve_sif_elmer_missing(self, tool, ctx):
        """ElmerSolver not installed → sif exported, status=sif_exported."""
        with patch("huginn.tools.sim.elmer_tool._elmer_available", return_value=False):
            result = tool.call(
                {
                    "action": "solve_sif",
                    "sif_content": "Header\n  Mesh DB = \".\"\nEnd\n",
                    "working_dir": ctx.workspace,
                },
                ctx,
            )
        assert result.success is True
        assert result.data["status"] == "sif_exported"
        assert "sif_path" in result.data

    def test_solve_sif_elmer_available(self, tool, ctx):
        """ElmerSolver installed → sandbox.run called."""
        with patch("huginn.tools.sim.elmer_tool._elmer_available", return_value=True), \
             patch.object(tool.sandbox, "run") as mock_run:
            mock_run.return_value = _mock_subprocess(0, "ELMER SOLVER DONE", "")
            result = tool.call(
                {
                    "action": "solve_sif",
                    "sif_content": "Header\nEnd\n",
                    "working_dir": ctx.workspace,
                },
                ctx,
            )
        assert result.success is True
        mock_run.assert_called_once()

    def test_validate_sif_valid(self, tool, ctx):
        """validate_sif with proper sections → valid=True."""
        sif = """
        Header
          Mesh DB = "."
        End
        Simulation
          Max Output Level = 3
          Steady State Max = 1
          Coordinate System = Cartesian
        End
        Equation 1
          Active Solvers(1) = 1
        End
        """
        result = tool.call(
            {"action": "validate_sif", "sif_content": sif, "working_dir": ctx.workspace},
            ctx,
        )
        assert result.success is True
        assert result.data["valid"] is True
        assert len(result.data["issues"]) == 0

    def test_validate_sif_missing_sections(self, tool, ctx):
        """validate_sif with missing sections → issues reported."""
        result = tool.call(
            {"action": "validate_sif", "sif_content": "just some text", "working_dir": ctx.workspace},
            ctx,
        )
        assert result.success is True
        assert result.data["valid"] is False
        assert len(result.data["issues"]) > 0

    def test_mesh_to_elmer_missing_dir(self, tool, ctx):
        """mesh_to_elmer without mesh_dir → error."""
        result = tool.call({"action": "mesh_to_elmer", "working_dir": ctx.workspace}, ctx)
        assert result.success is False
        assert "mesh_dir" in result.error


# ════════════════════════════════════════════════════════
#  GromacsTool
# ════════════════════════════════════════════════════════


class TestGromacsTool:
    @pytest.fixture
    def tool(self):
        return GromacsTool()

    def test_md_run_no_tpr(self, tool, ctx):
        """md_run without tpr_file → error."""
        result = tool.call({"action": "md_run", "working_dir": ctx.workspace}, ctx)
        assert result.success is False
        assert "tpr" in result.error.lower()

    def test_md_run_gmx_missing(self, tool, ctx, tmp_path):
        """gmx not installed → status=skipped."""
        tpr = tmp_path / "run.tpr"
        tpr.write_text("mock")
        with patch("huginn.tools.sim.gromacs_tool._gmx_available", return_value=False):
            result = tool.call(
                {"action": "md_run", "tpr_file": str(tpr), "working_dir": ctx.workspace},
                ctx,
            )
        assert result.success is True
        assert result.data["status"] == "skipped"

    def test_md_run_gmx_available(self, tool, ctx, tmp_path):
        """gmx installed → sandbox.run called."""
        tpr = tmp_path / "run.tpr"
        tpr.write_text("mock")
        with patch("huginn.tools.sim.gromacs_tool._gmx_available", return_value=True), \
             patch.object(tool.sandbox, "run") as mock_run:
            mock_run.return_value = _mock_subprocess(0, "Finished!", "")
            result = tool.call(
                {"action": "md_run", "tpr_file": str(tpr), "nsteps": 100, "working_dir": ctx.workspace},
                ctx,
            )
        assert result.success is True
        assert result.data["nsteps"] == 100
        mock_run.assert_called_once()

    def test_energy_minimize_gmx_missing(self, tool, ctx, tmp_path):
        """energy_minimize with gmx missing → skipped."""
        tpr = tmp_path / "em.tpr"
        tpr.write_text("mock")
        with patch("huginn.tools.sim.gromacs_tool._gmx_available", return_value=False):
            result = tool.call(
                {"action": "energy_minimize", "tpr_file": str(tpr), "working_dir": ctx.workspace},
                ctx,
            )
        assert result.success is True
        assert result.data["status"] == "skipped"

    def test_analyze_traj_no_file(self, tool, ctx):
        """analyze_traj without trajectory → error."""
        result = tool.call(
            {"action": "analyze_traj", "working_dir": ctx.workspace},
            ctx,
        )
        assert result.success is False
        assert "trajectory" in result.error.lower()

    def test_analyze_traj_gmx_missing(self, tool, ctx, tmp_path):
        """analyze_traj with gmx missing → skipped."""
        traj = tmp_path / "traj.xtc"
        traj.write_text("mock")
        with patch("huginn.tools.sim.gromacs_tool._gmx_available", return_value=False):
            result = tool.call(
                {
                    "action": "analyze_traj",
                    "trajectory_file": str(traj),
                    "analysis_type": "rms",
                    "working_dir": ctx.workspace,
                },
                ctx,
            )
        assert result.success is True
        assert result.data["status"] == "skipped"
