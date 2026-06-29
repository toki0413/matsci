"""Tests for Handle/Value pre-flight validation (Phase 2 Feature 1).

Covers:
- HandleValidator built-in checkers (file_path, job_id, material_id, formula)
- validate_input() overrides on vasp, lammps, structure, job, potential tools
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from huginn.types import HandleType, ToolContext, ValidationResult
from huginn.validation.handle_validator import HandleValidator


# ── Helpers ──────────────────────────────────────────────────────────


def _ctx(workspace: str = "") -> ToolContext:
    return ToolContext(session_id="test", workspace=workspace)


# ── HandleValidator built-in checkers ───────────────────────────────


class TestFilePathChecker:
    def test_existing_absolute_path(self, tmp_path):
        f = tmp_path / "data.xyz"
        f.write_text("data")
        vr = HandleValidator.validate(HandleType.FILE_PATH, str(f), _ctx())
        assert vr.result is True

    def test_missing_path(self):
        vr = HandleValidator.validate(
            HandleType.FILE_PATH, "/nonexistent/nope.xyz", _ctx()
        )
        assert vr.result is False
        assert vr.error_code == 404

    def test_relative_path_resolved_via_workspace(self, tmp_path):
        f = tmp_path / "rel.xyz"
        f.write_text("data")
        vr = HandleValidator.validate(
            HandleType.FILE_PATH, "rel.xyz", _ctx(workspace=str(tmp_path))
        )
        assert vr.result is True

    def test_relative_path_no_workspace(self):
        vr = HandleValidator.validate(
            HandleType.FILE_PATH, "does_not_exist.xyz", _ctx(workspace="")
        )
        assert vr.result is False


class TestJobIdChecker:
    def test_valid_job_id(self):
        vr = HandleValidator.validate(HandleType.JOB_ID, "job_12345", _ctx())
        assert vr.result is True

    def test_empty_job_id(self):
        vr = HandleValidator.validate(HandleType.JOB_ID, "", _ctx())
        assert vr.result is False
        assert vr.error_code == 400

    def test_whitespace_job_id(self):
        vr = HandleValidator.validate(HandleType.JOB_ID, "   ", _ctx())
        assert vr.result is False


class TestMaterialIdChecker:
    def test_valid_material_id(self):
        vr = HandleValidator.validate(HandleType.MATERIAL_ID, "mp-1234", _ctx())
        assert vr.result is True

    def test_empty_material_id(self):
        vr = HandleValidator.validate(HandleType.MATERIAL_ID, "", _ctx())
        assert vr.result is False


class TestFormulaChecker:
    def test_valid_formula(self):
        vr = HandleValidator.validate(HandleType.FORMULA, "Fe2O3", _ctx())
        assert vr.result is True

    def test_empty_formula(self):
        vr = HandleValidator.validate(HandleType.FORMULA, "", _ctx())
        assert vr.result is False

    def test_no_uppercase(self):
        vr = HandleValidator.validate(HandleType.FORMULA, "12345", _ctx())
        assert vr.result is False


class TestUnknownHandleType:
    def test_unregistered_type_passes(self):
        # Use a string that isn't in HandleType enum
        # The validator should return True for unknown types
        class FakeType:
            value = "unknown_type"

        vr = HandleValidator.validate(FakeType(), "anything", _ctx())  # type: ignore
        assert vr.result is True


class TestListTypes:
    def test_list_types_includes_builtins(self):
        types = HandleValidator.list_types()
        assert "file_path" in types
        assert "job_id" in types
        assert "material_id" in types
        assert "formula" in types


# ── Tool-level validate_input() ─────────────────────────────────────


class TestVaspToolValidateInput:
    @pytest.mark.asyncio
    async def test_missing_working_dir(self, tmp_path):
        from huginn.tools.vasp_tool import VaspTool, VaspToolInput

        tool = VaspTool()
        args = VaspToolInput(
            action="relax",
            working_dir=str(tmp_path / "nonexistent"),
        )
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is False
        assert "Working directory" in vr.message

    @pytest.mark.asyncio
    async def test_missing_poscar(self, tmp_path):
        from huginn.tools.vasp_tool import VaspTool, VaspToolInput

        work = tmp_path / "vasp_run"
        work.mkdir()
        # No POSCAR created

        tool = VaspTool()
        args = VaspToolInput(action="relax", working_dir=str(work))
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is False
        assert "POSCAR" in vr.message

    @pytest.mark.asyncio
    async def test_valid_input(self, tmp_path):
        from huginn.tools.vasp_tool import VaspTool, VaspToolInput

        work = tmp_path / "vasp_run"
        work.mkdir()
        (work / "POSCAR").write_text("Fe\n1.0\n...")

        tool = VaspTool()
        args = VaspToolInput(action="scf", working_dir=str(work))
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is True


class TestStructureToolValidateInput:
    @pytest.mark.asyncio
    async def test_missing_file(self):
        from huginn.tools.structure_tool import StructureTool, StructureToolInput

        tool = StructureTool()
        args = StructureToolInput(
            action="read", file_path="/nonexistent/structure.cif"
        )
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is False
        assert "Structure file" in vr.message

    @pytest.mark.asyncio
    async def test_valid_file(self, tmp_path):
        from huginn.tools.structure_tool import StructureTool, StructureToolInput

        f = tmp_path / "POSCAR"
        f.write_text("Fe\n1.0\n...")

        tool = StructureTool()
        args = StructureToolInput(action="read", file_path=str(f))
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is True

    @pytest.mark.asyncio
    async def test_missing_reference(self, tmp_path):
        from huginn.tools.structure_tool import StructureTool, StructureToolInput

        f = tmp_path / "POSCAR"
        f.write_text("Fe\n1.0\n...")

        tool = StructureTool()
        args = StructureToolInput(
            action="compare",
            file_path=str(f),
            reference_path="/nonexistent/ref.cif",
        )
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is False
        assert "Reference file" in vr.message


class TestJobToolValidateInput:
    @pytest.mark.asyncio
    async def test_submit_missing_script(self):
        from huginn.tools.job_tool import JobTool, JobToolInput

        tool = JobTool()
        args = JobToolInput(
            action="submit", script_path="/nonexistent/job.sh"
        )
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is False
        assert "Job script" in vr.message

    @pytest.mark.asyncio
    async def test_submit_valid_script(self, tmp_path):
        from huginn.tools.job_tool import JobTool, JobToolInput

        script = tmp_path / "job.sh"
        script.write_text("#!/bin/bash\necho hello")

        tool = JobTool()
        args = JobToolInput(action="submit", script_path=str(script))
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is True

    @pytest.mark.asyncio
    async def test_status_missing_job_id(self):
        from huginn.tools.job_tool import JobTool, JobToolInput

        tool = JobTool()
        args = JobToolInput(action="status")
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is False
        assert "job_id" in vr.message

    @pytest.mark.asyncio
    async def test_status_valid_job_id(self):
        from huginn.tools.job_tool import JobTool, JobToolInput

        tool = JobTool()
        args = JobToolInput(action="status", job_id="slurm_12345")
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is True

    @pytest.mark.asyncio
    async def test_list_always_passes(self):
        from huginn.tools.job_tool import JobTool, JobToolInput

        tool = JobTool()
        args = JobToolInput(action="list")
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is True

    @pytest.mark.asyncio
    async def test_submit_no_script_path_passes(self):
        """submit without script_path skips file check (may use command instead)."""
        from huginn.tools.job_tool import JobTool, JobToolInput

        tool = JobTool()
        args = JobToolInput(action="submit", command="echo hello")
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is True


class TestLammpsToolValidateInput:
    @pytest.mark.asyncio
    async def test_analyze_trajectory_missing_file(self):
        from huginn.tools.lammps_tool import LammpsTool, LammpsToolInput

        tool = LammpsTool()
        args = LammpsToolInput(
            action="analyze_trajectory",
            trajectory_file="/nonexistent/traj.lammpstrj",
        )
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is False
        assert "Trajectory" in vr.message

    @pytest.mark.asyncio
    async def test_analyze_trajectory_no_file_specified(self):
        from huginn.tools.lammps_tool import LammpsTool, LammpsToolInput

        tool = LammpsTool()
        args = LammpsToolInput(action="analyze_trajectory")
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is False
        assert "not specified" in vr.message

    @pytest.mark.asyncio
    async def test_missing_structure_file(self):
        from huginn.tools.lammps_tool import LammpsTool, LammpsToolInput

        tool = LammpsTool()
        args = LammpsToolInput(
            action="run",
            input_script="units metal",
            structure_file="/nonexistent/struct.data",
        )
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is False
        assert "Structure file" in vr.message

    @pytest.mark.asyncio
    async def test_missing_potential_file(self):
        from huginn.tools.lammps_tool import LammpsTool, LammpsToolInput

        tool = LammpsTool()
        args = LammpsToolInput(
            action="run",
            input_script="units metal",
            potentials=["/nonexistent/potential.eam"],
        )
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is False
        assert "Potential file" in vr.message

    @pytest.mark.asyncio
    async def test_valid_run_inline_script(self, tmp_path):
        from huginn.tools.lammps_tool import LammpsTool, LammpsToolInput

        struct = tmp_path / "struct.data"
        struct.write_text("LAMMPS data file")

        tool = LammpsTool()
        args = LammpsToolInput(
            action="run",
            input_script="units metal\natom_style atomic",
            structure_file=str(struct),
        )
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is True

    @pytest.mark.asyncio
    async def test_analyze_trajectory_valid(self, tmp_path):
        from huginn.tools.lammps_tool import LammpsTool, LammpsToolInput

        traj = tmp_path / "traj.lammpstrj"
        traj.write_text("ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n1\n")

        tool = LammpsTool()
        args = LammpsToolInput(
            action="analyze_trajectory", trajectory_file=str(traj)
        )
        vr = await tool.validate_input(args, _ctx())
        assert vr.result is True



