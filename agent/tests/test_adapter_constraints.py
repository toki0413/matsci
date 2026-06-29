"""Tests for automatic constraint checks in ToolAdapter."""

from __future__ import annotations

from pydantic import BaseModel, Field

from huginn.constraints import BoundaryState
from huginn.permissions import PermissionConfig
from huginn.tools.adapter import ToolAdapter
from huginn.tools.base import HuginnTool, ToolProfile
from huginn.types import ToolContext, ToolResult


class _FakeDftInput(BaseModel):
    command: str = Field(default="run")


class _FakeDftTool(HuginnTool):
    name = "vasp_tool"
    description = "Fake DFT tool for testing constraints"
    input_schema = _FakeDftInput
    profile = ToolProfile(constraint_scope="dft")

    async def call(self, args: _FakeDftInput, context: ToolContext) -> ToolResult:
        return ToolResult(data={"energy": 50.0, "max_force": 0.1, "band_gap": -0.5})


class _FakeMdTool(HuginnTool):
    name = "lammps_tool"
    description = "Fake MD tool for testing constraints"
    input_schema = _FakeDftInput
    profile = ToolProfile(constraint_scope="md")

    async def call(self, args: _FakeDftInput, context: ToolContext) -> ToolResult:
        return ToolResult(
            data={
                "energy_drift_per_atom": 0.1,
                "temperature_std": 100.0,
                "target_temperature": 300.0,
                "initial_atom_count": 10,
                "final_atom_count": 8,
                "density": 200.0,
            }
        )


class _FakeReadOnlyTool(HuginnTool):
    name = "structure_tool"
    description = "Fake read-only tool with no constraint scope"
    input_schema = _FakeDftInput
    read_only = True

    async def call(self, args: _FakeDftInput, context: ToolContext) -> ToolResult:
        return ToolResult(data={"formula": "Si"})


class TestAdapterConstraints:
    def test_dft_block_on_invalid_volume(self):
        class _BadVolumeDftTool(HuginnTool):
            name = "vasp_tool"
            description = "Fake DFT tool with invalid volume"
            input_schema = _FakeDftInput
            profile = ToolProfile(constraint_scope="dft")

            async def call(self, args, context):
                return ToolResult(
                    data={
                        "energy": -10.0,
                        "max_force": 0.005,
                        "band_gap": 1.0,
                        "volume": -1.0,
                    }
                )

        lc_tool = ToolAdapter().adapt(
            _BadVolumeDftTool(),
            permission_config=PermissionConfig(auto_approve_all=True),
        )
        output = lc_tool.invoke({"command": "run"})
        assert output.get("error") is not None
        assert "Constraint check failed" in output["error"]
        assert "volume_positive" in output["error"]

    def test_dft_warning_attached_but_not_blocked(self):
        class _AlmostOkDftTool(HuginnTool):
            name = "vasp_tool"
            description = "Fake DFT tool with only warning-level issues"
            input_schema = _FakeDftInput
            profile = ToolProfile(constraint_scope="dft")

            async def call(self, args, context):
                # energy sign OK, force slightly high (warn), band gap OK
                return ToolResult(
                    data={"energy": -10.0, "max_force": 0.1, "band_gap": 1.0}
                )

        lc_tool = ToolAdapter().adapt(
            _AlmostOkDftTool(),
            permission_config=PermissionConfig(auto_approve_all=True),
        )
        output = lc_tool.invoke({"command": "run"})
        assert "error" not in output
        result = output["result"]
        assert "_constraint_warnings" in result
        warnings = {w["name"] for w in result["_constraint_warnings"]}
        assert "force_convergence" in warnings

    def test_unmapped_tool_skips_constraints(self):
        lc_tool = ToolAdapter().adapt(_FakeReadOnlyTool())
        output = lc_tool.invoke({"command": "run"})
        assert "error" not in output
        assert output["result"]["formula"] == "Si"

    def test_md_block_on_lost_atoms(self):
        lc_tool = ToolAdapter().adapt(
            _FakeMdTool(), permission_config=PermissionConfig(auto_approve_all=True)
        )
        output = lc_tool.invoke({"command": "run"})
        assert output.get("error") is not None
        assert "atom_count" in output["error"]


class TestBoundaryEvolution:
    def test_block_evolution_sets_require_confirmation(self):
        class _BadDftTool(HuginnTool):
            name = "vasp_tool"
            description = "Bad DFT"
            input_schema = _FakeDftInput
            profile = ToolProfile(constraint_scope="dft")

            async def call(self, args, context):
                return ToolResult(
                    data={
                        "energy": -10.0,
                        "max_force": 0.005,
                        "band_gap": 1.0,
                        "volume": -1.0,
                    }
                )

        boundary = BoundaryState(require_confirmation=False)
        lc_tool = ToolAdapter().adapt(
            _BadDftTool(),
            permission_config=PermissionConfig(auto_approve_all=True),
            boundary_state=boundary,
        )
        output = lc_tool.invoke({"command": "run"})
        assert "error" in output
        assert boundary.require_confirmation is True
        assert boundary.max_retries == 1

    def test_blocked_tool_denied_on_subsequent_call(self):
        class _BadDftTool(HuginnTool):
            name = "vasp_tool"
            description = "Bad DFT"
            input_schema = _FakeDftInput
            profile = ToolProfile(constraint_scope="dft")

            async def call(self, args, context):
                return ToolResult(
                    data={
                        "energy": -10.0,
                        "max_force": 0.005,
                        "band_gap": 1.0,
                        "volume": -1.0,
                    }
                )

        class _AnotherDftTool(HuginnTool):
            name = "qe_tool"
            description = "Another DFT"
            input_schema = _FakeDftInput
            profile = ToolProfile(constraint_scope="dft")

            async def call(self, args, context):
                return ToolResult(data={"energy": -10.0})

        boundary = BoundaryState()
        bad = ToolAdapter().adapt(
            _BadDftTool(),
            permission_config=PermissionConfig(auto_approve_all=True),
            boundary_state=boundary,
        )
        another = ToolAdapter().adapt(
            _AnotherDftTool(),
            permission_config=PermissionConfig(auto_approve_all=True),
            boundary_state=boundary,
        )
        bad.invoke({"command": "run"})
        assert boundary.require_confirmation is True
        output = another.invoke({"command": "run"})
        assert "error" in output
        assert "blocked by dynamic boundary" in output["error"]
