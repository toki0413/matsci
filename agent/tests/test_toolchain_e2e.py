"""Toolchain end-to-end integration tests.

Validates that multiple tools chained in a realistic research workflow
exchange data correctly through their public contracts. Each test class
is one workflow (Wn); each step feeds its output into the next tool.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from huginn.tools.descriptor_tool import DescriptorTool
from huginn.tools.numerical_tool import NumericalTool
from huginn.tools.structure_tool import StructureTool, StructureToolInput
from huginn.tools.symmetry_tool import SymmetryTool
from huginn.tools.validate_tool import ValidateTool
from huginn.types import ToolContext

SI_POSCAR = str(Path(__file__).parent.parent / "Si_diamond" / "POSCAR")


@pytest.fixture
def context(tmp_path):
    return ToolContext(session_id="e2e", workspace=str(tmp_path))


# ── W1: structure → symmetry → descriptor ─────────────────────────────

@pytest.mark.asyncio
class TestStructureSymmetryDescriptorChain:
    """W1: structure_tool(analyze) → symmetry_tool(analyze) → descriptor_tool(composition)."""

    async def test_chain(self, context):
        pytest.importorskip("pymatgen", reason="pymatgen not installed")

        # Step 1: structure_tool analyzes the Si POSCAR
        struct_tool = StructureTool()
        s1 = await struct_tool.call(
            StructureToolInput(action="analyze", file_path=SI_POSCAR), context
        )
        assert s1.success
        assert s1.data["formula"].startswith("Si")
        assert s1.data["num_atoms"] == 8
        assert "Fd-3m" in (s1.data["spacegroup"] or "")

        # Step 2: symmetry_tool analyzes the same structure
        symm_tool = SymmetryTool()
        s2 = await symm_tool.call(
            {"action": "analyze", "file_path": SI_POSCAR}, context
        )
        assert s2.success
        assert s2.data["space_group_number"] == 227
        assert s2.data["crystal_system"] == "cubic"

        # Step 3: descriptor_tool computes composition features
        desc_tool = DescriptorTool()
        s3 = await desc_tool.call(
            desc_tool.input_schema(action="composition", structure_file=SI_POSCAR),
            context,
        )
        assert s3.success
        features = s3.data["features"]
        assert features["num_elements"] == 1


# ── W3: numerical curve_fit → validate ────────────────────────────────

@pytest.mark.asyncio
class TestEOSFittingChain:
    """W3: numerical_tool(curve_fit) → validate_tool(validate).

    Fits a synthetic E-V curve with a quadratic, then feeds the fit
    diagnostics into validate_tool to confirm the data contract.
    """

    async def test_chain(self, context):
        # Synthetic energy-volume data: exact parabola E = E0 + B*(V-V0)^2
        v0, e0, b = 20.0, -10.5, 0.05
        xdata = [18.0, 18.5, 19.0, 19.5, 20.0, 20.5, 21.0, 21.5, 22.0]
        ydata = [e0 + b * (v - v0) ** 2 for v in xdata]

        # Step 1: numerical_tool fits a quadratic a0*x**2 + a1*x + a2
        num_tool = NumericalTool()
        s1 = await num_tool.call(
            {
                "action": "curve_fit",
                "func_fit": "a0*x**2 + a1*x + a2",
                "xdata": xdata,
                "ydata": ydata,
            },
            context,
        )
        assert s1.success
        params = s1.data["values"]["params"]
        assert len(params) == 3
        r_squared = s1.data["diagnostics"]["r_squared"]
        assert r_squared > 0.999  # exact parabola → near-perfect fit

        # Step 2: validate_tool checks the DFT-style result data
        validate_tool = ValidateTool()
        s2 = await validate_tool.call(
            {
                "action": "validate",
                "result_type": "dft",
                "result_data": {
                    "converged": True,
                    "eos_fit": {"r_squared": r_squared, "params": params},
                    "energy": e0,
                    "volume": v0,
                },
            },
            context,
        )
        assert s2.success
        assert "checks" in s2.data
        assert isinstance(s2.data["checks"], list)


# ── W7: structure → symmetry kpath ────────────────────────────────────

@pytest.mark.asyncio
class TestSymmetryKpathChain:
    """W7: structure_tool(read) → symmetry_tool(kpath)."""

    async def test_chain(self, context):
        pytest.importorskip("pymatgen", reason="pymatgen not installed")

        # Step 1: structure_tool reads the Si POSCAR
        struct_tool = StructureTool()
        s1 = await struct_tool.call(
            StructureToolInput(action="read", file_path=SI_POSCAR), context
        )
        assert s1.success
        assert s1.data["num_atoms"] == 8

        # Step 2: symmetry_tool generates the high-symmetry k-path
        symm_tool = SymmetryTool()
        s2 = await symm_tool.call(
            {"action": "kpath", "file_path": SI_POSCAR}, context
        )
        assert s2.success
        assert s2.data["n_kpoints"] > 0
        assert len(s2.data["kpoints"]) == s2.data["n_kpoints"]
        assert len(s2.data["segments"]) > 0
