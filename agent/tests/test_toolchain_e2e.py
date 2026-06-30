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
from huginn.tools.xrd_sim_tool import XrdSimTool
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


# ── W2: structure → XRD simulation ────────────────────────────────────

@pytest.mark.asyncio
class TestStructureXRDSimChain:
    """W2: structure_tool(read) → xrd_sim_tool(simulate_xrd)."""

    async def test_chain(self, context):
        pytest.importorskip("pymatgen", reason="pymatgen not installed")

        # Step 1: structure_tool reads the Si POSCAR
        struct_tool = StructureTool()
        s1 = await struct_tool.call(
            StructureToolInput(action="read", file_path=SI_POSCAR), context
        )
        assert s1.success
        assert s1.data["num_atoms"] == 8

        # Step 2: xrd_sim_tool simulates the powder pattern
        xrd_tool = XrdSimTool()
        s2 = await xrd_tool.call(
            {"action": "simulate_xrd", "file_path": SI_POSCAR, "wavelength": 1.5406},
            context,
        )
        assert s2.success
        peaks = s2.data["peaks"]
        assert len(peaks) > 0
        # Si (a=4.43) strongest (111) reflection near 2θ ≈ 35°
        two_theta_values = [p["two_theta"] for p in peaks]
        assert any(34.0 <= t <= 36.0 for t in two_theta_values)
        assert s2.data["structure"] == "Si"


# ── W4: structure → symmetry conventional → validate ─────────────────

@pytest.mark.asyncio
class TestStructureSymmetryValidateChain:
    """W4: structure_tool(analyze) → symmetry_tool(conventional) → validate_tool(validate)."""

    async def test_chain(self, context):
        pytest.importorskip("pymatgen", reason="pymatgen not installed")

        # Step 1: structure_tool analyzes the Si POSCAR
        struct_tool = StructureTool()
        s1 = await struct_tool.call(
            StructureToolInput(action="analyze", file_path=SI_POSCAR), context
        )
        assert s1.success
        lattice_a = s1.data["lattice_params"]["a"]

        # Step 2: symmetry_tool gets the conventional cell
        symm_tool = SymmetryTool()
        s2 = await symm_tool.call(
            {"action": "conventional", "file_path": SI_POSCAR}, context
        )
        assert s2.success
        assert s2.data["formula"] == "Si"
        assert s2.data["n_atoms_conventional"] >= 8

        # Step 3: validate_tool checks the lattice constant
        validate_tool = ValidateTool()
        s3 = await validate_tool.call(
            {
                "action": "validate",
                "result_type": "dft",
                "result_data": {
                    "converged": True,
                    "lattice_constant": lattice_a,
                    "structure": "Si",
                },
            },
            context,
        )
        assert s3.success
        assert isinstance(s3.data["checks"], list)


# ── W5: numerical root → integrate → validate ─────────────────────────

@pytest.mark.asyncio
class TestNumericalChain:
    """W5: numerical_tool(root) → numerical_tool(integrate) → validate_tool(validate).

    Finds the root of x²−4 (=2), integrates x² from 0 to that root, then
    validates the computed integral as a DFT-style scalar result.
    """

    async def test_chain(self, context):
        num_tool = NumericalTool()

        # Step 1: find root of x**2 - 4 starting at x0=1.0 → 2.0
        s1 = await num_tool.call(
            {"action": "root", "func": "x**2 - 4", "x0": 1.0}, context
        )
        assert s1.success
        root = s1.data["values"]["root"]
        assert root == pytest.approx(2.0, abs=1e-4)

        # Step 2: integrate x**2 from 0 to root → 8/3
        s2 = await num_tool.call(
            {"action": "integrate", "func": "x**2", "a": 0.0, "b": root}, context
        )
        assert s2.success
        integral = s2.data["values"]["integral"]
        assert integral == pytest.approx(8.0 / 3.0, abs=1e-4)

        # Step 3: validate the computed value
        validate_tool = ValidateTool()
        s3 = await validate_tool.call(
            {
                "action": "validate",
                "result_type": "dft",
                "result_data": {
                    "converged": True,
                    "computed_value": integral,
                    "structure": "Si",
                },
            },
            context,
        )
        assert s3.success
        assert "checks" in s3.data


# ── W6: structure → XRD simulate → index peaks ────────────────────────

@pytest.mark.asyncio
class TestXRDSimIndexChain:
    """W6: structure_tool(read) → xrd_sim_tool(simulate_xrd) → xrd_sim_tool(index_peaks)."""

    async def test_chain(self, context):
        pytest.importorskip("pymatgen", reason="pymatgen not installed")

        # Step 1: structure_tool reads the Si POSCAR
        struct_tool = StructureTool()
        s1 = await struct_tool.call(
            StructureToolInput(action="read", file_path=SI_POSCAR), context
        )
        assert s1.success

        # Step 2: xrd_sim_tool simulates the pattern
        xrd_tool = XrdSimTool()
        s2 = await xrd_tool.call(
            {"action": "simulate_xrd", "file_path": SI_POSCAR}, context
        )
        assert s2.success
        observed_peaks = [p["two_theta"] for p in s2.data["peaks"]]
        assert len(observed_peaks) > 0

        # Step 3: xrd_sim_tool indexes the observed peaks against the structure
        s3 = await xrd_tool.call(
            {
                "action": "index_peaks",
                "file_path": SI_POSCAR,
                "peaks": observed_peaks,
                "tolerance": 0.3,
            },
            context,
        )
        assert s3.success
        indexed = s3.data["indexed_peaks"]
        assert len(indexed) == len(observed_peaks)
        # every peak should match a Miller index from the simulated pattern
        assert s3.data["n_indexed"] == len(observed_peaks)
        for entry in indexed:
            assert entry["hkl"] is not None
