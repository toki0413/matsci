"""Tests for DescriptorTool."""

from __future__ import annotations

import pytest

from huginn.tools.descriptor_tool import DescriptorTool
from huginn.types import ToolContext


@pytest.fixture
def tool():
    return DescriptorTool()


@pytest.fixture
def context(tmp_path):
    return ToolContext(session_id="test", workspace=str(tmp_path))


@pytest.mark.asyncio
async def test_composition_descriptor(tool, context):
    result = await tool.call(
        tool.input_schema(action="composition", formula="H2O"), context
    )
    assert result.success is True
    features = result.data["features"]
    assert features["num_elements"] == 2
    assert features["atomic_fractions"]["H"] == pytest.approx(2 / 3, rel=1e-3)


@pytest.mark.asyncio
async def test_composition_from_structure_file(tool, context, tmp_path):
    pytest.importorskip("pymatgen", reason="pymatgen not installed")
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Si1\n1.0\n2.7 0.0 0.0\n0.0 2.7 0.0\n0.0 0.0 2.7\nSi\n1\nDirect\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    result = await tool.call(
        tool.input_schema(action="composition", structure_file=str(poscar)), context
    )
    assert result.success is True
    assert result.data["features"]["formula"] == "Si1"


@pytest.mark.asyncio
async def test_soap_missing_deps(tool, context, tmp_path):
    poscar = tmp_path / "POSCAR"
    poscar.write_text(
        "Si1\n1.0\n2.7 0.0 0.0\n0.0 2.7 0.0\n0.0 0.0 2.7\nSi\n1\nDirect\n0.0 0.0 0.0\n",
        encoding="utf-8",
    )
    result = await tool.call(
        tool.input_schema(action="soap", structure_file=str(poscar)), context
    )
    assert result.success is False
    assert "dscribe" in result.error or "ase" in result.error
