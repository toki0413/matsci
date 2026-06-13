"""Tests for the native Abaqus helper tool."""

from pathlib import Path

from matsci_agent.tools.abaqus_tool import AbaqusTool, AbaqusToolInput


def test_abaqus_tool_import_packing_reference_points(tmp_path: Path) -> None:
    """AbaqusTool import_packing should generate a script with reference points."""
    tool = AbaqusTool()
    packing_data = {
        "objects": [
            {"id": 0, "center": [1.0, 2.0, 3.0], "radius": 0.5, "symbol": "Si"},
            {"id": 1, "center": [4.0, 5.0, 6.0], "radius": 0.8, "symbol": "Si"},
        ]
    }
    result = tool.call(
        {
            "action": "import_packing",
            "packing_data": packing_data,
            "particle_shape": "reference_point",
            "output_prefix": "particle_abaqus",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    script_path = Path(result.data["script_path"])
    assert script_path.exists()
    script = script_path.read_text(encoding="utf-8")
    assert "ReferencePoint" in script
    assert "Particle_" in script
    assert "1.000000, 2.000000, 3.000000" in script


def test_abaqus_tool_import_packing_spheres(tmp_path: Path) -> None:
    """AbaqusTool import_packing with sphere shape should generate sphere parts."""
    tool = AbaqusTool()
    packing_data = {
        "objects": [
            {"id": 0, "center": [0.0, 0.0, 0.0], "radius": 1.0, "symbol": "Si"},
        ]
    }
    result = tool.call(
        {
            "action": "import_packing",
            "packing_data": packing_data,
            "particle_shape": "sphere",
            "output_prefix": "sphere_abaqus",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    script = Path(result.data["script_path"]).read_text(encoding="utf-8")
    assert "BaseSolidRevolve" in script


def test_abaqus_tool_input_schema() -> None:
    """AbaqusToolInput should accept valid parameters."""
    inp = AbaqusToolInput(
        action="import_packing",
        particle_shape="sphere",
        base_model="Model-1",
    )
    assert inp.particle_shape == "sphere"
    assert inp.base_model == "Model-1"
