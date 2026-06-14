"""Tests for the COMSOL Multiphysics tool."""

from pathlib import Path

import pytest

from huginn.tools.comsol_tool import ComsolTool, ComsolToolInput


def test_comsol_tool_generates_script(tmp_path: Path) -> None:
    """ComsolTool should generate a Java script in generate mode."""
    tool = ComsolTool()
    result = tool.call(
        {
            "action": "generate",
            "physics": "solid_mechanics",
            "working_dir": str(tmp_path),
            "output_prefix": "beam",
        }
    )
    assert result.success is True
    script_path = Path(result.data["script_path"])
    assert script_path.exists()
    assert "com.comsol.model" in script_path.read_text(encoding="utf-8")
    assert result.data["comsol_available"] is False  # Not installed in test env


def test_comsol_tool_run_fallback_to_script(tmp_path: Path) -> None:
    """ComsolTool run mode should fall back to script export when COMSOL is missing."""
    tool = ComsolTool(comsol_executable=None)
    result = tool.call(
        {
            "action": "run",
            "physics": "solid_mechanics",
            "working_dir": str(tmp_path),
            "output_prefix": "beam",
        }
    )
    assert result.success is True
    assert result.data["comsol_available"] is False
    assert Path(result.data["script_path"]).exists()


def test_comsol_tool_parse_csv(tmp_path: Path) -> None:
    """ComsolTool parse mode should read space/CSV result files."""
    tool = ComsolTool()
    result_file = tmp_path / "results.txt"
    result_file.write_text(
        "x y z stress\n"
        "0.0 0.0 0.0 1.0\n"
        "1.0 0.0 0.0 2.0\n",
        encoding="utf-8",
    )

    result = tool.call(
        {
            "action": "parse",
            "working_dir": str(tmp_path),
            "result_files": ["results.txt"],
        }
    )
    assert result.success is True
    parsed = result.data["results"]["results.txt"]
    assert parsed["header"] == ["x", "y", "z", "stress"]
    assert parsed["n_rows"] == 2
    assert parsed["rows"][0]["stress"] == pytest.approx(1.0, abs=1e-9)


def test_comsol_tool_thermal_script(tmp_path: Path) -> None:
    """ComsolTool should generate a thermal physics script."""
    tool = ComsolTool()
    result = tool.call(
        {
            "action": "generate",
            "physics": "thermal",
            "working_dir": str(tmp_path),
            "output_prefix": "thermal",
        }
    )
    assert result.success is True
    script = Path(result.data["script_path"]).read_text(encoding="utf-8")
    assert "HeatTransfer" in script


def test_comsol_tool_input_schema() -> None:
    """ComsolToolInput should accept valid parameters."""
    inp = ComsolToolInput(
        action="run",
        physics="solid_mechanics",
        material={"youngs_modulus": 70e9, "poissons_ratio": 0.33, "density": 2700.0},
    )
    assert inp.physics == "solid_mechanics"
    assert inp.material["youngs_modulus"] == 70e9


def test_comsol_tool_import_packing(tmp_path: Path) -> None:
    """ComsolTool import_packing should generate a Java script with spheres."""
    tool = ComsolTool()
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
            "output_prefix": "particle_comsol",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    script_path = Path(result.data["script_path"])
    assert script_path.exists()
    script = script_path.read_text(encoding="utf-8")
    assert "Sphere" in script
    assert '"sph" + i' in script
    assert "1.000000, 2.000000, 3.000000" in script
