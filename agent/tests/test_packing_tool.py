"""Tests for the unified packing and preview tool."""

from pathlib import Path

from huginn.tools.packing_tool import PackingTool, PackingToolInput


def test_packing_tool_molecules(tmp_path: Path) -> None:
    """Pack built-in water molecules into a box."""
    tool = PackingTool()
    result = tool.call(
        {
            "action": "pack",
            "mode": "molecules",
            "components": [
                {"source": "water", "count": 5, "source_type": "name"},
            ],
            "box": [15.0, 15.0, 15.0],
            "tolerance": 2.0,
            "output_prefix": "water_box",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    assert result.data["total_atoms"] == 15
    assert Path(result.data["output_files"]["structure"]).exists()


def test_packing_tool_particles(tmp_path: Path) -> None:
    """Pack spherical particles into a matrix box."""
    tool = PackingTool()
    result = tool.call(
        {
            "action": "pack",
            "mode": "particles",
            "components": [
                {
                    "source": '{"shape": "sphere", "radius": 2.0, "n_points": 30, "symbol": "Si"}',
                    "count": 4,
                    "source_type": "particle",
                },
            ],
            "box": [30.0, 30.0, 30.0],
            "tolerance": 6.0,
            "output_prefix": "composite",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    assert result.data["components"][0]["placed"] == 4
    assert Path(result.data["output_files"]["structure"]).exists()


def test_packing_tool_generates_packmol_input(tmp_path: Path) -> None:
    """Generate a Packmol input file for existing XYZ components."""
    water_xyz = tmp_path / "water.xyz"
    water_xyz.write_text(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n",
        encoding="utf-8",
    )
    tool = PackingTool()
    result = tool.call(
        {
            "action": "generate",
            "mode": "molecules",
            "components": [
                {"source": "water.xyz", "count": 10, "source_type": "xyz"},
            ],
            "box": [20.0, 20.0, 20.0],
            "output_prefix": "mix",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    inp_path = Path(result.data["packmol_input"])
    assert inp_path.exists()
    text = inp_path.read_text(encoding="utf-8")
    assert "tolerance" in text
    assert "structure" in text


def test_packing_tool_preview(tmp_path: Path) -> None:
    """Preview an existing XYZ file."""
    xyz = tmp_path / "existing.xyz"
    xyz.write_text(
        "3\nwater\nO 0.0 0.0 0.0\nH 0.96 0.0 0.0\nH -0.24 0.93 0.0\n",
        encoding="utf-8",
    )
    tool = PackingTool()
    result = tool.call(
        {
            "action": "preview",
            "structure_file": "existing.xyz",
            "output_prefix": "prev",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    assert Path(result.data["image"]).exists()


def test_packing_tool_lammps_data_output(tmp_path: Path) -> None:
    """Pack molecules and write LAMMPS data file."""
    tool = PackingTool()
    result = tool.call(
        {
            "action": "pack",
            "mode": "molecules",
            "components": [
                {"source": "water", "count": 3, "source_type": "name"},
            ],
            "box": [12.0, 12.0, 12.0],
            "output_format": "lammps-data",
            "output_prefix": "lmp_system",
            "working_dir": str(tmp_path),
            "visualize": False,
        }
    )
    assert result.success is True
    data_path = Path(result.data["output_files"]["structure"])
    assert data_path.suffix == ".data"
    text = data_path.read_text(encoding="utf-8")
    assert "LAMMPS data file" in text
    assert "Atoms # atomic" in text


def test_packing_tool_input_schema() -> None:
    """PackingToolInput should validate parameters."""
    inp = PackingToolInput(
        action="pack",
        mode="particles",
        box=[10.0, 10.0, 10.0],
        tolerance=3.0,
    )
    assert inp.mode == "particles"
    assert inp.tolerance == 3.0
