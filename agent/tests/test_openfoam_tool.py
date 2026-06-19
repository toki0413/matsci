"""Tests for the OpenFOAM CFD tool."""

from pathlib import Path

import pytest

from huginn.tools.openfoam_tool import OpenFoamTool, OpenFoamToolInput


def test_openfoam_tool_generates_case(tmp_path: Path) -> None:
    """OpenFoamTool should generate a full case directory in generate mode."""
    tool = OpenFoamTool(openfoam_dir=None)
    result = tool.call(
        {
            "action": "generate",
            "solver": "icoFoam",
            "case_name": "pipe_flow",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    case_dir = Path(result.data["case_dir"])
    assert case_dir.exists()
    assert (case_dir / "system" / "controlDict").exists()
    assert (case_dir / "system" / "blockMeshDict").exists()
    assert (case_dir / "system" / "fvSchemes").exists()
    assert (case_dir / "system" / "fvSolution").exists()
    assert (case_dir / "constant" / "transportProperties").exists()
    assert (case_dir / "0" / "p").exists()
    assert (case_dir / "0" / "U").exists()
    assert result.data["openfoam_available"] is False


def test_openfoam_tool_run_fallback(tmp_path: Path) -> None:
    """Run mode should fall back to case export when OpenFOAM is missing."""
    tool = OpenFoamTool(openfoam_dir=None)
    result = tool.call(
        {
            "action": "run",
            "solver": "icoFoam",
            "case_name": "pipe_flow",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    assert result.data["openfoam_available"] is False
    assert Path(result.data["case_dir"]).exists()


def test_openfoam_tool_ras_case_has_turbulence_fields(tmp_path: Path) -> None:
    """simpleFoam case should include RAS turbulence properties and fields."""
    tool = OpenFoamTool(openfoam_dir=None)
    result = tool.call(
        {
            "action": "generate",
            "solver": "simpleFoam",
            "case_name": "turbulent_channel",
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    case_dir = Path(result.data["case_dir"])
    assert (case_dir / "constant" / "turbulenceProperties").exists()
    assert (case_dir / "0" / "k").exists()
    assert (case_dir / "0" / "omega").exists()
    assert (case_dir / "0" / "nut").exists()


def test_openfoam_tool_parse_log(tmp_path: Path) -> None:
    """Parse mode should extract final time and residuals from a solver log."""
    tool = OpenFoamTool()
    log = tmp_path / "icoFoam.log"
    log.write_text(
        "\n"
        "Time = 0.5\n"
        "\n"
        "PISO loop\n"
        "DILUPBiCGStab: Solving for Ux, Initial residual = 0.01, Final residual = 1e-08, No Iterations 2\n"
        "DICPCG: Solving for p, Initial residual = 0.1, Final residual = 1e-06, No Iterations 10\n"
        "time step continuity errors : sum local = 1.2e-06, global = -4.3e-08\n"
        "ExecutionTime = 0.1 s\n"
        "\n"
        "Time = 1.0\n"
        "\n"
        "End\n",
        encoding="utf-8",
    )

    result = tool.call(
        {
            "action": "parse",
            "working_dir": str(tmp_path),
            "result_files": ["icoFoam.log"],
        }
    )
    assert result.success is True
    parsed = result.data["results"]["icoFoam.log"]
    assert parsed["final_time"] == pytest.approx(1.0, abs=1e-9)
    assert parsed["converged"] is True
    assert parsed["final_residuals"]["Ux"] == pytest.approx(1e-08, abs=1e-12)
    assert parsed["final_residuals"]["p"] == pytest.approx(1e-06, abs=1e-12)
    assert parsed["continuity_errors"]


def test_openfoam_tool_input_schema() -> None:
    """OpenFoamToolInput should accept valid parameters."""
    inp = OpenFoamToolInput(
        action="run",
        solver="simpleFoam",
        end_time=5.0,
        delta_t=0.01,
    )
    assert inp.solver == "simpleFoam"
    assert inp.end_time == 5.0


def test_openfoam_tool_set_fields_from_packing(tmp_path: Path) -> None:
    """OpenFoamTool set_fields action should consume packing_tool output."""
    from huginn.tools.packing_tool import PackingTool

    packer = PackingTool()
    pack_result = packer.call(
        {
            "action": "pack",
            "mode": "particles",
            "components": [
                {
                    "source": '{"shape": "sphere", "radius": 1.5, "n_points": 30, "symbol": "Si"}',
                    "count": 3,
                    "source_type": "particle",
                },
            ],
            "box": [20.0, 20.0, 20.0],
            "tolerance": 4.0,
            "output_prefix": "particles",
            "working_dir": str(tmp_path),
            "visualize": False,
        }
    )
    assert pack_result.success is True
    assert len(pack_result.data["objects"]) == 3

    openfoam = OpenFoamTool(openfoam_dir=None)
    result = openfoam.call(
        {
            "action": "set_fields",
            "case_name": "multiphase",
            "packing_data": pack_result.data,
            "field_name": "alpha.water",
            "default_value": 0.0,
            "set_value": 1.0,
            "working_dir": str(tmp_path),
        }
    )
    assert result.success is True
    case_dir = Path(result.data["case_dir"])
    assert (case_dir / "system" / "setFieldsDict").exists()
    set_fields_text = (case_dir / "system" / "setFieldsDict").read_text(
        encoding="utf-8"
    )
    assert "sphereToCell" in set_fields_text
    assert "alpha.water" in set_fields_text
    assert (case_dir / "0" / "alpha.water").exists()
    alpha_text = (case_dir / "0" / "alpha.water").read_text(encoding="utf-8")
    assert "volScalarField" in alpha_text
    assert "uniform 0" in alpha_text
