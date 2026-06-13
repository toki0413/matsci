"""Tests for the CP2K tool."""

from pathlib import Path

import pytest

from matsci_agent.tools.cp2k_tool import Cp2kTool, Cp2kToolInput


def test_cp2k_tool_generates_input(tmp_path: Path) -> None:
    """Cp2kTool should generate a CP2K input file."""
    tool = Cp2kTool(cp2k_executable=None)
    result = tool.call(
        {
            "action": "generate",
            "working_dir": str(tmp_path),
            "output_prefix": "si_dft",
        }
    )
    assert result.success is True
    input_path = Path(result.data["input_path"])
    assert input_path.exists()
    text = input_path.read_text(encoding="utf-8")
    assert "&GLOBAL" in text
    assert "RUN_TYPE ENERGY_FORCE" in text
    assert "&FORCE_EVAL" in text
    assert result.data["cp2k_available"] is False


def test_cp2k_tool_run_fallback(tmp_path: Path) -> None:
    """Run mode should fall back to input export when CP2K is missing."""
    tool = Cp2kTool(cp2k_executable=None)
    result = tool.call(
        {
            "action": "run",
            "run_type": "MD",
            "working_dir": str(tmp_path),
            "output_prefix": "si_md",
        }
    )
    assert result.success is True
    assert result.data["cp2k_available"] is False
    assert Path(result.data["input_path"]).exists()


def test_cp2k_tool_parse_output(tmp_path: Path) -> None:
    """Parse a synthetic CP2K output file."""
    tool = Cp2kTool()
    out_file = tmp_path / "cp2k.out"
    out_file.write_text(
        " ***  CP2K ***\n"
        "\n"
        " SCF iteration\n"
        " SCF iteration\n"
        " *** SCF run converged ***\n"
        "\n"
        "  Total energy:                                              -10.1234567890\n"
        "\n"
        " ATOMIC FORCES in [a.u.]\n"
        " # Atom   Kind   Element       X            Y            Z\n"
        "      1      1     Si          0.001000     0.002000     0.003000\n"
        "      2      1     Si         -0.001000    -0.002000    -0.003000\n"
        "\n"
        " STRESS|                        1          2          3\n"
        " STRESS|      1    -0.12345678   0.00000000   0.00000000\n"
        " STRESS|      2     0.00000000  -0.12345678   0.00000000\n"
        " STRESS|      3     0.00000000   0.00000000  -0.12345678\n",
        encoding="utf-8",
    )

    result = tool.call(
        {
            "action": "parse",
            "working_dir": str(tmp_path),
            "result_files": ["cp2k.out"],
        }
    )
    assert result.success is True
    parsed = result.data["results"]["cp2k.out"]
    assert parsed["energy"] == pytest.approx(-10.1234567890, abs=1e-9)
    assert parsed["converged"] is True
    assert parsed["n_scf_steps"] == 2
    assert len(parsed["forces"]) == 2
    assert parsed["forces"][0] == pytest.approx([0.001, 0.002, 0.003], abs=1e-9)
    assert len(parsed["stress"]) == 3


def test_cp2k_tool_input_schema() -> None:
    """Cp2kToolInput should accept valid parameters."""
    inp = Cp2kToolInput(
        action="run",
        run_type="GEO_OPT",
        cutoff=600,
    )
    assert inp.run_type == "GEO_OPT"
    assert inp.cutoff == 600
