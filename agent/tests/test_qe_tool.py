"""Tests for the Quantum ESPRESSO tool."""

from pathlib import Path

import pytest

from matsci_agent.tools.qe_tool import QuantumEspressoTool, QuantumEspressoToolInput


def test_qe_tool_generates_input(tmp_path: Path) -> None:
    """QuantumEspressoTool should generate a pw.x input file."""
    tool = QuantumEspressoTool(qe_executable=None)
    result = tool.call(
        {
            "action": "generate",
            "working_dir": str(tmp_path),
            "output_prefix": "si_scf",
        }
    )
    assert result.success is True
    input_path = Path(result.data["input_path"])
    assert input_path.exists()
    text = input_path.read_text(encoding="utf-8")
    assert "&CONTROL" in text
    assert "ATOMIC_SPECIES" in text
    assert "K_POINTS" in text
    assert result.data["qe_available"] is False


def test_qe_tool_run_fallback(tmp_path: Path) -> None:
    """Run mode should fall back to input export when pw.x is missing."""
    tool = QuantumEspressoTool(qe_executable=None)
    result = tool.call(
        {
            "action": "run",
            "calculation": "scf",
            "working_dir": str(tmp_path),
            "output_prefix": "si_scf",
        }
    )
    assert result.success is True
    assert result.data["qe_available"] is False
    assert Path(result.data["input_path"]).exists()


def test_qe_tool_parse_output(tmp_path: Path) -> None:
    """Parse a synthetic QE output file."""
    tool = QuantumEspressoTool()
    out_file = tmp_path / "qe.out"
    out_file.write_text(
        "Program PWSCF v.7.0\n"
        "\n"
        "     iteration #  1\n"
        "     iteration #  2\n"
        "\n"
        "!    total energy              =     -10.12345678 Ry\n"
        "\n"
        "     convergence has been achieved\n"
        "\n"
        "     Forces acting on atoms (cartesian axes, Ry/au):\n"
        "\n"
        "     atom    1 type  1   force =     0.0010000    0.0020000    0.0030000\n"
        "     atom    2 type  1   force =    -0.0010000   -0.0020000   -0.0030000\n"
        "\n"
        "     total   stress  (Ry/bohr**3)                   (kbar)     P=       -0.12\n"
        "          -0.00123456   -0.00000000    0.00000000\n"
        "          -0.00000000   -0.00123456    0.00000000\n"
        "           0.00000000    0.00000000   -0.00123456\n",
        encoding="utf-8",
    )

    result = tool.call(
        {
            "action": "parse",
            "working_dir": str(tmp_path),
            "result_files": ["qe.out"],
        }
    )
    assert result.success is True
    parsed = result.data["results"]["qe.out"]
    assert parsed["energy"] == pytest.approx(-10.12345678, abs=1e-9)
    assert parsed["converged"] is True
    assert parsed["n_scf_steps"] == 2
    assert len(parsed["forces"]) == 2
    assert parsed["forces"][0] == pytest.approx([0.001, 0.002, 0.003], abs=1e-9)
    assert len(parsed["stress"]) == 3


def test_qe_tool_input_schema() -> None:
    """QuantumEspressoToolInput should accept valid parameters."""
    inp = QuantumEspressoToolInput(
        action="run",
        calculation="relax",
        ecutwfc=50.0,
    )
    assert inp.calculation == "relax"
    assert inp.ecutwfc == 50.0
