"""Tests for the Rust-accelerated VASP OUTCAR parser."""

from pathlib import Path

import pytest

from matsci_agent.tools.vasp_tool import VaspTool, _HAS_MATSCI_EXT


def _write_synthetic_outcar(path: Path) -> None:
    """Write a minimal synthetic OUTCAR for parser testing."""
    lines = [
        "VASP output",
        "ENCUT  =  520.0 eV",
        "ISPIN  =  2",
        "NELM   =  100",
        "NELMIN =  3",
        "k-points in units of 2pi/SCALE and weight:",
        "  0.000  0.000  0.000  1.000",
        "",
        "direct lattice vectors                 reciprocal lattice vectors",
        "  3.500000000  0.000000000  0.000000000     0.285714286  0.000000000  0.000000000",
        "  0.000000000  3.500000000  0.000000000     0.000000000  0.285714286  0.000000000",
        "  0.000000000  0.000000000  3.500000000     0.000000000  0.000000000  0.285714286",
        "",
        "volume of cell :      42.875",
        "",
        "FREE ENERGIE OF THE ION-ELECTRON SYSTEM (eV)",
        "  free  energy   TOTEN  =       -10.1234 eV",
        "",
        "TOTAL-FORCE (eV/Angst)",
        "  0.000  0.000  0.000    0.010  0.020  0.030",
        "  1.750  1.750  0.000   -0.010 -0.020 -0.030",
        "",
        "E-fermi :   5.5000     XC(G=0):  -0.1234     alpha+bet : -0.0500",
        "",
        "magnetization (x)",
        "  # of ion     s       p       d       tot",
        "  1            0.1     0.2     0.3     0.600",
        "  2            0.1     0.2     0.3     0.600",
        "",
        "reached required accuracy - stopping structural energy minimisation",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture
def synthetic_outcar(tmp_path: Path) -> Path:
    path = tmp_path / "OUTCAR"
    _write_synthetic_outcar(path)
    return path


def test_python_outcar_parser_baseline(synthetic_outcar: Path) -> None:
    """Ensure the pure-Python OUTCAR parser still works as a baseline."""
    tool = VaspTool()
    result = tool._parse_outcar_python(synthetic_outcar)
    assert result["energy"] == pytest.approx(-10.1234, abs=1e-9)
    assert result["converged"] is True
    assert result["encut"] == pytest.approx(520.0, abs=1e-9)
    assert result["ispin"] == 2
    assert result["nelm"] == 100
    assert result["volume"] == pytest.approx(42.875, abs=1e-9)
    assert result["efermi"] == pytest.approx(5.5, abs=1e-9)
    assert result["kpoints"] == "found"
    assert len(result["forces"]) == 2
    assert len(result["magnetic_moments"]) == 2


def test_rust_extension_available() -> None:
    """The Rust extension should be importable in this environment."""
    assert _HAS_MATSCI_EXT is True


def test_rust_outcar_parser_matches_python(synthetic_outcar: Path) -> None:
    """Rust OUTCAR parser output should match the Python parser output."""
    tool = VaspTool()

    python_result = tool._parse_outcar_python(synthetic_outcar)
    rust_result = tool._parse_outcar(synthetic_outcar)

    assert rust_result["energy"] == pytest.approx(python_result["energy"], abs=1e-9)
    assert rust_result["converged"] == python_result["converged"]
    assert rust_result["encut"] == pytest.approx(python_result["encut"], abs=1e-9)
    assert rust_result["ispin"] == python_result["ispin"]
    assert rust_result["nelm"] == python_result["nelm"]
    assert rust_result["volume"] == pytest.approx(python_result["volume"], abs=1e-9)
    assert rust_result["efermi"] == pytest.approx(python_result["efermi"], abs=1e-9)
    assert rust_result["kpoints"] == python_result["kpoints"]
    assert rust_result["band_gap"] == python_result["band_gap"]

    assert len(rust_result["forces"]) == len(python_result["forces"])
    for r_f, p_f in zip(rust_result["forces"], python_result["forces"]):
        assert r_f["position"] == pytest.approx(p_f["position"], abs=1e-9)
        assert r_f["force"] == pytest.approx(p_f["force"], abs=1e-9)

    assert rust_result["magnetic_moments"] == pytest.approx(
        python_result["magnetic_moments"], abs=1e-9
    )

    assert len(rust_result["lattice_vectors"]) == len(python_result["lattice_vectors"])
    for r_lv, p_lv in zip(rust_result["lattice_vectors"], python_result["lattice_vectors"]):
        assert r_lv == pytest.approx(p_lv, abs=1e-9)
