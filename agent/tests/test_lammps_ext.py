"""Tests for the Rust-accelerated LAMMPS trajectory parser."""

from pathlib import Path

import pytest

from matsci_agent.tools.lammps_tool import LammpsTool, _HAS_MATSCI_EXT


TRJ_PATH = Path(__file__).parent.parent / "lammps_traj_test" / "traj.lammpstrj"


def test_python_parser_baseline():
    """Ensure the pure-Python parser still works as a baseline."""
    tool = LammpsTool()
    result = tool._parse_trajectory_python(TRJ_PATH)
    assert result["n_frames"] == 3
    assert result["n_atoms"] == 32
    assert result["atom_types"] == [1]
    assert result["timesteps"] == [0, 10, 20]
    assert "msd" in result
    assert "rdf" in result


@pytest.mark.skipif(not _HAS_MATSCI_EXT, reason="matsci_ext Rust extension not installed")
def test_rust_extension_available():
    """The Rust extension should be importable in this environment."""
    assert _HAS_MATSCI_EXT is True


@pytest.mark.skipif(not _HAS_MATSCI_EXT, reason="matsci_ext Rust extension not installed")
def test_rust_parser_matches_python():
    """Rust parser output should match the Python parser output."""
    tool = LammpsTool()

    python_result = tool._parse_trajectory_python(TRJ_PATH)
    rust_result = tool.parse_trajectory(TRJ_PATH)

    assert rust_result["n_frames"] == python_result["n_frames"]
    assert rust_result["n_atoms"] == python_result["n_atoms"]
    assert rust_result["atom_types"] == python_result["atom_types"]
    assert rust_result["timesteps"] == python_result["timesteps"]
    assert rust_result["box_bounds"] == python_result["box_bounds"]

    # MSD values should match exactly for this deterministic trajectory.
    for rust_msd, py_msd in zip(rust_result["msd"], python_result["msd"]):
        assert rust_msd["timestep"] == py_msd["timestep"]
        assert rust_msd["msd"] == pytest.approx(py_msd["msd"], abs=1e-9)

    # RDF values should match within tolerance.
    rust_rdf = rust_result["rdf"]
    py_rdf = python_result["rdf"]
    assert rust_rdf["bins"] == py_rdf["bins"]
    assert rust_rdf["r_max"] == pytest.approx(py_rdf["r_max"], abs=1e-9)
    for r_r, p_r in zip(rust_rdf["r"], py_rdf["r"]):
        assert r_r == pytest.approx(p_r, abs=1e-9)
    for r_g, p_g in zip(rust_rdf["g"], py_rdf["g"]):
        assert r_g == pytest.approx(p_g, abs=1e-9)


@pytest.mark.skipif(not _HAS_MATSCI_EXT, reason="matsci_ext Rust extension not installed")
def test_rust_parser_raw_api():
    """Test the raw matsci_ext API directly."""
    import matsci_ext

    result = matsci_ext.parse_lammps_dump(
        str(TRJ_PATH),
        compute_msd=True,
        compute_rdf=True,
        include_frames=True,
    )
    assert result["n_frames"] == 3
    assert result["n_atoms"] == 32
    assert len(result["frames"]) == 3
    assert len(result["frames"][0]["atoms"]) == 32
    assert "id" in result["frames"][0]["atoms"][0]


@pytest.mark.skipif(not _HAS_MATSCI_EXT, reason="matsci_ext Rust extension not installed")
def test_general_msd() -> None:
    """Test compute_msd on a NumPy array against a reference implementation."""
    import numpy as np
    import matsci_ext

    np.random.seed(0)
    positions = np.random.rand(5, 8, 3).astype(np.float64)
    timesteps = [0, 10, 20, 30, 40]

    result = matsci_ext.compute_msd(positions, timesteps=timesteps)
    msd = result["msd"]

    ref = positions[0]
    for i, entry in enumerate(msd):
        frame = positions[i + 1]
        displacements = np.sum((frame - ref) ** 2, axis=1)
        expected = float(np.mean(displacements))
        assert entry["timestep"] == timesteps[i + 1]
        assert entry["msd"] == pytest.approx(expected, abs=1e-9)


@pytest.mark.skipif(not _HAS_MATSCI_EXT, reason="matsci_ext Rust extension not installed")
def test_general_rdf() -> None:
    """Test compute_rdf on a NumPy array against a reference implementation."""
    import numpy as np
    import matsci_ext

    np.random.seed(1)
    positions = np.random.rand(20, 3).astype(np.float64)
    box = [1.0, 1.0, 1.0]
    bins = 20

    result = matsci_ext.compute_rdf(positions, box_dims=box, bins=bins)
    assert result["bins"] == bins
    assert len(result["r"]) == bins
    assert len(result["g"]) == bins
    assert result["r_max"] == pytest.approx(0.5, abs=1e-9)

    # Sanity checks: non-negative g and monotonically increasing r.
    assert all(g >= 0 for g in result["g"])
    assert all(result["r"][i] < result["r"][i + 1] for i in range(bins - 1))
