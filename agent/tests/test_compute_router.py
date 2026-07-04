"""Unit tests for huginn/execution/compute_router.py.

Covers the routing decision matrix: user preference overrides, atom-count and
walltime thresholds, fallback for unknown tools, structure parsing, and the
convenience route_stage() entry point.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from huginn.execution.compute_router import ComputeRoute, ComputeRouter


def _cfg(execution_backend="auto", hpc_host=None):
    """Build a lightweight mock config with just the attributes ComputeRouter reads."""
    return SimpleNamespace(execution_backend=execution_backend, hpc_host=hpc_host)


# ── Routing decisions ────────────────────────────────────────────────────────


class TestComputeRouting:
    def test_route_local_small_system(self):
        # Small VASP SCF run, no HPC configured -> stays local.
        router = ComputeRouter(_cfg(execution_backend="auto", hpc_host=None))
        route = router.route("vasp_tool", "scf", {"n_atoms": 10})

        assert route.target == "local"
        assert isinstance(route, ComputeRoute)

    def test_route_hpc_large_system(self):
        # 350-atom relaxation is well past the 50-atom local cap for relax.
        router = ComputeRouter(
            _cfg(execution_backend="auto", hpc_host="cluster.example.com")
        )
        route = router.route("vasp_tool", "relax", {"n_atoms": 350})

        assert route.target == "hpc"
        assert "350" in route.reason
        assert "50" in route.reason

    def test_route_user_pref_local(self):
        # execution_backend=local forces local even for a huge system.
        router = ComputeRouter(
            _cfg(execution_backend="local", hpc_host="cluster.example.com")
        )
        route = router.route("vasp_tool", "relax", {"n_atoms": 350})

        assert route.target == "local"
        assert "user preference" in route.reason

    def test_route_user_pref_remote(self):
        # execution_backend=remote forces HPC even for a tiny system with no host.
        router = ComputeRouter(_cfg(execution_backend="remote", hpc_host=None))
        route = router.route("vasp_tool", "scf", {"n_atoms": 10})

        assert route.target == "hpc"
        assert "user preference" in route.reason

    def test_route_no_hpc_configured(self):
        # Large system but no HPC host -> forced local, can't offload.
        router = ComputeRouter(_cfg(execution_backend="auto", hpc_host=None))
        route = router.route("vasp_tool", "relax", {"n_atoms": 350})

        assert route.target == "local"
        assert "no HPC" in route.reason

    def test_route_walltime_exceeds(self):
        # 10 atoms is fine locally, but 5h walltime blows past the 2h SCF cap.
        router = ComputeRouter(
            _cfg(execution_backend="auto", hpc_host="cluster.example.com")
        )
        route = router.route(
            "vasp_tool", "scf", {"n_atoms": 10, "walltime_hours": 5}
        )

        assert route.target == "hpc"
        assert route.estimated_walltime_hours == 5
        assert "walltime" in route.reason

    def test_route_unknown_tool(self):
        # Tool with no explicit rule falls back to the default 50-atom cap.
        router = ComputeRouter(
            _cfg(execution_backend="auto", hpc_host="cluster.example.com")
        )

        # 100 atoms exceeds the default threshold -> HPC.
        route_hpc = router.route("unknown_tool", "x", {"n_atoms": 100})
        assert route_hpc.target == "hpc"
        assert "50" in route_hpc.reason  # default threshold applied

        # 10 atoms is under the default threshold -> local.
        route_local = router.route("unknown_tool", "x", {"n_atoms": 10})
        assert route_local.target == "local"

    def test_route_stage_dict(self):
        # route_stage() should pull tool/action/params straight from the stage.
        router = ComputeRouter(
            _cfg(execution_backend="auto", hpc_host="cluster.example.com")
        )
        stage = {
            "id": "relax_step",
            "tool": "vasp_tool",
            "action": "relax",
            "params": {"n_atoms": 350},
        }
        route = router.route_stage(stage)

        assert route.target == "hpc"
        assert "350" in route.reason


# ── Atom-count extraction ────────────────────────────────────────────────────


class TestExtractNAtoms:
    def test_extract_n_atoms_xyz(self):
        # XYZ: first line is the atom count, second is a comment.
        xyz = (
            "8\n"
            "water cluster\n"
            "O 0.000 0.000 0.000\n"
            "H 0.757 0.586 0.000\n"
            "H -0.757 0.586 0.000\n"
            "O 2.900 0.000 0.000\n"
            "H 3.657 0.586 0.000\n"
            "H 2.143 0.586 0.000\n"
            "O 5.800 0.000 0.000\n"
            "H 6.557 0.586 0.000\n"
        )
        router = ComputeRouter()

        assert router._extract_n_atoms("vasp_tool", {"structure": xyz}) == 8

    def test_extract_n_atoms_poscar(self):
        # POSCAR: element counts sit on line index 6 (0-based).
        poscar = (
            "SiO2 bulk\n"        # 0: comment (non-numeric, so not misread as XYZ)
            "1.0\n"              # 1: scale
            "5.0 0.0 0.0\n"      # 2: lattice a
            "0.0 5.0 0.0\n"      # 3: lattice b
            "0.0 0.0 5.0\n"      # 4: lattice c
            "Si O\n"             # 5: element symbols
            "4 2\n"              # 6: element counts -> 6 atoms total
            "Direct\n"           # 7: coordinate header
            "0.0 0.0 0.0\n"
            "0.5 0.5 0.5\n"
            "0.0 0.5 0.5\n"
            "0.5 0.0 0.5\n"
            "0.5 0.5 0.0\n"
            "0.25 0.25 0.25\n"
        )
        router = ComputeRouter()

        assert router._extract_n_atoms("vasp_tool", {"poscar": poscar}) == 6
