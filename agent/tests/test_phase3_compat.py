"""Tests for Phase 3 compat.py — analysis and visualization endpoints."""

from __future__ import annotations

import pytest
import numpy as np

from huginn.routes.compat import (
    _crystal_system,
    _guess_crystal_system,
    _ascii_plot,
    _ascii_phase,
)


# ═══════════════════════════════════════════════════════════════════
# Crystal system helpers
# ═══════════════════════════════════════════════════════════════════


class TestCrystalSystem:
    def test_triclinic(self):
        assert _crystal_system(1) == "triclinic"
        assert _crystal_system(2) == "triclinic"

    def test_monoclinic(self):
        assert _crystal_system(3) == "monoclinic"
        assert _crystal_system(15) == "monoclinic"

    def test_orthorhombic(self):
        assert _crystal_system(16) == "orthorhombic"
        assert _crystal_system(74) == "orthorhombic"

    def test_tetragonal(self):
        assert _crystal_system(75) == "tetragonal"
        assert _crystal_system(142) == "tetragonal"

    def test_trigonal(self):
        assert _crystal_system(143) == "trigonal"
        assert _crystal_system(167) == "trigonal"

    def test_hexagonal(self):
        assert _crystal_system(168) == "hexagonal"
        assert _crystal_system(194) == "hexagonal"

    def test_cubic(self):
        assert _crystal_system(195) == "cubic"
        assert _crystal_system(230) == "cubic"


class TestGuessCrystalSystem:
    def test_cubic(self):
        norms = np.array([4.0, 4.0, 4.0])
        angles = [90.0, 90.0, 90.0]
        assert _guess_crystal_system(norms, angles) == "cubic"

    def test_tetragonal(self):
        norms = np.array([4.0, 4.0, 6.0])
        angles = [90.0, 90.0, 90.0]
        assert _guess_crystal_system(norms, angles) == "tetragonal"

    def test_orthorhombic(self):
        norms = np.array([3.0, 4.0, 5.0])
        angles = [90.0, 90.0, 90.0]
        assert _guess_crystal_system(norms, angles) == "orthorhombic"

    def test_hexagonal(self):
        norms = np.array([3.0, 3.0, 5.0])
        angles = [90.0, 90.0, 120.0]
        assert _guess_crystal_system(norms, angles) == "hexagonal"

    def test_triclinic_fallback(self):
        norms = np.array([3.0, 4.0, 5.0])
        angles = [80.0, 85.0, 95.0]
        assert _guess_crystal_system(norms, angles) == "triclinic"


# ═══════════════════════════════════════════════════════════════════
# ASCII plot helpers
# ═══════════════════════════════════════════════════════════════════


class TestAsciiPlot:
    def test_basic_plot(self):
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        result = _ascii_plot(x, y, "sin")
        assert "sin" in result
        assert len(result) > 0

    def test_empty_data(self):
        result = _ascii_plot(np.array([]), np.array([]), "test")
        assert result == "(empty data)"


class TestAsciiPhase:
    def test_basic_phase(self):
        t = np.linspace(0, 2 * np.pi, 100)
        x = np.cos(t)
        y = np.sin(t)
        result = _ascii_phase(x, y)
        assert "." in result  # should have some points

    def test_empty_phase(self):
        result = _ascii_phase(np.array([]), np.array([]))
        assert result == "(empty data)"


# ═══════════════════════════════════════════════════════════════════
# Compat API endpoint tests (via TestClient)
# ═══════════════════════════════════════════════════════════════════


class TestCompatEndpoints:
    @pytest.fixture
    def client(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from huginn.routes.compat import router

        app = FastAPI()
        app.include_router(router)
        return TestClient(app)

    def test_firewall_status(self, client):
        resp = client.get("/firewall/status")
        assert resp.status_code == 200
        assert resp.json()["enabled"] is False

    # ── Symmetry ──

    def test_symmetry_missing_params(self, client):
        resp = client.post("/analyze/symmetry", json={})
        assert "error" in resp.json()

    def test_symmetry_fallback(self, client):
        # Cubic cell: a=b=c=4, all 90°
        lattice = [[4, 0, 0], [0, 4, 0], [0, 0, 4]]
        positions = [[0, 0, 0]]
        numbers = [14]
        resp = client.post("/analyze/symmetry", json={
            "lattice": lattice,
            "positions": positions,
            "numbers": numbers,
        })
        data = resp.json()
        assert "crystal_system" in data
        assert data["crystal_system"] == "cubic"

    # ── Spectral ──

    def test_spectral_missing_data(self, client):
        resp = client.post("/analyze/spectral", json={})
        assert "error" in resp.json()

    def test_spectral_basic(self, client):
        # Sine wave at 5 Hz
        t = np.linspace(0, 1, 100)
        signal = np.sin(2 * np.pi * 5 * t).tolist()
        resp = client.post("/analyze/spectral", json={
            "data": signal,
            "sample_rate": 100.0,
        })
        data = resp.json()
        assert data["n_samples"] == 100
        assert len(data["peaks"]) > 0
        # Dominant frequency should be ~5 Hz
        assert abs(data["dominant_frequency"] - 5.0) < 2.0

    # ── Dynamics ──

    def test_dynamics_missing_positions(self, client):
        resp = client.post("/analyze/dynamics", json={})
        assert "error" in resp.json()

    def test_dynamics_basic(self, client):
        # 3 frames, 2 atoms, 3D
        positions = [
            [[0, 0, 0], [1, 0, 0]],
            [[0.1, 0, 0], [1.1, 0, 0]],
            [[0.2, 0, 0], [1.2, 0, 0]],
        ]
        resp = client.post("/analyze/dynamics", json={
            "positions": positions,
            "timestep": 1.0,
        })
        data = resp.json()
        assert data["n_frames"] == 3
        assert data["n_atoms"] == 2
        assert "msd" in data
        assert len(data["msd"]) == 3
        assert data["msd"][0] == 0.0  # first frame = no displacement

    # ── TDA ──

    def test_tda_missing_data(self, client):
        resp = client.post("/analyze/tda", json={})
        assert "error" in resp.json()

    def test_tda_point_cloud(self, client):
        # Random point cloud
        np.random.seed(42)
        points = np.random.randn(20, 3).tolist()
        resp = client.post("/analyze/tda", json={"data": points})
        data = resp.json()
        assert data["n_points"] == 20
        assert data["embedding_dimension"] == 3

    # ── SINDy ──

    def test_sindy_missing_data(self, client):
        resp = client.post("/analyze/sindy", json={})
        assert "error" in resp.json()

    def test_sindy_linear_system(self, client):
        # dx/dt = -x (exponential decay)
        t = np.linspace(0, 5, 100)
        x = np.exp(-t)
        data = x.reshape(-1, 1).tolist()
        resp = client.post("/analyze/sindy", json={
            "data": data,
            "time": t.tolist(),
            "threshold": 0.01,
            "library": "polynomial",
        })
        result = resp.json()
        assert result["n_samples"] == 100
        assert result["n_variables"] == 1
        assert len(result["discovered_equations"]) == 1

    # ── Visualization ──

    def test_viz_dos_no_data(self, client):
        resp = client.post("/viz/dos", json={})
        assert resp.json()["fallback"] is True

    def test_viz_dos_with_data(self, client):
        energies = np.linspace(-10, 10, 50).tolist()
        dos = np.exp(-np.linspace(-10, 10, 50) ** 2).tolist()
        resp = client.post("/viz/dos", json={
            "energies": energies,
            "dos": dos,
            "fermi_level": 0.0,
        })
        data = resp.json()
        assert data["fallback"] is False
        assert "total_states" in data

    def test_viz_phase_no_data(self, client):
        resp = client.post("/viz/phase", json={})
        assert resp.json()["fallback"] is True

    def test_viz_phase_with_data(self, client):
        t = np.linspace(0, 2 * np.pi, 50)
        resp = client.post("/viz/phase", json={
            "x": np.cos(t).tolist(),
            "y": np.sin(t).tolist(),
        })
        data = resp.json()
        assert data["fallback"] is False
        assert data["n_points"] == 50

    def test_viz_persistence_no_data(self, client):
        resp = client.post("/viz/persistence", json={})
        assert resp.json()["fallback"] is True

    def test_viz_persistence_with_data(self, client):
        diagram = [
            {"dimension": 0, "birth": 0.0, "death": 0.5},
            {"dimension": 0, "birth": 0.1, "death": 0.8},
            {"dimension": 1, "birth": 0.3, "death": 1.0},
        ]
        resp = client.post("/viz/persistence", json={"diagram": diagram})
        data = resp.json()
        assert data["fallback"] is False
        assert data["n_features"] == 3

    def test_viz_sindy_no_data(self, client):
        resp = client.post("/viz/sindy", json={})
        assert resp.json()["fallback"] is True

    def test_viz_sindy_with_data(self, client):
        resp = client.post("/viz/sindy", json={
            "equations": ["-1.0000*x0", "0.5000*x1"],
        })
        data = resp.json()
        assert data["fallback"] is False
        assert data["n_equations"] == 2
