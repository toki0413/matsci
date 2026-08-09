"""Tests for Born stability criteria — all 7 crystal systems.

Previously only cubic/hexagonal/orthorhombic/triclinic were implemented.
P2 fix added tetragonal, trigonal, monoclinic (Mouhat & Coudert PRB 2014).
"""

from __future__ import annotations

import numpy as np

from huginn.mechanics import BornStabilityChecker


def _make_stable_cubic():
    """A clearly stable cubic tensor."""
    return np.array([
        [250, 100, 100, 0, 0, 0],
        [100, 250, 100, 0, 0, 0],
        [100, 100, 250, 0, 0, 0],
        [0, 0, 0, 120, 0, 0],
        [0, 0, 0, 0, 120, 0],
        [0, 0, 0, 0, 0, 120],
    ], dtype=float)


def _make_unstable_cubic():
    """A clearly unstable cubic tensor (negative C44)."""
    C = _make_stable_cubic()
    C[3, 3] = -10
    C[4, 4] = -10
    C[5, 5] = -10
    return C


def _make_stable_tetragonal():
    """Stable tetragonal (C11=C22, C13=C23, C44=C55)."""
    return np.array([
        [300, 150, 100, 0, 0, 0],
        [150, 300, 100, 0, 0, 0],
        [100, 100, 400, 0, 0, 0],
        [0, 0, 0, 100, 0, 0],
        [0, 0, 0, 0, 100, 0],
        [0, 0, 0, 0, 0, 80],
    ], dtype=float)


def _make_unstable_tetragonal():
    """Unstable tetragonal (C11-C12 < 0)."""
    C = _make_stable_tetragonal()
    C[0, 1] = 350  # C12 > C11 → C11-C12 < 0
    C[1, 0] = 350
    return C


def _make_stable_trigonal():
    """Stable trigonal (7 independent constants)."""
    return np.array([
        [300, 100, 80, 5, 0, 0],
        [100, 300, 80, -5, 0, 0],
        [80, 80, 350, 0, 0, 0],
        [5, -5, 0, 80, 0, 0],
        [0, 0, 0, 0, 80, 0],
        [0, 0, 0, 0, 0, 100],
    ], dtype=float)


def _make_unstable_trigonal():
    """Unstable trigonal (C14 too large)."""
    C = _make_stable_trigonal()
    C[0, 3] = 200  # Huge C14 coupling
    C[3, 0] = 200
    C[1, 3] = -200
    C[3, 1] = -200
    return C


def _make_stable_monoclinic():
    """Stable monoclinic (13 independent constants, b-axis unique)."""
    return np.array([
        [250, 80, 60, 0, 10, 0],
        [80, 220, 50, 0, -5, 0],
        [60, 50, 280, 0, 8, 0],
        [0, 0, 0, 90, 0, 0],
        [10, -5, 8, 0, 70, 0],
        [0, 0, 0, 0, 0, 65],
    ], dtype=float)


def _make_unstable_monoclinic():
    """Unstable monoclinic (negative leading minor)."""
    C = _make_stable_monoclinic()
    C[0, 0] = -10  # C11 < 0 → M1 < 0
    return C


class TestCubic:
    def test_stable_cubic(self):
        r = BornStabilityChecker.check(_make_stable_cubic(), crystal_system="cubic")
        assert r["stable"] is True
        assert r["crystal_system"] == "cubic"
        assert len(r["criteria"]) == 3

    def test_unstable_cubic(self):
        r = BornStabilityChecker.check(_make_unstable_cubic(), crystal_system="cubic")
        assert r["stable"] is False


class TestHexagonal:
    def test_stable_hexagonal(self):
        C = np.array([
            [300, 100, 80, 0, 0, 0],
            [100, 300, 80, 0, 0, 0],
            [80, 80, 350, 0, 0, 0],
            [0, 0, 0, 100, 0, 0],
            [0, 0, 0, 0, 100, 0],
            [0, 0, 0, 0, 0, 100],
        ], dtype=float)
        r = BornStabilityChecker.check(C, crystal_system="hexagonal")
        assert r["stable"] is True
        assert r["crystal_system"] == "hexagonal"


class TestTetragonal:
    """P2 fix: tetragonal criteria newly implemented."""

    def test_stable_tetragonal(self):
        r = BornStabilityChecker.check(_make_stable_tetragonal(), crystal_system="tetragonal")
        assert r["stable"] is True
        assert r["crystal_system"] == "tetragonal"
        assert len(r["criteria"]) == 7

    def test_unstable_tetragonal(self):
        r = BornStabilityChecker.check(_make_unstable_tetragonal(), crystal_system="tetragonal")
        assert r["stable"] is False
        # C11 - C12 > 0 should fail
        failing = [c for c in r["criteria"] if not c["passed"]]
        assert any("C11 - C12" in c["name"] for c in failing)


class TestTrigonal:
    """P2 fix: trigonal criteria newly implemented."""

    def test_stable_trigonal(self):
        r = BornStabilityChecker.check(_make_stable_trigonal(), crystal_system="trigonal")
        assert r["stable"] is True
        assert r["crystal_system"] == "trigonal"
        assert len(r["criteria"]) == 8

    def test_unstable_trigonal(self):
        r = BornStabilityChecker.check(_make_unstable_trigonal(), crystal_system="trigonal")
        assert r["stable"] is False


class TestMonoclinic:
    """P2 fix: monoclinic criteria newly implemented."""

    def test_stable_monoclinic(self):
        r = BornStabilityChecker.check(_make_stable_monoclinic(), crystal_system="monoclinic")
        assert r["stable"] is True
        assert r["crystal_system"] == "monoclinic"
        assert len(r["criteria"]) == 6  # 6 leading principal minors

    def test_unstable_monoclinic(self):
        r = BornStabilityChecker.check(_make_unstable_monoclinic(), crystal_system="monoclinic")
        assert r["stable"] is False
        # M1 = C11 should be negative
        failing = [c for c in r["criteria"] if not c["passed"]]
        assert any("M1" in c["name"] for c in failing)


class TestOrthorhombic:
    def test_stable_orthorhombic(self):
        C = np.array([
            [250, 80, 60, 0, 0, 0],
            [80, 220, 50, 0, 0, 0],
            [60, 50, 280, 0, 0, 0],
            [0, 0, 0, 90, 0, 0],
            [0, 0, 0, 0, 70, 0],
            [0, 0, 0, 0, 0, 65],
        ], dtype=float)
        r = BornStabilityChecker.check(C, crystal_system="orthorhombic")
        assert r["stable"] is True


class TestTriclinic:
    def test_stable_triclinic(self):
        C = np.array([
            [250, 10, 10, 5, 5, 5],
            [10, 240, 8, 4, 3, 4],
            [10, 8, 260, 3, 6, 2],
            [5, 4, 3, 90, 2, 1],
            [5, 3, 6, 2, 80, 1],
            [5, 4, 2, 1, 1, 70],
        ], dtype=float)
        r = BornStabilityChecker.check(C, crystal_system="triclinic")
        assert r["stable"] is True
        assert len(r["criteria"]) == 6


class TestUnknownCrystalSystem:
    """Verify unimplemented crystal systems return stable=None, not False."""

    def test_unknown_system_returns_none(self):
        C = np.eye(6) * 100
        r = BornStabilityChecker.check(C, crystal_system="nonexistent")
        assert r["stable"] is None
        assert "not implemented" in r.get("error", "")


class TestAutoDetection:
    def test_auto_detect_cubic(self):
        r = BornStabilityChecker.check(_make_stable_cubic(), crystal_system="auto")
        assert r["crystal_system"] in ("cubic", "hexagonal", "triclinic")
