"""Tests for XrdSimTool — powder XRD simulation and comparison."""

from __future__ import annotations

from pathlib import Path

import pytest

from huginn.tools.sci.xrd_sim_tool import XrdSimTool
from huginn.types import ToolContext

SI_POSCAR = str(Path(__file__).parent.parent / "Si_diamond" / "POSCAR")


@pytest.fixture
def tool():
    return XrdSimTool()


@pytest.fixture
def context(tmp_path):
    return ToolContext(session_id="test", workspace=str(tmp_path))


@pytest.mark.asyncio
class TestSimulateXrd:
    async def test_simulate_xrd_from_poscar(self, tool, context):
        pytest.importorskip("pymatgen", reason="pymatgen not installed")
        result = await tool.call(
            {"action": "simulate_xrd", "file_path": SI_POSCAR, "wavelength": 1.5406},
            context,
        )
        assert result.success
        peaks = result.data["peaks"]
        assert len(peaks) > 0
        # Si (Fd-3m, a=4.43) strongest reflection (111) near 2θ ≈ 35°
        two_theta_values = [p["two_theta"] for p in peaks]
        assert any(34.5 <= t <= 35.5 for t in two_theta_values)
        # each peak carries the expected keys
        for p in peaks:
            assert "two_theta" in p
            assert "intensity" in p
            assert "hkls" in p
        assert result.data["n_peaks"] == len(peaks)
        assert result.data["structure"] == "Si"

    async def test_simulate_xrd_requires_file_or_string(self, tool, context):
        result = await tool.call(
            {"action": "simulate_xrd", "wavelength": 1.5406}, context
        )
        assert not result.success
        # call() bypasses validate_input, so the load step fails directly
        assert "load" in result.error.lower() or "file_path" in result.error.lower()


@pytest.mark.asyncio
class TestParsePattern:
    async def test_parse_pattern_detects_peaks(self, tool, context, tmp_path):
        pytest.importorskip("scipy", reason="scipy not installed")
        # synthetic pattern: three Gaussian peaks on a flat baseline
        import numpy as np

        two_theta = np.linspace(10.0, 90.0, 801)
        intensity = (
            10.0
            + 100.0 * np.exp(-((two_theta - 28.4) ** 2) / 0.5)
            + 60.0 * np.exp(-((two_theta - 47.3) ** 2) / 0.5)
            + 40.0 * np.exp(-((two_theta - 56.1) ** 2) / 0.5)
        )
        csv = tmp_path / "pattern.csv"
        lines = ["# two_theta,intensity"]
        for t, y in zip(two_theta, intensity):
            lines.append(f"{t:.3f},{y:.3f}")
        csv.write_text("\n".join(lines), encoding="utf-8")

        result = await tool.call(
            {"action": "parse_pattern", "file_path": str(csv)}, context
        )
        assert result.success
        assert result.data["n_peaks"] >= 3
        peaks = result.data["peaks"]
        # the strongest peak near 28.4° should be detected
        assert any(28.0 <= p <= 29.0 for p in peaks)

    async def test_parse_pattern_missing_file(self, tool, context):
        result = await tool.call(
            {"action": "parse_pattern", "file_path": "nonexistent.csv"}, context
        )
        assert not result.success


@pytest.mark.asyncio
class TestComparePatterns:
    async def test_compare_patterns_with_observed_peaks(self, tool, context):
        # compare_patterns can run without pymatgen when given simulated_peaks
        # directly plus an observed peak list
        sim_peaks = [
            {"two_theta": 28.4, "intensity": 100.0, "hkls": [(1, 1, 1)]},
            {"two_theta": 47.3, "intensity": 60.0, "hkls": [(2, 2, 0)]},
            {"two_theta": 56.1, "intensity": 40.0, "hkls": [(3, 1, 1)]},
        ]
        observed = [28.45, 47.2, 56.3, 70.0]  # last one has no match
        result = await tool.call(
            {
                "action": "compare_patterns",
                "simulated_peaks": sim_peaks,
                "peaks": observed,
                "tolerance": 0.5,
            },
            context,
        )
        assert result.success
        assert result.data["n_matched"] == 3
        assert len(result.data["unmatched_experimental"]) == 1
        assert pytest.approx(result.data["overlap_ratio"], rel=1e-3) == (2 * 3) / (4 + 3)


@pytest.mark.asyncio
class TestIndexPeaks:
    async def test_index_peaks_matches_simulated(self, tool, context):
        pytest.importorskip("pymatgen", reason="pymatgen not installed")
        # First simulate to get real peak positions for Si
        sim = await tool.call(
            {"action": "simulate_xrd", "file_path": SI_POSCAR}, context
        )
        assert sim.success
        observed = [p["two_theta"] for p in sim.data["peaks"][:3]]

        result = await tool.call(
            {
                "action": "index_peaks",
                "file_path": SI_POSCAR,
                "peaks": observed,
                "tolerance": 0.3,
            },
            context,
        )
        assert result.success
        indexed = result.data["indexed_peaks"]
        assert len(indexed) == len(observed)
        # every observed peak should find a Miller index assignment
        assert result.data["n_indexed"] == len(observed)
        for entry in indexed:
            assert entry["hkl"] is not None
